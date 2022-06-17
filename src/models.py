import abc
import time

import numpy as np
import pandas as pd
import scipy.stats as st
import torch
import torch.distributions as dist
import torch.utils.data
import tqdm
from torch import nn


def softplus_inverse(x):
    """log(exp(x) - 1)"""
    return torch.where(x > 10, x, x.expm1().log())


class VariationalDepth(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def compute_depth(self):
        pass

    @abc.abstractmethod
    def probability_vector(self, depth):
        """
        Return a vector of size depth+1
        [p_0, ..., p_depth]
        """
        pass

    @abc.abstractmethod
    def mean(self):
        pass


class FixedDepth(VariationalDepth):
    def __init__(self, n_layers):
        super().__init__()
        self.n_layers = n_layers

    def compute_depth(self):
        return self.n_layers

    def mean(self):
        return self.n_layers

    def probability_vector(self, depth):
        v = torch.zeros(depth + 1, requires_grad=False)
        v[-1] = 1.0
        return v


class TruncatedPoisson(VariationalDepth):
    def __init__(self, initial_L=2.0, truncation_quantile=0.95):
        super().__init__()
        self.truncation_quantile = truncation_quantile
        self._nu_L = nn.Parameter(softplus_inverse(torch.tensor(float(initial_L))))

    @property
    def nu_L(self):
        return nn.Softplus()(self._nu_L)

    def compute_depth(self):
        p = st.poisson(self.nu_L.item())
        for a in range(int(self.nu_L.item()) + 1, 10000):
            if p.cdf(a) >= self.truncation_quantile:
                return a + 1
        raise Exception()

    def probability_vector(self, depth):
        ks = torch.arange(0, depth, dtype=self._nu_L.dtype, device=self._nu_L.device)
        alpha_L = (ks * self.nu_L.log() - torch.lgamma(ks + 1)).exp()
        alpha_L = torch.cat([torch.zeros(1, device=ks.device, dtype=ks.dtype), alpha_L])
        return alpha_L / alpha_L.sum()

    def mean(self):
        proba = self.probability_vector(self.compute_depth()).cpu()
        return (proba * torch.arange(len(proba))).sum().item()


class CategoricalDUN(VariationalDepth):
    def __init__(self, max_depth):
        super().__init__()
        self.max_depth = max_depth
        self.logits = nn.Parameter(torch.ones(self.max_depth))

    def compute_depth(self):
        return self.max_depth

    def probability_vector(self, depth):
        probs = nn.Softmax(dim=0)(self.logits)
        return torch.cat([torch.zeros(1, device=probs.device, dtype=probs.dtype), probs])

    def mean(self):
        proba = self.probability_vector(self.compute_depth()).cpu()
        return (proba * torch.arange(len(proba))).sum().item()


PositivePoisson = lambda p: torch.distributions.TransformedDistribution(
    torch.distributions.Poisson(p, validate_args=False),
    torch.distributions.AffineTransform(1, 1),
)


class UnboundedNN(nn.Module):
    def __init__(
        self,
        n_obs: int,
        hidden_layer_generator,
        output_layer_generator,
        L_variational_distribution: VariationalDepth,
        in_dimension: int,
        out_dimension: int,
        mode: str = "classification",
        L_prior_poisson=1.0,
        theta_prior_scale=10.0,
        seed=0,
    ):
        super().__init__()
        torch.manual_seed(seed)

        self.n_obs = n_obs
        self.device = None
        self.optimizer = None

        self.hidden_layer_generator = hidden_layer_generator
        self.output_layer_generator = output_layer_generator

        self.variational_posterior_L = L_variational_distribution

        self.theta_prior_scale = theta_prior_scale
        self.prior_theta = dist.Normal(0, self.theta_prior_scale)

        self.prior_L = PositivePoisson(torch.tensor(float(L_prior_poisson)))
        if isinstance(self.variational_posterior_L, FixedDepth):
            if self.variational_posterior_L.n_layers == 0:
                self.prior_L = torch.distributions.Poisson(torch.tensor(float(L_prior_poisson)))
        elif isinstance(self.variational_posterior_L, CategoricalDUN):
            d = self.variational_posterior_L.max_depth
            self.prior_L = torch.distributions.Categorical(torch.tensor([0.0] + [1 / d] * d))

        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.mode = mode

        self.hidden_layers = nn.ModuleList([])
        self.output_layers = nn.ModuleList([self.output_layer_generator(-1, self.hidden_layer_generator)])

        self.current_depth = None
        # self.update_depth()

        if isinstance(self.variational_posterior_L, TruncatedPoisson):
            self.model_name = "UDN-inf"
        elif isinstance(self.variational_posterior_L, CategoricalDUN):
            self.model_name = "UDN-DUN%d" % self.variational_posterior_L.max_depth
        else:
            self.model_name = "UDN-f%d" % self.variational_posterior_L.n_layers

        self._add_output_layer_to_optimizer = True
        if isinstance(self.variational_posterior_L, CategoricalDUN):
            self._add_output_layer_to_optimizer = False

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_device(self, device):
        self.to(device)
        self.device = device

    def update_depth(self):
        self.current_depth = self.variational_posterior_L.compute_depth()
        while self.current_depth > len(self.hidden_layers):
            layer, *_ = self.hidden_layer_generator(len(self.hidden_layers))
            output_layer = self.output_layer_generator(len(self.hidden_layers), self.hidden_layer_generator)

            layer.to(self.device)
            output_layer.to(self.device)

            self.hidden_layers.append(layer)
            self.output_layers.append(output_layer)

            if self.optimizer is not None:
                self.optimizer.param_groups[0]["params"].extend(self.hidden_layers[-1].parameters())
                if self._add_output_layer_to_optimizer or len(self.output_layers) == 2:
                    self.optimizer.param_groups[0]["params"].extend(self.output_layers[-1].parameters())

    def loss(self, X, y):
        return -self.elbo(X, y)

    def elbo(self, X, y):
        res = self.forward(X, y)
        return sum(res["losses"]) + res["entropy_qL"]

    def forward(
        self,
        X,
        y=None,
    ):
        """
        Compute all the forward pass metrics, depending on if y is given
        """
        TRAIN_OUTPUT_LAYERS = True

        self.update_depth()
        alpha_L = self.variational_posterior_L.probability_vector(self.current_depth)

        intermediary_state_list = []
        output_state_list = []

        logp_theta_hidden_list = []
        logp_theta_output_list = []
        logp_theta_both_list = []
        logp_L_list = []
        logp_list = []
        logp_y_list = []
        predictions_per_layer = []
        losses = []

        log_theta_hidden_cumulative = torch.tensor(0.0, device=X.device)
        global_predictions = torch.zeros(X.shape[0], self.out_dimension, device=X.device, dtype=X.dtype)

        current_state = X
        i = 0
        while len(intermediary_state_list) - 1 < self.current_depth:
            a = alpha_L[i]

            if i > 0:
                hidden_layer = self.hidden_layers[i - 1]
                current_state = hidden_layer(current_state)
                # logp_theta_hidden = sum([self.prior_theta.log_prob(p).sum() for p in hidden_layer.parameters()])
                logp_theta_hidden = sum(
                    [-(p ** 2).sum() / 2 / self.theta_prior_scale ** 2 for p in hidden_layer.parameters()]
                )
            else:
                logp_theta_hidden = torch.tensor(0.0, device=X.device)

            output_layer = self.output_layers[i]
            # logp_theta_output = sum([self.prior_theta.log_prob(p).sum() for p in output_layer.parameters()])
            logp_theta_output = sum(
                [-(p ** 2).sum() / 2 / self.theta_prior_scale ** 2 for p in output_layer.parameters()]
            )
            logp_L = self.prior_L.log_prob(torch.tensor(i).float())

            log_theta_hidden_cumulative += logp_theta_hidden
            # log_theta_output_cumulative += logp_theta_output

            if a.item() > 0:
                current_output = output_layer(current_state)
            else:
                # No weight on this layer, we compute just for inspection; no gradient to propagate
                current_output = output_layer(current_state.detach())

            intermediary_state_list.append(current_state)
            output_state_list.append(current_output)
            logp_theta_hidden_list.append(log_theta_hidden_cumulative.item())
            logp_theta_output_list.append(logp_theta_output.item())
            logp_theta_both_list.append(log_theta_hidden_cumulative.item() + logp_theta_output.item())
            logp_L_list.append(logp_L.item())

            if y is not None:
                if self.mode == "classification":
                    logpy = -nn.CrossEntropyLoss(reduction="mean")(current_output, y) * self.n_obs
                elif self.mode == "regression":
                    # logpy = -nn.GaussianNLLLoss(reduction="mean")(current_output, y, torch.ones_like(y)) * self.n_obs
                    logpy = (-((current_output - y) ** 2).mean() / 2 - np.log(2 * np.pi) / 2) * self.n_obs
                else:
                    raise NotImplementedError
            else:
                logpy = torch.tensor(0.0, device=X.device)

            logp_y_list.append(logpy.item())

            if a.item() > 0:
                logp = logpy + log_theta_hidden_cumulative + logp_theta_output + logp_L
                losses.append(a * logp)
            else:
                if TRAIN_OUTPUT_LAYERS:
                    logp = logpy + logp_theta_output
                else:
                    logp = torch.tensor(0.0)
                losses.append(logp)

            logp_list.append(logp.item())

            if self.mode == "classification":
                current_predictions = nn.Softmax(dim=-1)(current_output)
            elif self.mode == "regression":
                current_predictions = current_output
            else:
                raise NotImplementedError

            predictions_per_layer.append(current_predictions)
            global_predictions = global_predictions + (a * current_predictions).detach()

            i += 1

        entropy_qL = torch.distributions.Categorical(alpha_L).entropy()

        return dict(
            predictions_global=global_predictions,
            predictions_per_layer=predictions_per_layer,
            losses=losses,
            entropy_qL=entropy_qL,
            logp_per_layer=logp_list,
            logp_L_per_layer=logp_L_list,
            logp_theta_per_layer=logp_theta_both_list,
            logp_y_per_layer=logp_y_list,
        )

    def predict_classification(self):
        assert self.mode == "classification"

