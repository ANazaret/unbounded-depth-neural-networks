import abc

import numpy as np
import scipy.stats as st
import torch
import torch.distributions as dist
import torch.utils.data
import torch.nn as nn


def softplus_inverse(x):
    """log(exp(x) - 1)"""
    return torch.where(x > 10, x, x.expm1().log())


class VariationalDepth(nn.Module, abc.ABC):
    """
    Abstract class for a variational posterior approximation q(L) over the depth L of the UDN

    Methods
    -------
    compute_depth()
        Returns the largest value of L with non-zero mass: max(i | q(L=i) > 0)
    probability_vector()
        Returns the vector of probabilities of q over integers up to its depth: [p(L=i) for i=0 to depth].
    mean()
        Returns the expectation of q: E[L] for L ~ q(L).
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def compute_depth(self):
        pass

    @abc.abstractmethod
    def probability_vector(self):
        pass

    @abc.abstractmethod
    def mean(self):
        pass


class FixedDepth(VariationalDepth):
    """
    Variational posterior approximation q(L) which is a constant mass at a given depth `n_layers`.
    Used to emulate a standard finite neural networks with `n_layers` layers.

    Parameters
    -------
    n_layers: int
        Depth such that q(L=n_layers) = 1

    """

    def __init__(self, n_layers: int):
        super().__init__()
        self.n_layers = n_layers

    def compute_depth(self):
        return self.n_layers

    def mean(self):
        return self.n_layers

    def probability_vector(self):
        v = torch.zeros(self.n_layers + 1, requires_grad=False)
        v[-1] = 1.0
        return v


class TruncatedPoisson(VariationalDepth):
    """
    Variational posterior approximation q(L) which is a Truncated Poisson.
    Used to adapt the depth during training.

    Parameters
    -------
    initial_nu_L: float, default 2.0
        Initial value of the variational parameter nu_L. nu_L is almost equal to the
        mean of the TruncatedPoisson, so the defaults 2.0 starts with two layers.

    truncation_quantile: float (between 0.0 and 1.0), default 0.95
        Truncation level of the Truncated Poisson, recommended to leave at 0.95

    """

    def __init__(self, initial_nu_L: float = 2.0, truncation_quantile: float = 0.95):
        super().__init__()
        self.truncation_quantile = truncation_quantile
        self._nu_L = nn.Parameter(softplus_inverse(torch.tensor(float(initial_nu_L))))

    @property
    def nu_L(self):
        """Returns the variational parameter nu_L, which is reparametrized to be positive."""
        return nn.Softplus()(self._nu_L)

    def compute_depth(self):
        p = st.poisson(self.nu_L.item())
        for a in range(int(self.nu_L.item()) + 1, 10000):
            if p.cdf(a) >= self.truncation_quantile:
                return a + 1
        raise Exception()

    def probability_vector(self):
        depth = self.compute_depth()
        ks = torch.arange(0, depth, dtype=self._nu_L.dtype, device=self._nu_L.device)
        alpha_L = (ks * self.nu_L.log() - torch.lgamma(ks + 1)).exp()
        alpha_L = torch.cat([torch.zeros(1, device=ks.device, dtype=ks.dtype), alpha_L])
        return alpha_L / alpha_L.sum()

    def mean(self):
        proba = self.probability_vector().cpu()
        return (proba * torch.arange(len(proba))).sum().item()


class CategoricalDUN(VariationalDepth):
    """
    Variational posterior approximation q(L) which is a categorical (=non parametric) distribution of fixed
    depth. It is equivalent to using a Depth Uncertainty Network [1].

    Parameters
    -------
    max_depth: int
        Depth of the categorical q(L).

     References
     ----------
     [1]

    """

    def __init__(self, max_depth):
        super().__init__()
        self.max_depth = max_depth
        self.logits = nn.Parameter(torch.ones(self.max_depth))

    def compute_depth(self):
        return self.max_depth

    def probability_vector(self):
        probs = nn.Softmax(dim=0)(self.logits)
        return torch.cat([torch.zeros(1, device=probs.device, dtype=probs.dtype), probs])

    def mean(self):
        proba = self.probability_vector().cpu()
        return (proba * torch.arange(len(proba))).sum().item()


# Shift a Poisson distribution by 1 so it only takes (strictly) positive values.
PositivePoisson = lambda p: torch.distributions.TransformedDistribution(
    torch.distributions.Poisson(p, validate_args=False),
    torch.distributions.AffineTransform(1, 1),
)


class UnboundedDepthNetwork(nn.Module):
    """
    Abstract class for a variational posterior approximation q(L) over the depth L of the UDN

    Parameters
    -------
    n_obs: int
        Number of observations that will e used for training. It is needed to scale prior when doing
        stochastic variational optimization.
    hidden_layer_generator: callable
        function that takes an integer L and return a torch.nn.Module representing hidden layer L.
    output_layer_generator: callable
        function that takes an integer L and return a torch.nn.Module representing output layer L.
    L_variational_distribution: VariationalDepth
        The variational distribution q(L)
    in_dimension: int
        Input dimension of the neural network
    out_dimension: int
        Output dimension of the neural network
    mode: str {"classification", "regression"}
        Specify if the neural network is for regression or for classification. It impacts the forward pass.
    L_prior_poisson: float
        Mean of the Poisson prior.
    theta_prior_scale: float
        Standard deviation (scale) of the Gaussian prior for the neural network weights.
    seed: int
        Random seed for the initialization of the neural networks layers.

    Methods
    -------
    set_optimizer()
        Set the optimizer to later add the dynamically created layers's parameters to it.
    set_device()
        Set the device of the model.
    update_depth()
        Compute the current maximal depth of the variational posterior q(L) and create new layers if needed.
    loss()
        Compute the loss (to minimize) of the UDN for the variational inference.
    elbo()
        Compute the ELBO (to maximize) of the UDN for the variational inference.
    forward

    """

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

        # set priors
        self.theta_prior_scale = theta_prior_scale
        self.prior_theta = dist.Normal(0, self.theta_prior_scale)
        self.prior_L = PositivePoisson(torch.tensor(float(L_prior_poisson)))
        if isinstance(self.variational_posterior_L, FixedDepth):
            # in this case the prior dosen't matter (since q is fixed), yet if q(L) is set to have a depth of 0
            # then the prior cannot be a PositivePoisson, as the posterior would have mass outside the prior
            # support. We set it to a regular Poisson just to avoid computation error.
            if self.variational_posterior_L.n_layers == 0:
                self.prior_L = torch.distributions.Poisson(torch.tensor(float(L_prior_poisson)))
        elif isinstance(self.variational_posterior_L, CategoricalDUN):
            # For the DUN, we set a uniform prior.
            d = self.variational_posterior_L.max_depth
            self.prior_L = torch.distributions.Categorical(torch.tensor([0.0] + [1 / d] * d))

        self.in_dimension = in_dimension
        self.out_dimension = out_dimension
        self.mode = mode

        # We generate only the first output layer at first (from the input directly to the output)
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

        # The DUN share the output layers. We won't add the additional output layers to the optimizer.
        if isinstance(self.variational_posterior_L, CategoricalDUN):
            self._add_output_layer_to_optimizer = False
        else:
            self._add_output_layer_to_optimizer = True

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        """Set the optimizer to later add the dynamically created layers's parameters to it."""
        self.optimizer = optimizer

    def set_device(self, device):
        """Set the device of the model."""
        self.to(device)
        self.device = device

    def update_depth(self):
        """
        Compute the current maximal depth of the variational posterior q(L) and create new layers if needed.
        """
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
        """Compute the loss (to minimize) of the UDN for the variational inference."""
        return -self.elbo(X, y)

    def elbo(self, X, y):
        """Compute the ELBO (to maximize) of the UDN for the variational inference."""
        res = self.forward(X, y)
        return sum(res["losses"]) + res["entropy_qL"]

    def forward(self, X, y=None):
        """
        Compute the neural network output, in a single forward pass.
        Returns a detailed description of the forward pass.
        If y is given, also computes the ELBO, otherwise, just computes the predictions.

        Returns
        -------
        predictions_global: array like of shape (X.shape[0], self.out_dimension)
            Posterior predictive expectation (averaged over the layers according to the posterior q(L))
        predictions_per_layer: list of array like of shape (X.shape[0], self.out_dimension)
            Posterior predictive expectation of each layer
        losses: list of scalar torch.Tensor
            Loss for each layer (computes the elbo for all the term related to each layer except the entropy of qL)
        entropy_qL: scalar torch.Tensor
            Entropy of qL
        logp_per_layer:
            Detailed access to layer specific ELBO terms, here the reconstruction.
        logp_L_per_layer:
            Detailed access to layer specific ELBO terms, here the prior regularization for L.
        logp_theta_per_layer:
            Detailed access to layer specific ELBO terms, here the prior regularization for L theta.
        logp_y_per_layer:
            Detailed access to layer specific ELBO terms, here the predictive likelihood per layer.

        """
        TRAIN_OUTPUT_LAYERS = True

        self.update_depth()
        variational_qL_probabilities = self.variational_posterior_L.probability_vector()

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
            a = variational_qL_probabilities[i]

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

        entropy_qL = torch.distributions.Categorical(variational_qL_probabilities).entropy()

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
