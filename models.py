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
    torch.distributions.Poisson(p, validate_args=False), torch.distributions.AffineTransform(1, 1)
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
                logpy = -nn.CrossEntropyLoss(reduction="mean")(current_output, y) * self.n_obs
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

            current_predictions = nn.Softmax(dim=-1)(current_output)
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


def SCE(predictions, labels, n_bins=10, n_labels=10):
    """Bin k,b contains prediction of label k w probability in [b/B, (b+1)/B]"""
    confidence = np.zeros((n_bins, n_labels))
    accuracy = np.zeros((n_bins, n_labels))
    nbk = np.zeros((n_bins, n_labels))

    for b, (bl, br) in enumerate(zip(np.linspace(0, 1.0001, n_bins + 1), np.linspace(0, 1.0001, n_bins + 1)[1:])):
        for k in range(n_labels):
            indices = ((predictions[:, k] < br) & (predictions[:, k] >= bl)).nonzero()[0]
            nbk[b, k] = len(indices)
            confidence[b, k] = predictions[indices, k].mean()
            accuracy[b, k] = np.mean(labels[indices] == k)
    np.nan_to_num(confidence, copy=False)
    np.nan_to_num(accuracy, copy=False)

    return (np.abs(confidence - accuracy) * nbk).sum() / (n_labels * len(labels))


def ECE(
    predictions,
    labels,
    n_bins=10,
):
    """Bin k,b contains prediction of label k w probability in [b/B, (b+1)/B]"""
    confidence = np.zeros((n_bins,))
    accuracy = np.zeros((n_bins,))
    nb = np.zeros((n_bins,))

    predictions_label = np.argmax(predictions, axis=1)
    predictions_max = np.max(predictions, axis=1)

    for b, (bl, br) in enumerate(zip(np.linspace(0, 1.0001, n_bins + 1), np.linspace(0, 1.0001, n_bins + 1)[1:])):
        indices = ((predictions_max < br) & (predictions_max >= bl)).nonzero()[0]
        nb[b] = len(indices)
        confidence[b] = predictions_max[indices].mean()
        accuracy[b] = np.mean(labels[indices] == predictions_label[indices])
    np.nan_to_num(confidence, copy=False)
    np.nan_to_num(accuracy, copy=False)

    assert sum(nb) == len(labels)

    return (np.abs(confidence - accuracy) * nb).sum() / (len(labels))


def ACE(
    predictions,
    labels,
    n_bins=10,
):
    """Bin k,b contains prediction of label k w probability in [b/B, (b+1)/B]"""
    confidence = np.zeros((n_bins,))
    accuracy = np.zeros((n_bins,))
    nb = np.zeros((n_bins,))

    predictions_label = np.argmax(predictions, axis=1)
    predictions_max = np.max(predictions, axis=1)

    n_bins = min(n_bins, len(np.unique(predictions_max)))

    bins = np.quantile(predictions_max, np.linspace(0, 1, n_bins + 1))
    bins[-1] += 0.0001
    for b, (bl, br) in enumerate(zip(bins, bins[1:])):
        indices = ((predictions_max < br) & (predictions_max >= bl)).nonzero()[0]
        nb[b] = len(indices)
        confidence[b] = predictions_max[indices].mean()
        accuracy[b] = np.mean(labels[indices] == predictions_label[indices])
    np.nan_to_num(confidence, copy=False)
    np.nan_to_num(accuracy, copy=False)

    return (np.abs(confidence - accuracy) * nb).sum() / (len(labels))


def train_one_epoch(
    epoch,
    train_loader,
    valid_loader,
    test_loader,
    model: UnboundedNN,
    optimizer,
    scheduler,
    PREFIX="./",
    normalize_loss=False,
):
    # ############### TRAINING  ##################
    train_loss_epoch = 0
    iterations = 0

    start_time = time.time()
    model.train()
    for features, labels in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        features = features.to(model.device)
        labels = labels.to(model.device)
        loss = model.loss(features, labels)
        if normalize_loss:
            loss = loss / model.n_obs
        train_loss_epoch += loss.item()

        loss.backward()
        optimizer.step()
        iterations += 1
    train_loss_epoch = train_loss_epoch / iterations
    train_one_epoch_time = time.time() - start_time

    if isinstance(model, UnboundedNN):
        depth_max = model.current_depth
        depth_mean = model.variational_posterior_L.mean()
    else:
        depth_max = model.n_layers
        depth_mean = model.n_layers

    # ############### VALIDATION  ##################
    accuracy_counts = 0
    model.eval()
    validation_predictive_loss = torch.tensor(0.0)
    validation_brier_score = torch.tensor(0.0)
    for features, labels in valid_loader:
        features = features.to(model.device)
        labels = labels.cpu()
        pred = model(features)["predictions_global"].detach().cpu()
        validation_predictive_loss += torch.gather(pred, 1, labels.view(-1, 1)).log().sum()
        validation_brier_score += (pred.pow(2).sum() + (1 - 2 * torch.gather(pred, 1, labels.view(-1, 1))).sum()).item()
        accuracy_counts += (torch.max(pred, dim=1).indices == labels).sum().item()

    if len(valid_loader.sampler):
        validation_accuracy = accuracy_counts / len(valid_loader.sampler)
    else:
        validation_accuracy = 0
    validation_predictive_loss = validation_predictive_loss.item()

    # ############### TEST  ##################
    accuracy_counts = 0
    accuracy_counts_per_layer = 0
    predictions = []
    true_labels = []
    test_predictive_loss = 0
    brier_score = 0
    for features, labels in test_loader:
        features = features.to(model.device)
        labels = labels.cpu()
        tmp = model(features)
        pred = tmp["predictions_global"].detach().cpu()
        accuracy_counts_per_layer_batch = np.array(
            [(torch.max(p.detach().cpu(), dim=1).indices == labels).sum().item() for p in tmp["predictions_per_layer"]]
        )
        accuracy_counts_per_layer += accuracy_counts_per_layer_batch
        predictions.append(pred)
        true_labels.append(labels)
        test_predictive_loss += torch.gather(pred, 1, labels.view(-1, 1)).log().sum()
        brier_score += (pred.pow(2).sum() + (1 - 2 * torch.gather(pred, 1, labels.view(-1, 1))).sum()).item()
        accuracy_counts += (torch.max(pred, dim=1).indices == labels).sum().item()

    accuracy_counts_per_layer = accuracy_counts_per_layer / len(test_loader.sampler)
    accuracy = accuracy_counts / len(test_loader.sampler)
    predictions = torch.cat(predictions, dim=0).numpy()
    true_labels = torch.cat(true_labels, dim=0).numpy()
    sce = SCE(predictions, true_labels, 10, model.out_dimension)
    ace = ACE(predictions, true_labels, 20)
    ece = ECE(predictions, true_labels, 20)
    test_predictive_loss = test_predictive_loss.item()

    print(
        "Epoch: {}, Train Loss: {:.8f}, Val Accuracy: {:.8f}, Mean Post L: {:.2f}".format(
            epoch + 1, train_loss_epoch, accuracy, depth_mean
        ),
    )

    tmp = pd.read_csv(PREFIX + "tmp.%s.csv" % model.model_name, index_col=0)
    df_args = {
        "depth": [depth_max],
        "nu_L": depth_mean,
        "test_accuracy": accuracy,
        "validation_accuracy": validation_accuracy,
        "test_sce": sce,
        "test_ace": ace,
        "test_ece": ece,
        "test_predictive_LL": test_predictive_loss,
        "validation_predictive_LL": validation_predictive_loss,
        "lr": scheduler.get_last_lr()[0],
        "test_brier": brier_score,
        "validation_brier": validation_brier_score,
        "train_time_one_epoch": train_one_epoch_time,
        "size_train": len(train_loader.sampler),
        "size_validation": len(valid_loader.sampler),
        "size_test": len(test_loader.sampler),
        "train_loss": train_loss_epoch,
    }

    for i, acc in enumerate(accuracy_counts_per_layer):
        df_args["test_accuracy_layer_%d" % (i)] = acc

    tmp = tmp.append(pd.DataFrame(df_args))
    tmp.to_csv(PREFIX + "tmp2.%s.csv" % model.model_name)
    tmp.to_csv(PREFIX + "tmp.%s.csv" % model.model_name)

    return accuracy
