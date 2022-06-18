import time

import numpy as np
import pandas as pd
import torch
import tqdm

from src.models import UnboundedDepthNetwork


def train_one_epoch_classification(
    epoch,
    train_loader,
    valid_loader,
    test_loader,
    model: UnboundedDepthNetwork,
    optimizer,
    scheduler,
    PREFIX="./",
    normalize_loss=False,
):
    """Train the model for one epoch and log a lot of metrics."""
    train_loss_epoch, train_one_epoch_time = train(model, train_loader, optimizer, normalize_loss)
    (
        validation_accuracy,
        validation_predictive_loss,
        validation_accuracy_counts_per_layer,
        validation_predictions,
        validation_brier_score,
        validation_true_labels,
    ) = evaluate_classification(model, valid_loader)

    (
        test_accuracy,
        test_predictive_loss,
        test_accuracy_counts_per_layer,
        test_predictions,
        test_brier_score,
        test_true_labels,
    ) = evaluate_classification(model, test_loader)

    # i don't think the if is needed, model.variational_posterior_L.compute_depth() and .mean() should always work.
    if isinstance(model, UnboundedDepthNetwork):
        depth_max = model.current_depth
        depth_mean = model.variational_posterior_L.mean()
    else:
        depth_max = model.n_layers
        depth_mean = model.n_layers

    log_string = "Epoch: {}, Train Loss: {:.8f}, Test Accuracy: {:.8f}, Mean Post L: {:.2f}"
    print(log_string.format(epoch + 1, train_loss_epoch, test_accuracy, depth_mean))

    tmp = pd.read_csv(PREFIX + "tmp.%s.csv" % model.model_name, index_col=0)
    df_args = {
        "depth": [depth_max],
        "nu_L": depth_mean,
        "test_accuracy": test_accuracy,
        "validation_accuracy": validation_accuracy,
        "test_predictive_LL": test_predictive_loss,
        "validation_predictive_LL": validation_predictive_loss,
        "lr": scheduler.get_last_lr()[0],
        "test_brier": test_brier_score,
        "validation_brier": validation_brier_score,
        "train_time_one_epoch": train_one_epoch_time,
        "size_train": len(train_loader.sampler),
        "size_validation": len(valid_loader.sampler),
        "size_test": len(test_loader.sampler),
        "train_loss": train_loss_epoch,
    }

    for i, acc in enumerate(test_accuracy_counts_per_layer):
        df_args["test_accuracy_layer_%d" % (i)] = acc

    tmp = tmp.append(pd.DataFrame(df_args))
    tmp.to_csv(PREFIX + "tmp.%s.csv" % model.model_name)

    return test_accuracy


def train(model, train_loader, optimizer, normalize_loss=False):
    """
    Train the model for one epoch
    """
    train_loss_epoch = 0
    iterations = 0

    start_time = time.time()
    model.train()
    for features, target in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        features = features.to(model.device)
        target = target.to(model.device)
        loss = model.loss(features, target)
        if normalize_loss:
            loss = loss / model.n_obs
        train_loss_epoch += loss.item()

        loss.backward()
        optimizer.step()
        iterations += 1
    train_loss_epoch = train_loss_epoch / iterations
    train_one_epoch_time = time.time() - start_time

    return train_loss_epoch, train_one_epoch_time


def evaluate_classification(model, evaluation_loader):
    """Evaluate the model for classification."""
    accuracy_counts = 0
    accuracy_counts_per_layer = 0
    predictions = []
    true_labels = []
    predictive_loss = torch.tensor(0.0)
    brier_score = torch.tensor(0.0)

    model.eval()

    for features, labels in evaluation_loader:
        features = features.to(model.device)
        labels = labels.cpu()
        forward_pass = model(features)
        pred = forward_pass["predictions_global"].detach().cpu()

        accuracy_counts_per_layer_batch = np.array(
            [
                (torch.max(p.detach().cpu(), dim=1).indices == labels).sum().item()
                for p in forward_pass["predictions_per_layer"]
            ]
        )
        accuracy_counts_per_layer += accuracy_counts_per_layer_batch
        predictions.append(pred)
        true_labels.append(labels)
        predictive_loss += torch.gather(pred, 1, labels.view(-1, 1)).log().sum()
        brier_score += (pred.pow(2).sum() + (1 - 2 * torch.gather(pred, 1, labels.view(-1, 1))).sum()).item()
        accuracy_counts += (torch.max(pred, dim=1).indices == labels).sum().item()

    if len(evaluation_loader.sampler):
        accuracy = accuracy_counts / len(evaluation_loader.sampler)
        accuracy_counts_per_layer = accuracy_counts_per_layer / len(evaluation_loader.sampler)
        predictions = torch.cat(predictions, dim=0).numpy()
        true_labels = torch.cat(true_labels, dim=0).numpy()
    else:
        accuracy = 0
        accuracy_counts_per_layer = 0

    predictive_loss = predictive_loss.item()
    brier_score = brier_score.item()

    return accuracy, predictive_loss, accuracy_counts_per_layer, predictions, brier_score, true_labels


def train_one_epoch_regression(
    epoch,
    train_loader,
    valid_loader,
    test_loader,
    model: UnboundedDepthNetwork,
    optimizer,
    scheduler,
    PREFIX="./",
    normalize_loss=False,
):
    train_loss_epoch, train_one_epoch_time = train(model, train_loader, optimizer, normalize_loss)
    validation_predictive_loss = evaluate_regression(model, valid_loader)
    test_predictive_loss = evaluate_regression(model, test_loader)

    # i don't think the if is needed, model.variational_posterior_L.compute_depth() and .mean() should always work.
    if isinstance(model, UnboundedDepthNetwork):
        depth_max = model.current_depth
        depth_mean = model.variational_posterior_L.mean()
    else:
        depth_max = model.n_layers
        depth_mean = model.n_layers
    print(
        "Epoch: {}, Train Loss: {:.8f}, Val RMSE: {:.8f}, Mean Post L: {:.2f}".format(
            epoch + 1, train_loss_epoch, validation_predictive_loss, depth_mean
        ),
    )

    tmp = pd.read_csv(PREFIX + "tmp.%s.csv" % model.model_name, index_col=0)
    df_args = {
        "depth": [depth_max],
        "nu_L": depth_mean,
        "test_rmse": test_predictive_loss,
        "validation_rmse": validation_predictive_loss,
        "lr": scheduler.get_last_lr()[0],
        "train_time_one_epoch": train_one_epoch_time,
        "size_train": len(train_loader.sampler),
        "size_validation": len(valid_loader.sampler),
        "size_test": len(test_loader.sampler),
        "train_loss": train_loss_epoch,
    }

    tmp = tmp.append(pd.DataFrame(df_args))
    tmp.to_csv(PREFIX + "tmp.%s.csv" % model.model_name)

    return test_predictive_loss


def evaluate_regression(model, evaluation_loader):
    predictive_loss = torch.tensor(0.0)
    for features, labels in evaluation_loader:
        features = features.to(model.device)
        labels = labels.cpu()
        pred = model(features)["predictions_global"].detach().cpu()
        predictive_loss += ((pred - labels) ** 2).sum()

    if len(evaluation_loader.sampler):
        predictive_loss = (predictive_loss.item() / len(evaluation_loader.sampler)) ** 0.5

    return predictive_loss
