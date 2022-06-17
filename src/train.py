import time

import numpy as np
import pandas as pd
import torch
import tqdm

from src.models import UnboundedNN


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
        validation_brier_score += (
            pred.pow(2).sum() + (1 - 2 * torch.gather(pred, 1, labels.view(-1, 1))).sum()
        ).item()
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
            [
                (torch.max(p.detach().cpu(), dim=1).indices == labels).sum().item()
                for p in tmp["predictions_per_layer"]
            ]
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


def train_one_epoch_regression(
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
        validation_predictive_loss += ((pred - labels) ** 2).sum()
        # validation_brier_score += (pred.pow(2).sum() + (1 - 2 * torch.gather(pred, 1, labels.view(-1, 1))).sum()).item()
        # accuracy_counts += (torch.max(pred, dim=1).indices == labels).sum().item()

    if len(valid_loader.sampler):
        validation_predictive_loss = validation_predictive_loss / len(valid_loader.sampler)
    else:
        validation_accuracy = 0
    validation_predictive_loss = validation_predictive_loss.item() ** 0.5

    # ############### TEST  ##################
    accuracy_counts = 0
    accuracy_counts_per_layer = 0
    predictions = []
    true_labels = []
    test_predictive_loss = torch.tensor(0.0)
    brier_score = 0
    for features, labels in test_loader:
        features = features.to(model.device)
        labels = labels.cpu()
        pred = model(features)["predictions_global"].detach().cpu()
        test_predictive_loss += ((pred - labels) ** 2).sum()

    test_predictive_loss = (test_predictive_loss.item() / len(test_loader.sampler)) ** 0.5

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
    # tmp.to_csv(PREFIX + "tmp2.%s.csv" % model.model_name)
    tmp.to_csv(PREFIX + "tmp.%s.csv" % model.model_name)

    return test_predictive_loss
