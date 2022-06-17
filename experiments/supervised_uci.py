import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
from torch import nn

from models import (
    train_one_epoch,
    UnboundedNN,
    TruncatedPoisson,
    FixedDepth,
    CategoricalDUN,
    train_one_epoch_regression,
)
from UCI_gap_loader import load_gap_UCI


def load_data(dataset_name, n_split, batchsize, seed=0):
    X_train, X_test, x_means, x_stds, y_train, y_test, y_means, y_stds = load_gap_UCI(
        "data",
        dataset_name,
        n_split,
        gap=False,
    )

    torch.manual_seed(seed)
    print(y_stds)

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

    return train_loader, valid_loader, test_loader, x_means.shape[0]


def load_data2(dataset_name, n_split, batchsize, seed=0):
    base_dir = "data"
    dir_load = base_dir + "/UCI_for_sharing/standard/" + dataset_name + "/data/"

    data = np.loadtxt(dir_load + "data.txt")
    feature_idx = np.loadtxt(dir_load + "index_features.txt").astype(int)
    target_idx = np.loadtxt(dir_load + "index_target.txt").astype(int)

    np.random.seed(n_split)
    indices = np.array(list(range(data.shape[0])))
    np.random.shuffle(indices)
    train_end, val_end = int(len(indices) * 0.8), int(len(indices) * 0.9)
    train_idx = indices[:train_end]
    validation_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    np.random.seed(0)

    data_train = data[train_idx]
    data_validation = data[validation_idx]
    data_test = data[test_idx]

    X_train = data_train[:, feature_idx].astype(np.float32)
    X_test = data_test[:, feature_idx].astype(np.float32)
    X_validation = data_validation[:, feature_idx].astype(np.float32)
    y_train = data_train[:, target_idx].astype(np.float32)
    y_test = data_test[:, target_idx].astype(np.float32)
    y_validation = data_validation[:, target_idx].astype(np.float32)

    x_means, x_stds = X_train.mean(axis=0), X_train.std(axis=0)
    y_means, y_stds = y_train.mean(axis=0), y_train.std(axis=0)

    x_stds[x_stds < 1e-10] = 1.0

    X_train = (X_train - x_means) / x_stds
    y_train = ((y_train - y_means) / y_stds)[:, np.newaxis]
    X_test = (X_test - x_means) / x_stds
    y_test = ((y_test - y_means) / y_stds)[:, np.newaxis]
    X_validation = (X_validation - x_means) / x_stds
    y_validation = ((y_validation - y_means) / y_stds)[:, np.newaxis]

    torch.manual_seed(seed)
    print(y_stds)

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_validation), torch.FloatTensor(y_validation)
    )
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

    return train_loader, valid_loader, test_loader, x_means.shape[0]


def make_generator_uci(hidden_size, input_size, output_size):
    def main_layers(layer_id):
        if layer_id == -1:
            return None, input_size

        if layer_id == 0:
            return (
                nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU()),
                hidden_size,
            )

        return (
            nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()),
            hidden_size,
        )

    def output_layers(layer_id, main_layers):
        _, s = main_layers(layer_id)
        return nn.Linear(s, output_size)

    return main_layers, output_layers


def make_generator_uci_DUN(hidden_size, input_size, output_size):
    def main_layers(layer_id):
        if layer_id == -1:
            return None, input_size

        if layer_id == 0:
            return (
                nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU()),
                hidden_size,
            )

        return (
            nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()),
            hidden_size,
        )

    o_layer = nn.Linear(input_size, output_size)
    final_layer = nn.Linear(hidden_size, output_size)

    def output_layers(layer_id, main_layers):
        if layer_id >= 0:
            return final_layer
        else:
            return o_layer

    return main_layers, output_layers


OUTPUT_SIZE = 1
BATCH_SIZE = 256
CUDA = False
DEVICE = torch.device("cuda" if CUDA and torch.cuda.is_available() else "cpu")
print(DEVICE)

DATASET_NAMES = [
    "boston",
    "concrete",
    "energy",
    "power",
    "wine",
    "yacht",
    "kin8nm",
    "naval",
    "protein",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-e", "--epochs", default=1000, type=int, help="number of epochs")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("-l", "--layers", default=-1, help="number of layers", type=int)
    parser.add_argument("-s", "--seed", default=0, help="random seed", type=int)
    parser.add_argument("--dataset", choices=DATASET_NAMES, help="spiral", type=str)
    parser.add_argument("-d", "--dun", action="store_true")
    parser.add_argument("-i", "--init", default=1.0, type=float, help="truncated poisson init")
    parser.add_argument("-p", "--prior", default=1.0, type=float, help=" poisson prior")
    parser.add_argument("--split", default=0, type=int, help=" split_id")

    args = parser.parse_args()

    SEED = args.seed
    L = args.layers
    LR = args.lr
    DUN = args.dun

    torch.manual_seed(SEED)
    device = DEVICE

    # LOAD DATA
    train_loader, valid_loader, test_loader, INPUT_SIZE = load_data2(
        args.dataset, args.split, BATCH_SIZE, SEED
    )
    N_train = len(train_loader.sampler)
    # N_test = len(test_loader.sampler)

    # CREATE MODEL
    if DUN:
        generator_layers, generator_residual = make_generator_uci_DUN(8, INPUT_SIZE, 1)
    else:
        generator_layers, generator_residual = make_generator_uci(8, INPUT_SIZE, 1)

    model_name = ""

    if L < 0:
        if DUN:
            raise ValueError
        else:
            vpost = TruncatedPoisson(1.0)

    else:
        if DUN:
            vpost = CategoricalDUN(L)
        else:
            vpost = FixedDepth(L)

    model = UnboundedNN(
        N_train,
        generator_layers,
        generator_residual,
        vpost,
        INPUT_SIZE,
        OUTPUT_SIZE,
        L_prior_poisson=0.5,
        theta_prior_scale=1.0,
        seed=SEED,
        mode="regression",
    )
    # V2: prior 0.1

    model.model_name += ".uci.%s-split%d.v1.O.seed-%d.lr%.4f" % (
        args.dataset,
        args.split,
        SEED,
        LR,
    )

    model.set_device(device)

    optimizer = optim.Adam(
        [
            {"params": [p for n, p in model.named_parameters() if n != "variational_posterior_L._nu_L"]},
            {
                "params": [p for n, p in model.named_parameters() if n == "variational_posterior_L._nu_L"],
                "lr": LR / 10,
            },
        ],
        lr=LR,
    )  # 0.02?
    # optimizer = optim.Adam(model.parameters(), lr=LR)  # 0.02?

    STEP = 100
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.6)
    model.set_optimizer(optimizer)
    PREFIX = "log-uci-5/"
    import os

    os.makedirs(PREFIX, exist_ok=True)

    tmp = pd.DataFrame({"depth": [], "nu_L": [], "test_acc": []})
    tmp.to_csv(PREFIX + "tmp.%s.csv" % model.model_name)

    for epoch in range(args.epochs):
        start_time = time.time()
        test_accuracy = train_one_epoch_regression(
            epoch,
            train_loader,
            valid_loader,
            test_loader,
            model,
            optimizer,
            scheduler,
            PREFIX=PREFIX,
        )
        scheduler.step()
        print(time.time() - start_time)

    # torch.save(model.state_dict(), PREFIX + "model.%s.pth" % model.model_name)
