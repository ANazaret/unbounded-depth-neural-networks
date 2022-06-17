import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
from torch import nn

from src.models import (
    train_one_epoch,
    UnboundedNN,
    TruncatedPoisson,
    FixedDepth,
    CategoricalDUN,
)


def generate_spiral(phase: float, n: int = 1000, seed: int = 0):
    """Generate a spiral dataset

    Parameters
    ----------
    phase: float
        Phase of the spiral.
    n: int
        Number of samples.
    seed: int
        Random seed.

    Returns
    -------
    labels: list of int
        Labels of each each generated point, corresponding to which arm of the spiral the point is from.
    xy: array like
        Coordinates of the generated points

    """
    omega = lambda x: phase * np.pi / 2 * np.abs(x)
    rng = np.random.default_rng(seed)
    ts = rng.uniform(-1, 1, n)
    ts = np.sign(ts) * np.sqrt(np.abs(ts))
    xy = np.array([ts * np.cos(omega(ts)), ts * np.sin(omega(ts))]).T
    xy = rng.normal(xy, 0.02)
    labels = (ts >= 0).astype(int)
    return labels, xy


def load_data_spiral(phase: float, batch_size:  int, seed:int=0):
    """Build the dataloaders for the spiral dataset.
    The spiral datasets are generated and then wrapped in a dataloader.

    Parameters
    ----------
    phase: float
        Phase of the spiral.
    batch_size: int
        Batch size of the dataloaders.
    seed: int
        Random seed to use to generate the spiral data.

    Returns
    -------
    train_loader: DataLoader
        DataLoader for the training points of the spiral dataset.
    valid_loader: DataLoader
        DataLoader for the validation points of the spiral dataset.
    train_loader: DataLoader
        test_loader for the testing points of the spiral dataset.

    """
    t, xy = generate_spiral(phase, n=1024, seed=0 + seed)
    t_val, xy_val = generate_spiral(phase, n=1024, seed=1 + seed)
    t_test, xy_test = generate_spiral(phase, n=1024, seed=2 + seed)

    torch.manual_seed(seed)

    train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(xy), torch.LongTensor(t))
    val_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(xy_val), torch.LongTensor(t_val))
    test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(xy_test), torch.LongTensor(t_test))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def make_generator_spiral(hidden_size, input_size, output_size):
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


def make_generator_spiral_DUN(hidden_size, input_size, output_size):
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


INPUT_SIZE = 2
OUTPUT_SIZE = 2
BATCH_SIZE = 256
CUDA = False
DEVICE = torch.device("cuda" if CUDA and torch.cuda.is_available() else "cpu")
print(DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-e", "--epochs", default=1000, type=int, help="number of epochs")
    parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
    parser.add_argument("-l", "--layers", default=-1, help="number of layers", type=int)
    parser.add_argument("-s", "--seed", default=0, help="random seed", type=int)
    parser.add_argument("-r", "--spiral", default=1, help="spiral", type=int)
    parser.add_argument("-d", "--dun", action="store_true")
    parser.add_argument("-i", "--init", default=1.0, type=float, help="truncated poisson init")
    parser.add_argument("-p", "--prior", default=1.0, type=float, help=" poisson prior")
    args = parser.parse_args()

    SEED = args.seed
    spiral_R = args.spiral
    L = args.layers
    LR = args.lr
    DUN = args.dun

    torch.manual_seed(SEED)
    device = DEVICE

    # LOAD DATA
    train_loader, valid_loader, test_loader = load_data_spiral(spiral_R, BATCH_SIZE, SEED)
    N_train = len(train_loader.sampler)
    # N_test = len(test_loader.sampler)

    # CREATE MODEL
    if DUN:
        generator_layers, generator_residual = make_generator_spiral_DUN(8, 2, 2)
    else:
        generator_layers, generator_residual = make_generator_spiral(8, 2, 2)

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
    )
    # V2: prior 0.1

    model.model_name += ".spiral.v2.O-%d.seed-%d.lr%.4f" % (spiral_R, SEED, LR)

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
    PREFIX = "log-spiral-3/"
    import os

    os.makedirs(PREFIX, exist_ok=True)

    tmp = pd.DataFrame({"depth": [], "nu_L": [], "test_acc": []})
    tmp.to_csv(PREFIX + "tmp.%s.csv" % model.model_name)

    for epoch in range(args.epochs):
        start_time = time.time()
        test_accuracy = train_one_epoch(
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
