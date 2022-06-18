import argparse
import os
import time

import pandas as pd
import torch.optim as optim
import torch.utils.data

from experiments.layer_generators import make_generators_fcn, make_generators_fcn_DUN
from experiments.load_data import load_data_spiral
from src.models import (
    UnboundedDepthNetwork,
    TruncatedPoisson,
    FixedDepth,
    CategoricalDUN,
)
from src.train import train_one_epoch_classification

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

    # CREATE MODEL
    if DUN:
        generator_layers, generator_residual = make_generators_fcn_DUN(8, 2, 2)
    else:
        generator_layers, generator_residual = make_generators_fcn(8, 2, 2)

    # Negative L means UDN (TruncatedPoisson); non-negative L means either standard NN (FixedDepth) or UDN (Categorical)
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

    model = UnboundedDepthNetwork(
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

    model.model_name += ".spiral.v2.O-%d.seed-%d.lr%.4f" % (spiral_R, SEED, LR)
    model.set_device(device)

    # reduce the LR for he variational posterior qL
    optimizer = optim.Adam(
        [
            {"params": [p for n, p in model.named_parameters() if n != "variational_posterior_L._nu_L"]},
            {
                "params": [p for n, p in model.named_parameters() if n == "variational_posterior_L._nu_L"],
                "lr": LR / 10,
            },
        ],
        lr=LR,
    )
    # optimizer = optim.Adam(model.parameters(), lr=LR)

    STEP = 100
    # scheduler is useless here, we use a schedule for cifar only
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.6)
    model.set_optimizer(optimizer)
    PREFIX = "log-spiral/"
    os.makedirs(PREFIX, exist_ok=True)

    tmp = pd.DataFrame({"depth": [], "nu_L": [], "test_acc": []})
    tmp.to_csv(PREFIX + "tmp.%s.csv" % model.model_name)

    for epoch in range(args.epochs):
        start_time = time.time()
        test_accuracy = train_one_epoch_classification(
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
