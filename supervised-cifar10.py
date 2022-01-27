import argparse
import time
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
import tqdm
from torch import nn
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms

from models import train_one_epoch, UnboundedNN, TruncatedPoisson, FixedDepth

INPUT_SIZE = 784
OUTPUT_SIZE = 10
BATCH_SIZE = 256

CUDA = True
DEVICE = torch.device("cuda" if CUDA and torch.cuda.is_available() else "cpu")

print(DEVICE)


def load_data(batch_size, seed=0, validation_size=0.2, filter_labels=None):
    """
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)

    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if filter_labels is not None:
        train_val_indices = [i for i, v in enumerate(train_set.targets) if v in filter_labels]
    else:
        train_val_indices = np.arange(len(train_set))
    np.random.shuffle(train_val_indices)
    split = int(np.floor(validation_size * len(train_val_indices)))
    train_idx, valid_idx = train_val_indices[split:], train_val_indices[:split]

    if filter_labels is not None:
        test_idx = [i for i, v in enumerate(test_set.targets) if v in filter_labels]
    else:
        test_idx = np.arange(len(test_set))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    if os.environ["HOME"] == "/Users/achille":
        # should be my laptop
        print("Local laptop")
        num_workers = 0
    elif os.environ["HOME"] == "/home/achille":
        # Should be the cluster
        num_workers = 2
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=test_sampler,
    )

    return train_loader, valid_loader, test_loader


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, residual=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.is_residual = residual
        self.shortcut = nn.Sequential()
        if (stride != 1 or in_channels != self.expansion * out_channels) and self.is_residual:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.is_residual:
            out += self.shortcut(x)
        out = F.relu(out)
        return out


def generator_layers(L, is_residual=True):
    if L == -1:
        return None, 3, 32

    FIRST_LAYER_CHANNEL = 64
    out_dim = 32
    if L == 0:
        bloc = [
            nn.Conv2d(3, FIRST_LAYER_CHANNEL, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(FIRST_LAYER_CHANNEL),
            nn.ReLU(),
        ]
        return nn.Sequential(*bloc), FIRST_LAYER_CHANNEL, out_dim

    in_channel = FIRST_LAYER_CHANNEL
    total = 0
    # for block_id in , 4, 6, 100][block_id]
    for block_id in range(1, 4):
        n_of_blocks = [0, 3, 5, 1000][block_id]
        channels_of_block = 2 ** (block_id + 5)

        for i in range(n_of_blocks):
            total += 1
            if i == 0:
                stride = 2
            else:
                stride = 1

            out_channel = channels_of_block * Bottleneck.expansion
            out_dim //= stride

            if total == L:
                bloc = Bottleneck(in_channel, channels_of_block, stride, is_residual)
                return bloc, out_channel, out_dim

            in_channel = out_channel


def generator_output(layer_id, generator_layers):
    _, last_channels, last_dim = generator_layers(layer_id)
    last_hidden_size = (last_dim // 4) ** 2 * last_channels
    layers = [
        nn.AvgPool2d(4),
        nn.Flatten(),
        nn.Linear(last_hidden_size, 10),
    ]

    return nn.Sequential(*layers)


from torch.optim.lr_scheduler import _LRScheduler


class ExplicitLR(_LRScheduler):
    """"""

    def __init__(self, optimizer, lrs, last_epoch=-1, verbose=False):
        self.lrs = lrs
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch >= len(self.lrs):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self.lrs[self.last_epoch] for _ in self.optimizer.param_groups]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-e", "--epochs", default=1000, type=int, help="number of epochs")
    parser.add_argument("-l", "--layers", default=-1, help="number of layers", type=int)
    parser.add_argument("-s", "--seed", default=0, help="random seed", type=int)
    parser.add_argument("--categories", default="", help="filter categories", type=str)

    args = parser.parse_args()
    SEED = args.seed
    torch.manual_seed(SEED)
    L = args.layers

    device = DEVICE

    # LOAD DATA
    # Subsample categories to change the dataset complexity: [4,1] is (deer/car); [5,3] is (dog/cat)
    filter_labels = None
    if args.categories:
        filter_labels = list([int(c) for c in args.categories])
    # CIFAR-10 doesn't use validation
    train_loader, valid_loader, test_loader = load_data(
        BATCH_SIZE, seed=SEED, filter_labels=filter_labels, validation_size=0
    )
    N_train = len(train_loader.sampler)

    # CREATE MODEL

    torch.manual_seed(SEED)

    if L < 0:
        vpost = TruncatedPoisson(5.0)
    else:
        vpost = FixedDepth(L)

    model = UnboundedNN(
        N_train,
        lambda l: generator_layers(l, True),
        generator_output,
        vpost,
        INPUT_SIZE,
        OUTPUT_SIZE,
        L_prior_poisson=1,
        theta_prior_scale=1.0,
    )

    model.model_name += ".cifar.v2" + ("-f" + args.categories if args.categories else "") + "-s%d" % SEED

    print(model.n_obs)
    model.set_device(device)

    # TRAINING LOOP
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = ExplicitLR(optimizer, [0.01] * 5 + [0.1] * 195 + [0.01] * 100 + [0.001] * 100)
    model.set_optimizer(optimizer)

    tmp = pd.DataFrame({"depth": [], "nu_L": [], "test_acc": []})
    tmp.to_csv("tmp.%s.csv" % model.model_name)

    for epoch in range(args.epochs):
        start_time = time.time()
        test_accuracy = train_one_epoch(
            epoch, train_loader, valid_loader, test_loader, model, optimizer, scheduler, normalize_loss=True
        )
        scheduler.step()
        print(time.time() - start_time)
