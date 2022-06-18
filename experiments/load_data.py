import os
import numpy as np
import torch
import torchvision
from torch.utils.data import SubsetRandomSampler
from torchvision import transforms


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


def load_data_spiral(phase: float, batch_size: int, seed: int = 0):
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


def load_data_cifar(batch_size: int, seed: int = 0, validation_size: float = 0.2, filter_labels=None):
    """
    Load the CIFAR-10 dataset and wrap it in torch DataLoaders: train, validation, test.
    Has 10 classes: ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    Parameters
    ----------
    batch_size: int
        Batch size for the data loaders.
    seed: int
        Seed for splitting the dataset into train and validation.
    validation_size: float between 0.0 and 1.0, default 0.2
        Proportion of the train set used for the validation set.

    Returns
    -------
    train_loader, valid_loader, test_loader

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

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

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


def load_data_uci(dataset_name, n_split, batchsize, seed=0):
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
