## Datasets Module

import os
import utils
import torchvision.datasets as dset
import numpy as np
import xarray as xr
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch import from_numpy
import boto3

VALID_DSET_NAMES = {
    'CIFAR': ['cifar', 'cifar10', 'cifar-10'],
    'MNIST': ['mnist'],
    'FashionMNIST': ['fashionmnist', 'fashion-mnist', 'mnistfashion'],
    'GrapheneKirigami': ['graphene', 'graphenekirigami', 'graphene-kirigami', 'kirigami']
}

BUCKET_NAME = 'capstone2019-google'

def load_dataset(args, train=True):
    """ function to load datasets (e.g. CIFAR10, MNIST, FashionMNIST, Graphene)

    input:
        args - result of ArgumentParser.parse() containing a `dataset` property

    output:
        data - a torch Dataset
        output_dim - an integer representing necessary output dimension for a model
        is_regression - boolean indicator for regression problems
    """
    dset_name = args.dataset.lower().strip()

    if dset_name in VALID_DSET_NAMES['CIFAR']:
        # from the original DARTS code
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        tr = train_transform if train else valid_transform
        data = dset.CIFAR10(root=args.data, train=train, download=True, transform=tr)
        output_dim = 10
        in_channels = 3
        is_regression = False

    elif dset_name in VALID_DSET_NAMES['MNIST']:
        train_transform, valid_transform = utils._data_transforms_mnist(args)
        tr = train_transform if train else valid_transform
        data = dset.MNIST(root=args.data, train=train, download=True, transform=tr)
        output_dim = 10
        in_channels = 1
        is_regression = False

    elif dset_name in VALID_DSET_NAMES['FashionMNIST']:
        train_transform, valid_transform = utils._data_transforms_mnist(args)
        tr = train_transform if train else valid_transform
        data = dset.FashionMNIST(root=args.data, train=train, download=True, transform=tr)
        output_dim = 10
        in_channels = 1
        is_regression = False

    elif dset_name in VALID_DSET_NAMES['GrapheneKirigami']:
        # load xarray dataset
        data_path = os.path.join(args.data, 'graphene_processed.nc')
        ds = xr.open_dataset(data_path)

        # X = ds['coarse_image'].values  # coarse 3x5 image (not using it)
        X = ds['fine_image'].values  # the same model works worse on higher resolution image
        y = ds['strain'].values
        X = X[..., np.newaxis]  # add channel dimension
        y = y[:, np.newaxis]  # pytorch wants ending 1 dimension

        # pytorch conv2d wants channel-first, unlike Keras
        X = X.transpose([0, 3, 1, 2])  # (sample, x, y, channel) -> (sample, channel, x, y)

        # it appears we need each dimension to be twice divisible by 2
        # reshape from 30x80 -> 32x80 by zero-padding
        # see here for details https://stackoverflow.com/a/46115998
        X = np.pad(X, [(0, 0), (0, 0), (1, 1), (0, 0)], mode='constant', constant_values=0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if train:
            data = TensorDataset(from_numpy(X_train), from_numpy(y_train))
        else:
            data = TensorDataset(from_numpy(X_test), from_numpy(y_test))

        output_dim = 1
        in_channels = 1
        is_regression = True

    else:
        exc_str = 'Unable to match provided dataset name: {}'.format(dset_name)
        exc_str += '\nValid names are case-insensitive elements of: {}'.format(VALID_DSET_NAMES)
        raise RuntimeError(exc_str)

    return data, output_dim, in_channels, is_regression
