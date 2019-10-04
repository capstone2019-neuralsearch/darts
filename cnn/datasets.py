## Datasets Module

import utils
import torchvision.datasets as dset

VALID_DSET_NAMES = {
    'CIFAR': ['cifar', 'cifar10', 'cifar-10'],
    'MNIST': ['mnist'],
    'FashionMNIST': ['fashionmnist', 'fashion-mnist', 'mnistfashion'],
    'GrapheneKirigami': ['graphene', 'graphenekirigami', 'graphene-kirigami', 'kirigami']
}

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
        # TODO: implement loading graphene
        # output_dim =
        # is_regression = True
        raise NotImplementedError()

    else:
        exc_str = 'Unable to match provided dataset name: {}'.format(dset_name)
        exc_str += '\nValid names are case-insensitive elements of: {}'.format(VALID_DSET_NAMES)
        raise RuntimeError(exc_str)

    return data, output_dim, in_channels, is_regression
