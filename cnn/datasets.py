## Datasets Module

import utils
import torchvision.datasets as dset

VALID_DSET_NAMES = {
    'CIFAR': ['cifar', 'cifar10', 'cifar-10'],
    'MNIST': ['mnist'],
    'FashionMNIST': ['fashionmnist', 'fashion-mnist', 'mnistfashion']
}

def load_dataset(args):
    """ function to load datasets (e.g. CIFAR10, MNIST, FashionMNIST, Graphene) """
    dset_name = args.dataset.lower().strip()

    if dset_name in VALID_DSET_NAMES['CIFAR']:
        train_transform, valid_transform = utils._data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    elif dset_name in VALID_DSET_NAMES['MNIST']:
        # TODO: add transforms
        train_data = dset.MNIST(root=args.data, train=True, download=True, transform=None)

    elif dset_name in VALID_DSET_NAMES['FashionMNIST']:
        # TODO: add transforms
        train_data = dset.FashionMNIST(root=args.data, train=True, download=True, transform=None)

    else:
        exc_str = 'Unable to match provided dataset name: {}'.format(dset_name)
        exc_str += '\nValid names are case-insensitive elements of: {}'.format(VALID_DSET_NAMES)
        raise RuntimeError(exc_str)

    return train_data
