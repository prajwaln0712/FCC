import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import os
import random
import numpy as np

# ---------------------------------------------------------------------------
# Original CIFAR / TinyImagenet loaders (kept for reference, commented out)
# ---------------------------------------------------------------------------
# def load_inception():
#     global save_path
#     # Load pre-saved INCEPTION features
#     print('Loading Inception_v3 embeddings for CIFAR-10 and CIFAR-100...', end="")
#     cifar10_features_path = './data/my_cifar10_features.pt'
#     cifar100_features_path = './data/my_cifar100_features.pt'
#     features_cifar10 = torch.load(cifar10_features_path)
#     features_cifar100 = torch.load(cifar100_features_path)
#     print('done')
#     save_path = './data/tiny_imagenet_inception.pt'
#     print(f"Loading TinyImagenet Inception features loaded from {save_path}...", end="")
#     extracted_features = torch.load(save_path)
#     print("done")
#     return features_cifar10, features_cifar100, extracted_features
#
#
# def load_dinov3():
#     global save_path
#     # Load pre-saved DINOv3 embeddings
#     print('Loading Dino_v3 embeddings for CIFAR-10 and CIFAR-100...', end="")
#     dino_features_cifar10_path = './data/my_cifar10_dinov3.pt'
#     dino_features_cifar100_path = './data/my_cifar100_dinov3.pt'
#     dino_features_cifar10 = torch.load(dino_features_cifar10_path)
#     dino_features_cifar100 = torch.load(dino_features_cifar100_path)
#     print('done')
#     print(f"Loading TinyImagenet Dinov3 features loaded from {save_path}...", end="")
#     save_path = './data/tiny_imagenet_dinov3.pt'
#     tiny_dino_extracted_features = torch.load(save_path)
#     print("done")
#     return dino_features_cifar10, dino_features_cifar100, tiny_dino_extracted_features
#
#
# def load_cifar_pytorch_datasets():
#     # Load the CIFAR-10,100 pytorch training dataset
#     print('Loading CIFAR-10 and CIFAR-100 Torch Datasets...', end="")
#     transform = transforms.Compose([transforms.ToTensor()])  # No need of transforms
#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
#     trainset100 = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform)
#     print('done')
#     return trainset,trainset100
#
#
# def load_tinyimagenet_metadata():
#     # Load tinyimagenet METADATA
#     print(f"Loading TinyImagenet classes...", end="")
#     tiny_imagenet_cls_names = torch.load('./data/tiny_imagenet_cls_names.pt')
#     tiny_imagenet_cls_index = torch.load('./data/tiny_imagenet_cls_index.pt')
#     tiny_imagenet_cls_idx_to_name_dict = torch.load('./data/tiny_imagenet_cls_idx_to_name_dict.pt')
#     tiny_imagenet_cls_index_to_idx_dict = torch.load('./data/tiny_imagenet_cls_index_to_idx_dict.pt')
#     print("done")
#     return ( tiny_imagenet_cls_names, tiny_imagenet_cls_index, tiny_imagenet_cls_idx_to_name_dict, tiny_imagenet_cls_index_to_idx_dict )
#
#
# def sanity_check():
#     print(f'CIFAR-10 classes: {trainset.classes}')
#     print(f'CIFAR-100 classes: {trainset100.classes}')
#     print(f'Tiny-Imagenet-200 classes: {np.unique(tiny_imagenet_cls_names)}')
#
#
# def main():
#     load_inception()
#     load_dinov3()
#     load_cifar_pytorch_datasets()
#     load_tinyimagenet_metadata()
#     # sanity_check()
#
# # main()


# ---------------------------------------------------------------------------
# MNIST-family datasets: 28x28 greyscale -> tensor -> normalize -> flatten(784)
# ---------------------------------------------------------------------------
DATA_ROOT = './data'
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081

mnist_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    transforms.Lambda(lambda x: x.view(-1)),
])
DATASET_NORMALIZATION = {
    'ImageNet': ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))}

def load_imagenet(datadir=None, resize=None, tiny=False, augmentations=False, maxsize=None, seed=42, **kwargs):
    """ Load ImageNet dataset """
    random.seed(seed)
    torch.manual_seed(seed)
    if datadir is None and (not tiny):
        datadir = os.path.join(DATA_ROOT, 'imagenet')
    elif datadir is None and tiny:
        datadir = os.path.join(DATA_ROOT, 'tiny-imagenet-200')

    traindir = os.path.join(datadir, "train")
    validdir = os.path.join(datadir, "val")
    if augmentations:

        brightness = kwargs.get('brightness', 0.4)
        contrast = kwargs.get('contrast', 0.4)
        saturation = kwargs.get('saturation', 0.4)
        hue = kwargs.get('hue', 0.2)

        train_transform_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
            transforms.ToTensor(),
            transforms.Normalize(*DATASET_NORMALIZATION['ImageNet'])
        ]
    else:
        train_transform_list = [
            transforms.Resize(224), # revert back to 256
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(*DATASET_NORMALIZATION['ImageNet'])
        ]

    valid_transform_list = [
        transforms.Resize(224),# revert back to 256
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(*DATASET_NORMALIZATION['ImageNet'])
    ]

    if resize is not None:
        train_transform_list.insert(3, transforms.Resize(
            (resize, resize)))
        valid_transform_list.insert(2, transforms.Resize(
            (resize, resize)))

    train_data = dset.ImageFolder(
        traindir,
        transforms.Compose(
            train_transform_list
        ),
    )

    valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose(
            valid_transform_list
        ),
    )
    # ImageFolder stores .targets as a Python list; convert once here so every
    # later consumer (e.g. tiny_split.py's random_split_into_tasks) can compare
    # against it elementwise without re-converting at each call site.
    train_data.targets = torch.tensor(train_data.targets)
    valid_data.targets = torch.tensor(valid_data.targets)

    return train_data, valid_data


def load_cifar_pytorch_datasets(datadir=None, download=True, include_cifar100=True):
    """ Load CIFAR-10 (and optionally CIFAR-100) train+test datasets (same return convention as load_imagenet). """
    if datadir is None:
        datadir = DATA_ROOT
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10_train = torchvision.datasets.CIFAR10(root=datadir, train=True, download=download, transform=transform)
    cifar10_test = torchvision.datasets.CIFAR10(root=datadir, train=False, download=download, transform=transform)
    if not include_cifar100:
        return cifar10_train, cifar10_test
    cifar100_train = torchvision.datasets.CIFAR100(root=datadir, train=True, download=download, transform=transform)
    cifar100_test = torchvision.datasets.CIFAR100(root=datadir, train=False, download=download, transform=transform)
    return cifar10_train, cifar10_test, cifar100_train, cifar100_test


def load_mnist(train=True, download=True):
    print('Loading MNIST...', end="")
    ds = torchvision.datasets.MNIST(root=DATA_ROOT, train=train, download=download, transform=mnist_transform)
    print('done')
    return ds


def load_kmnist(train=True, download=True):
    print('Loading KMNIST...', end="")
    ds = torchvision.datasets.KMNIST(root=DATA_ROOT, train=train, download=download, transform=mnist_transform)
    print('done')
    return ds


def load_fashionmnist(train=True, download=True):
    print('Loading FashionMNIST...', end="")
    ds = torchvision.datasets.FashionMNIST(root=DATA_ROOT, train=train, download=download, transform=mnist_transform)
    print('done')
    return ds


def load_usps(train=True, download=True):
    print('Loading USPS...', end="")
    ds = torchvision.datasets.USPS(root=DATA_ROOT, train=train, download=download, transform=mnist_transform)
    print('done')
    return ds


def load_emnist_letters(train=True, download=True):
    print('Loading EMNIST (letters)...', end="")
    ds = torchvision.datasets.EMNIST(root=DATA_ROOT, split='letters', train=train, download=download, transform=mnist_transform)
    print('done')
    return ds


def load_mnist_family_datasets(train=True, download=True):
    return {
        'MNIST': load_mnist(train, download),
        'KMNIST': load_kmnist(train, download),
        'FashionMNIST': load_fashionmnist(train, download),
        'USPS': load_usps(train, download),
        'EMNIST-letters': load_emnist_letters(train, download),
    }


def dataset_to_tensors(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=False)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    X, y = torch.cat(xs), torch.cat(ys)
    # EMNIST-letters labels run 1..26; shift every dataset to 0..K-1
    return X.double(), y - y.min()


def sanity_check():
    for name, ds in load_mnist_family_datasets().items():
        x, y = ds[0]
        print(f'{name}: n={len(ds)} sample_shape={tuple(x.shape)} label={y}')


def main():
    load_mnist_family_datasets()
    # sanity_check()


# main()
