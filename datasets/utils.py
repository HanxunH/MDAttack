from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST, ImageNet

transform_options = {
    "CIFAR10": {
        "train_transform": [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ],
        "test_transform": [transforms.ToTensor()]},
    "CIFAR100": {
         "train_transform": [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation(15),
                             transforms.ToTensor()],
         "test_transform": [transforms.ToTensor()]},
    "SVHN": {
        "train_transform": [transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "MNIST": {
        "train_transform": [transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "ImageNet": {
        "train_transform": [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4,
                                                   hue=0.2),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor()]}
    }

transform_options["CIFAR10Noisy"] = transform_options["CIFAR10"]

dataset_options = {
    "CIFAR10": lambda path, transform, is_test, kwargs:
        CIFAR10(root=path, train=not is_test, download=True,
                transform=transform),
    "CIFAR100": lambda path, transform, is_test, kwargs:
        CIFAR100(root=path, train=not is_test, download=True,
                 transform=transform),
    "SVHN": lambda path, transform, is_test, kwargs:
        SVHN(root=path, split='test' if is_test else 'train', download=True,
             transform=transform),
    "MNIST": lambda path, transform, is_test, kwargs:
        MNIST(root=path, train=not is_test, download=True,
              transform=transform),
    "ImageNet": lambda path, transform, is_test, kwargs:
        ImageNet(root=path, split='val' if is_test else 'train',
                 transform=transform)
}
