import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


class CustomDatasets(object):
    """
    CustomDatasets is a multi class object supporting the following datasets: F-Mnist, SVHN,
    Cifar-10 and 100, STL-1 and Caltech 101-256.

    Augmentations for each dataset is provided in the _load_Dataset function as well as a
    transformation of the target labels to one-hot-encoding (OHE) if is required.

    The follow arguments must be given as input:
    - Path: the path in which the dataset is storted or externally provided via torchvision datasets.
    - target_type: None and OHE. If none is selected then no transformation is applied in labels,
                if Ohe then OHE transformation is applied in labels.
    - dataset_name: str argument "MNIST", "F-MNIST", "CIFAR10", "CIFAR100", "SVHN", "STL-1", "CALTECH101" and "CALTECH256".
    - batch_size: The desired batch size.
    - n_classes: Number of classes for the selected dataset.
    -transform_img: True, False if augmentations should be applied or not.

    """

    def __init__(self, config):
        self.path = config['path']
        self.target_type = config['target_type']
        self.dataset_name = config['dataset_name']
        self.batch_size = config['batch_size']
        self.n_classes = config['n_classes']
        self.transform_img = config['transform']
        self.dataset = None

    def _load_Dataset(self):

        if self.dataset_name == "MNIST" or self.dataset_name == "F-MNIST":
            if self.transform_img is True:
                transform = torchvision.transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=29),
                    # transforms.CenterCrop(size=28),
                    # transforms.Grayscale(num_output_channels=3),
                    transforms.Normalize((0.485,), (0.229,))
                ])
            else:
                transform = None

        elif self.dataset_name == "CIFAR10" or self.dataset_name == "CIFAR100" or self.dataset_name == "SVHN":
            if self.transform_img:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

                transform_test = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        elif self.dataset_name == "STL-1":
            if self.transform_img:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(96, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

                transform_test = transforms.Compose([

                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            else:
                tranform_train = None
                transform_test = None

        elif self.dataset_name == "CALTECH101" or self.dataset_name == "CALTECH256":
            if self.transform_img:
                transform_train = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                transform_test = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                tranform_train = None
                transform_test = None
        else:
            msg = 'Invalid name of dataset. Received {} but expected "MNIST", "F-MNIST", "CIFAR10" or "CIFAR100"'
            raise ValueError(msg.format(self.dataset_name))

        if self.target_type == 'Ohe':
            target_trans = (
                lambda y: torch.zeros(self.n_classes, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
        else:
            target_trans = None

        if self.dataset_name == "MNIST":
            self.dataset = {
                "Train": torchvision.datasets.MNIST(root=self.path, transform=transform, download=True,
                                                    target_transform=target_trans),
                "Test": torchvision.datasets.MNIST(root=self.path, train=False, transform=transform, download=True,
                                                   target_transform=target_trans)}

        elif self.dataset_name == "F-MNIST":
            self.dataset = {
                "Train": torchvision.datasets.FashionMNIST(root=self.path, transform=transform, download=True,
                                                           target_transform=target_trans),
                "Test": torchvision.datasets.FashionMNIST(root=self.path, train=False, transform=transform,
                                                          download=True, target_transform=target_trans)}
        elif self.dataset_name == "CIFAR10":
            self.dataset = {
                "Train": torchvision.datasets.CIFAR10(root=self.path, transform=transform_train, download=True,
                                                      target_transform=target_trans),
                "Test": torchvision.datasets.CIFAR10(root=self.path, train=False, transform=transform_test,
                                                     download=True, target_transform=target_trans)}

        elif self.dataset_name == "SVHN":
            self.dataset = {
                "Train": torchvision.datasets.SVHN(root=self.path, transform=transform_train, split="train",
                                                   download=True, target_transform=target_trans),
                "Test": torchvision.datasets.SVHN(root=self.path, split='test', transform=transform_test, download=True,
                                                  target_transform=target_trans)}

        elif self.dataset_name == "STL-1":
            self.dataset = {
                "Train": torchvision.datasets.STL10(root=self.path, transform=transform_train, split="train",
                                                    download=True, target_transform=target_trans),
                "Test": torchvision.datasets.STL10(root=self.path, split='test', transform=transform_test,
                                                   download=True, target_transform=target_trans)}

        elif self.dataset_name == "CIFAR100":
            self.dataset = {
                "Train": torchvision.datasets.CIFAR100(root=self.path, transform=transform_train, download=True,
                                                       target_transform=target_trans),
                "Test": torchvision.datasets.CIFAR100(root=self.path, train=False, transform=transform_test,
                                                      download=True, target_transform=target_trans)}
        elif self.dataset_name == "CALTECH101":
            self.dataset = torchvision.datasets.Caltech101(root=self.path, transform=transform_train, download=True,
                                                           target_transform=target_trans)

        elif self.dataset_name == "CALTECH256":
            self.dataset = {
                "Train": torchvision.datasets.Caltech256(root=self.path, transform=transform_train, download=True,
                                                         target_transform=target_trans),
                "Test": torchvision.datasets.Caltech256(root=self.path, train=False, transform=transform_test,
                                                        download=True, target_transform=target_trans)}
        else:
            self.dataset = None
            print("Error: {} dataset is not available.!".format(self.dataset_name))
        return self.dataset

    def make_loaders(self):

        if self.dataset is None:
            self.dataset = self._load_dataset()

        loader = {
            'Train_load': DataLoader(dataset=self.dataset['Train'], batch_size=self.batch_size),
            'Test_load': DataLoader(dataset=self.dataset['Test'], batch_size=self.batch_size)}
        return loader