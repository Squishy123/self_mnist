import torchvision
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models.knn import KNN

default_transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
        ])

augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(28),
        transforms.RandomHorizontalFlip()
    ])

'''
Wrapper Class to Provide Augmented Transform with Each Item
'''
class AugmentedMNIST(datasets.MNIST):
    def __init__(self,
            root,
            train = True,
            transform = None,
            augment_transform = None,
            target_transform = None,
            download = False,):

        super(AugmentedMNIST, self).__init__(root, train, transform, target_transform, download)

        self.augment_transform = augment_transform
        self.knn = KNN()

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        aug = None
        if self.augment_transform != None:
            aug = self.augment_transform(img)

        #print(index)
        #other_data = torch.cat([self.data[:index], self.data[index+1:]])
        #neighbors = self.knn(img, other_data)

        return img, aug, target

'''
Augmented Training and Test Datasets
'''
def generate_augmented_datasets():
    train_data = AugmentedMNIST('data', train=True, download=True, transform=default_transform, augment_transform=augment_transform)
    test_data = AugmentedMNIST('data', train=True, download=True, transform=default_transform, augment_transform=augment_transform)

    return train_data, test_data

'''
Default Training And Test Datasets
'''
def generate_default_datasets():
    train_data = datasets.MNIST('data', train=True, download=True, transform=default_transform)
    test_data = datasets.MNIST('data', train=False, download=True, transform=default_transform)
    
    return train_data, test_data