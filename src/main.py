from utils.datasets import generate_augmented_datasets, generate_default_datasets
from models.mnist_classifier import MNIST_Classifier
from utils.train import encoder_pretrain, classifier_train

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
import torch.nn 
import torch

# 1. Load Datasets
train_aug, test_aug = generate_augmented_datasets()
train_def, test_def = generate_default_datasets()

# 2. Pretrain Model Encoder on Augmentations
model = MNIST_Classifier()
optimizer = torch.optim.SGD(model.encoder.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

'''
for i in range(5):
    ep_loss = encoder_pretrain(model, "cuda", DataLoader(train_aug, batch_size=64, shuffle=True), optimizer, criterion, i)
    plt.scatter(i, ep_loss, color="blue")
    plt.savefig("results/encoder_pretraining.png")
'''
# 4. Train Model Encoder and Classifier

for i in range(5):
    ep_loss = classifier_train(model, "cuda",  train_def, optimizer, criterion, i)
    plt.scatter(i, ep_loss, color="blue")
    plt.savefig("results/classifier_training.png")

# Minimize Encoded Features between Neighbors

# 