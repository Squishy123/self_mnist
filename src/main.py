from utils.datasets import generate_augmented_datasets, generate_default_datasets
from models.encoder import Encoder
from utils.train import encoder_pretrain

import torch.nn 
import torch

# 1. Load Datasets
train_aug, test_aug = generate_augmented_datasets()
train_def, test_def = generate_default_datasets()

# 2. Pretrain Model Encoder on Augmentations
encoder = Encoder()
optimizer = torch.optim.SGD(encoder.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss(size_average=True)

for i in range(10):
    encoder_pretrain(encoder, "cuda", train_aug, optimizer, criterion, i)

# 3. Generate KNN Set

# 4. Train Model Classifier

# Minimize Encoded Features between Neighbors

# 