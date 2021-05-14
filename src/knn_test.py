from models.knn import KNN
from utils.datasets import generate_augmented_datasets, generate_default_datasets
from models.mnist_classifier import MNIST_Classifier
from utils.train import encoder_pretrain, classifier_train
from utils.plot import show_images

import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
import torch.nn 
import torch

import random

# 1. Load Datasets
train_aug, test_aug = generate_augmented_datasets()
train_def, test_def = generate_default_datasets()

# Labels for Validation Tests
by_labels = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

for i, (img, label) in enumerate(test_def):
    by_labels[label].append(img)

# 2. Pretrain Model Encoder on Augmentations
model = MNIST_Classifier().to("cuda")
optimizer = torch.optim.RMSprop(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=1e-3)
criterion = torch.nn.MSELoss()

model.load_state_dict(torch.load("results/encoder_pretraining_100.pth"))
model.eval()

# 3. Generate KNN for random sample
x, _ = random.sample(list(test_def), 1)[0]
x_embed,  _, s_class = model(x.to("cuda").unsqueeze(0))

knn = KNN()
samples = random.sample(list(test_def), 10000)

with torch.no_grad():
    for s in range(len(samples)):
        s_img, _ = samples[s]
        embedding, _, s_class = model(s_img.to("cuda").unsqueeze(0))
        samples[s] = (embedding, s_class, s_img)

# generate stochastic KNN for encodings
neighbors = knn(x_embed, samples)

fig, ax = plt.subplots(1, 6, constrained_layout=True)
np.vectorize(lambda ax:ax.axis('off'))(ax)

ax[0].imshow(x.squeeze(0))
ax[0].set_title("X")

for idx, (n_embed, n_class, n_img) in enumerate(neighbors):
    ax[idx + 1].imshow(n_img.squeeze(0))
    ax[idx + 1].set_title(idx) 

plt.show()