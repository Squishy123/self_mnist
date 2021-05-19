from utils.datasets import generate_augmented_datasets, generate_default_datasets
from models.mnist_classifier import MNIST_Classifier
from utils.train import encoder_pretrain, classifier_train, reform_train
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
optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

model.load_state_dict(torch.load("results/encoder_pretraining_200.pth"))
model.eval()

fig1, (ax1) = plt.subplots(1, constrained_layout=True)
fig2, (ax2) = plt.subplots(10, 2, constrained_layout=True)
np.vectorize(lambda ax:ax.axis('off'))(ax2)

ax1.set_title('Encoder Pretraining Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

for i in range(201, 501):
    ep_loss = encoder_pretrain(model, "cuda", DataLoader(train_aug, batch_size=512, shuffle=True), optimizer, criterion, i)
    ax1.scatter(i, ep_loss, color="blue")
    fig1.savefig("results/encoder_pretraining_loss.png")

    if i % 10 == 0:
        for k, images in by_labels.items():
            sample = random.sample(images, 1)[0]
            ax2[k][0].imshow(sample.squeeze(0))

            with torch.no_grad():
                _, decoded, _ = model(sample.unsqueeze(0).to("cuda"))

            ax2[k][1].imshow(decoded.squeeze(0).squeeze(0).detach().cpu().numpy())
        
        fig2.savefig(f"results/encoder_pretraining_decoded_{i}.png")

    if i % 10 == 0:
        torch.save(model.state_dict(), f"results/encoder_pretraining_{i}.pth")

plt.close()

exit()

model.load_state_dict(torch.load("results/encoder_pretraining_100.pth"))
model.eval()

optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

# 3. Reform Classes
'''
fig1, (ax1) = plt.subplots(1, constrained_layout=True)

ax1.set_title('Distribution Reforming')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss') 

for i in range(10):
    ep_loss = reform_train(model, "cuda", DataLoader(train_def, batch_size=512, shuffle=True), optimizer, criterion, i)
    ax1.scatter(i, ep_loss, color="blue")
    fig1.savefig("results/distribution_reforming.png")

torch.save(model.state_dict(), f"results/distribution_uniform.pth")
'''

# 3. Train Model Encoder and Classifier

fig1, (ax1) = plt.subplots(1, constrained_layout=True)
fig2, (ax2) = plt.subplots(1, constrained_layout=True)

ax1.set_title('Encoder/Classifier Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

ax2.set_title('Encoder/Classifier Validation Score')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Score')

for i in range(101):
    ep_loss = classifier_train(model, "cuda",  train_def, optimizer, criterion, i)
    ax1.scatter(i, ep_loss, color="blue")
    fig1.savefig("results/classifier_training_loss.png")

    # validation plot 
    images = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

    for j in range(20):
        j_sample = random.sample(by_labels[j % 10], 1)[0].unsqueeze(0).to("cuda")
        
        with torch.no_grad():
            _, _, j_class = model(j_sample)
        j_class = torch.argmax(j_class, dim=1).item()
        #print(j_sample)

        images[j_class].append(j_sample.squeeze(0).squeeze(0).detach().cpu().numpy())

    show_images(images, f"results/classifier_training_{i}_images.png", cols=20)

    #if i % 10 == 0:
    torch.save(model.state_dict(), f"results/classifier_training_{i}.pth")

plt.close()
'''
model.load_state_dict(torch.load("results/classifier_training_10.pth"))
model.eval()

# 4. Generate Prototype Plot
images = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

for j in range(20):
    j_sample = random.sample(by_labels[j % 10], 1)[0].unsqueeze(0).to("cuda")
    
    with torch.no_grad():
        _, _, j_class = model(j_sample)
    j_class = torch.argmax(j_class, dim=1).item()
    #print(j_sample)

    images[j_class].append(j_sample.squeeze(0).squeeze(0).detach().cpu().numpy())

show_images(images, f"training_images.png", cols=20)

#print(x0_class)
#print(x1_class)
'''