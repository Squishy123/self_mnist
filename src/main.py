from utils.datasets import generate_augmented_datasets, generate_default_datasets
from models.mnist_classifier import MNIST_Classifier
from utils.train import encoder_pretrain, classifier_train
from utils.plot import show_images

import matplotlib.pyplot as plt

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
optimizer = torch.optim.SGD(model.encoder.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()
'''
plt.title('Encoder Pretraining Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

for i in range(50):
    ep_loss = encoder_pretrain(model, "cuda", DataLoader(train_aug, batch_size=64, shuffle=True), optimizer, criterion, i)
    plt.scatter(i, ep_loss, color="blue")
    plt.savefig("results/encoder_pretraining.png")

    if i % 10 == 0:
        torch.save(model.state_dict(), f"results/encoder_pretraining_{i}.pth")

plt.close()
'''
model.load_state_dict(torch.load("results/encoder_pretraining_49.pth"))
model.eval()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# 3. Train Model Encoder and Classifier

fig1, (ax1) = plt.subplots(1, constrained_layout=True)
fig2, (ax2) = plt.subplots(1, constrained_layout=True)

ax1.set_title('Encoder/Classifier Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

ax2.set_title('Encoder/Classifier Validation Score')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Score')

for i in range(100):
    ep_loss = classifier_train(model, "cuda",  train_def, optimizer, criterion, i)
    ax1.scatter(i, ep_loss, color="blue")
    fig1.savefig("results/classifier_training.png")

    # validation plot 
    images = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

    for j in range(20):
        j_sample = random.sample(by_labels[j % 10], 1)[0].unsqueeze(0).to("cuda")
        
        with torch.no_grad():
            _, j_class = model(j_sample)
        j_class = torch.argmax(j_class, dim=1).item()
        #print(j_sample)

        images[j_class].append(j_sample.squeeze(0).squeeze(0).detach().cpu().numpy())

    show_images(images, f"results/training_{i}_images.png", cols=20)

    #if i % 10 == 0:
    torch.save(model.state_dict(), f"results/classifier_training_{i}.pth")

plt.close()

model.load_state_dict(torch.load("results/classifier_training_10.pth"))
model.eval()

# 4. Generate Prototype Plot
images = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

for j in range(20):
    j_sample = random.sample(by_labels[j % 10], 1)[0].unsqueeze(0).to("cuda")
    
    with torch.no_grad():
        _, j_class = model(j_sample)
    j_class = torch.argmax(j_class, dim=1).item()
    #print(j_sample)

    images[j_class].append(j_sample.squeeze(0).squeeze(0).detach().cpu().numpy())

show_images(images, f"training_images.png", cols=20)

#print(x0_class)
#print(x1_class)
