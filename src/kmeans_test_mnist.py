import random
from models.kmeans import KMeans
from models.mnist_classifier import MNIST_Classifier
from utils.datasets import generate_augmented_datasets, generate_default_datasets
import torch

import matplotlib.pyplot as plt

from utils.plot import show_images

model = MNIST_Classifier().to("cuda")
optimizer = torch.optim.RMSprop(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=1e-3)
model.load_state_dict(torch.load("results/encoder_pretraining_380.pth"))
model.eval()

# get samples
train_def, test_def = generate_default_datasets()

fig, (ax) = plt.subplots(1, constrained_layout=True)
ax.set_title('Encoder/Classifier Training Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')

X = []
with torch.no_grad():
    for item_idx, (img, _) in enumerate(random.sample(list(train_def), 10000)):
            # send to device
            img = img.to("cuda").unsqueeze(0)
            img_embedding, _, img_class = model(img)
            #X.append((torch.flatten(img_embedding).cpu(), img.squeeze(0).squeeze(0).cpu()))
            X.append((torch.flatten(img.squeeze(0).squeeze(0)).cpu(), img.squeeze(0).squeeze(0).cpu()))
#print(X)

cluster = KMeans(n_clusters=10)

# find best entropy
max_entropy = 0
max_centroids = None

'''
for i in range(10):
    cluster.init_centroids(X)
    cluster.fit(X, 20)

    if cluster.entropy() > max_entropy:
        max_entropy = cluster.entropy()
        max_centroids = cluster.centroids
        print(max_entropy)

cluster.centroids = max_centroids
'''
cluster.init_centroids(X)
cluster.fit(X, 20)
print(cluster.entropy())

images = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

#print(model.cluster_objects)
for _, (k, c) in enumerate(cluster.cluster_objects.items()):
    for (x,y) in c:
        images[k].append(y)

show_images(images, f"cluster_images.png", cols=10)

#plt.show()
#centroids = dict(zip(range(10), cluster.centroids))
#show_images(centroids, "centroids.png", cols=10)
