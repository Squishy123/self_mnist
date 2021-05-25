import random
from models.kmeans import KMeans
from models.mnist_classifier import MNIST_Classifier
from utils.datasets import generate_augmented_datasets, generate_default_datasets
import torch

import matplotlib.pyplot as plt

from utils.plot import show_images

model = MNIST_Classifier().to("cuda")
optimizer = torch.optim.RMSprop(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=1e-3)
model.load_state_dict(torch.load("results_new_base/encoder_pretraining_380.pth"))
model.eval()

# get samples
train_def, test_def = generate_default_datasets()

by_labels = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

for i, (img, label) in enumerate(train_def):
    by_labels[label].append(img)

with torch.no_grad():
    centroid_X = [torch.flatten(model(by_labels[i][random.randint(0, len(by_labels[i]))].to("cuda").unsqueeze(0))[0]).cpu() for i in range(10)]

fig, (ax) = plt.subplots(1, constrained_layout=True)
ax.set_title('Encoder/Classifier Training Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')

X = []
with torch.no_grad():
    for item_idx, (img, _) in enumerate(random.sample(list(train_def), 1000)):
            # send to device
            img = img.to("cuda").unsqueeze(0)
            img_embedding, _, img_class = model(img)
            X.append((torch.flatten(img_embedding).cpu(), img.squeeze(0).squeeze(0).cpu()))
            #X.append((torch.flatten(img).cpu(), img.squeeze(0).squeeze(0).cpu()))
#print(X.shape)

cluster = KMeans(n_clusters=10)

# find best entropy
max_entropy = 0
max_centroids = None

#for i in range(1):
cluster.init_centroids(X)
#cluster.centroids = centroid_X
cluster.fit(X, 10, dist="cosine")
    
    
    #if cluster.entropy() > max_entropy:
    #    max_entropy = cluster.entropy()
    #    max_centroids = cluster.centroids

print(cluster.entropy())
#cluster.centroids = max_centroids
#cluster.fit(X, 1)

images = {i:[] for i in range(20)}

#print(model.cluster_objects)
for _, (k, c) in enumerate(cluster.cluster_objects.items()):
    for (x,y) in c:
        images[k].append(y)

show_images(images, f"cluster_images.png", cols=20, rows=20)

#plt.show()
#centroids = dict(zip(range(10), cluster.centroids))
#show_images(centroids, "centroids.png", cols=10)
