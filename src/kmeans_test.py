from models.kmeans import KMeans
import torch

import matplotlib.pyplot as plt

torch.manual_seed(123)

model = KMeans(n_clusters=5, dimension=2)
X = torch.randn(100,2)/6
model.init_centroids(X)


fig, (ax) = plt.subplots(1, constrained_layout=True)
ax.set_title('Encoder/Classifier Training Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')


#plt.clf()
model.fit(X, 20)
#print(model.entropy())

colors = ["red", "green", "blue", "orange", "pink", "yellow"]

for _, (k, c) in enumerate(model.cluster_objects.items()):
    for ((x,y), _) in c:
        ax.scatter(x,y, c=colors[k], cmap='cool')

for i, (x, y) in enumerate(list(model.centroids)):
    ax.scatter(x,y,c='white',
        alpha=0.6,
        edgecolors='black',
        linewidths=2)
plt.show()
    #plt.draw()
   # plt.pause(1e-5)