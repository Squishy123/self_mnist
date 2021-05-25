import random
from numpy import log, log10, log2
import torch
import matplotlib.pyplot as plt

class KMeans():
    def __init__(self, n_clusters=10, dimension=1):
        self.n_clusters = n_clusters
        
        self.centroids = torch.rand(n_clusters, dimension)
        self.cluster_objects = {i: [] for i in range(n_clusters)}

    # helper to append objects into cluster
    def append_object(self, object, cluster_idx):
        self.cluster_objects[cluster_idx].append(object)

    def init_centroids(self, X):
        # init centroids
        if isinstance(X[0], tuple): 
            sample = random.sample([x[0] for x in X], self.n_clusters)
        else: 
            sample = random.sample([x for x in X], self.n_clusters)
        self.centroids = torch.stack(sample)

    def entropy(self):
        total_length = 0

        count = torch.zeros(self.n_clusters)

        # len of all items
        for idx, (k, cluster) in enumerate(self.cluster_objects.items()):
            total_length += len(cluster)
            count[k] = len(cluster)

        # get probabilities
        count /= total_length
        print(count)
        entropy = -sum([p * log(p + 1e-8)/log(self.n_clusters) for p in count])            

        return entropy.item()

    # fit vectors
    def fit(self, X, max_iterations=100, dist="cosine"):
        last_loss = 0
        for m in range(max_iterations):
            # assign all examples to clusters
            for x in X:
                #print(x[1])
                if not isinstance(x, tuple):
                    #print("not tuple")
                    x = (x, None)

                vector, *args = x
                #print(self.centroids.shape)
                #print(vector.repeat(self.n_clusters, 1).shape)
                #mean = torch.sqrt(((self.centroids-vector)**2)).sum(axis=1)
                if dist == "cosine":
                    mean = torch.nn.functional.cosine_similarity(torch.nn.functional.normalize(self.centroids, p=2, dim=1), torch.nn.functional.normalize(vector.repeat(self.n_clusters, 1), p=2, dim=1), dim=1)
                elif dist == "euclidean":
                    mean = torch.sqrt(((self.centroids-vector)**2)).sum(axis=1)
                #print(mean.shape)
                cluster_idx = torch.argmin(mean).item()
                #print(cluster_idx)
                self.append_object(x, cluster_idx)

            #print(self.cluster_objects)
            
            # update centroids
            loss = 0
            for c in range(self.n_clusters):
                for i in range(1):
                    len_cluster = len(self.cluster_objects[c])
                    if (len_cluster == 0):
                        mean = 0
                    else:
                        l = torch.stack([e[0] for e in self.cluster_objects[c]], axis=0)
                        #print(l.shape)
                        mean = torch.mean(l, 0)
                        #print(mean.shape)
                    
                    #print(mean-self.centroids[c])
                    temp = self.centroids[c].clone()
                    #print(loss)
                    self.centroids[c] = mean
                    loss += torch.sum(self.centroids[c] - temp)
            
            print(f"KMeans Clustering - Epoch: {m} - Loss: {loss.item()}")

            if (abs(loss) == abs(last_loss)):
                break
            else:
                last_loss = loss

            # clear cluster objects
            if m != max_iterations - 1:
                self.cluster_objects = {i: [] for i in range(self.n_clusters)}