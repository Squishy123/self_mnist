import random
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

    # fit vectors
    def fit(self, X, max_iterations=100):

        for m in range(max_iterations):
            # assign all examples to clusters
            for x in X:
                #print(x[1])
                if not isinstance(x, tuple):
                    x = tuple(x)

                vector, *args = x
                #print(torch.sqrt(((self.centroids-vector)**2)).sum(axis=1).shape)
                mean = torch.sqrt(((self.centroids-vector)**2)).sum(axis=1)
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
                        #exit()
                        mean = l.sum(axis=0)/len_cluster
                        #print(mean.shape)
                    
                    loss += torch.sum(mean - self.centroids[c])
                    #print(loss)
                    self.centroids[c] = mean

            print(f"KMeans Clustering - Epoch: {m} - Loss: {loss.item()}")

            # clear cluster objects
            if m != max_iterations - 1:
                self.cluster_objects = {i: [] for i in range(self.n_clusters)}