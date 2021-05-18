import torch

class KMeans():
    def __init__(self, n_clusters=10, dimension=1):
        self.n_clusters = n_clusters
        
        self.centroids = torch.rand(n_clusters, dimension)
        self.cluster_objects = {i: [] for i in range(n_clusters)}
        #print(self.cluster_objects)

        #self.append_object(torch.rand(5), 0)
        #self.append_object(torch.rand(5), 0)
        #print(self.cluster_objects)

    # helper to append objects into cluster
    def append_object(self, object, cluster_idx):
        self.cluster_objects[cluster_idx].append(object)

    # fit vectors
    def fit(self, X, max_iterations=100):
        for m in range(max_iterations):
            # assign all examples to clusters
            for x in X:
                if not isinstance(x, tuple):
                    x = tuple(x)
                mean = torch.sqrt(((self.centroids-x[0])**2).sum(axis=1))
                #print(mean)
                cluster_idx = torch.argmin(mean).item()
                #print(cluster_idx)
                self.append_object(x, cluster_idx)

            #print(self.cluster_objects)
            
            # update centroids
            loss = 0
            for c in range(10):
                for i in range(100):
                    len_cluster = len(self.cluster_objects[c])
                    if (len_cluster == 0):
                        mean = 0
                    else:
                        mean = torch.tensor([e[0] for e in self.cluster_objects[c]]).sum()/len_cluster
                    
                    loss += torch.sum(mean - self.centroids[c])
                    #print(loss)
                    self.centroids[c] = mean

            print(f"KMeans Clustering - Epoch: {m} - Loss: {loss.item()}")