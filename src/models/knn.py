import torch.nn as nn
import torch

from collections import OrderedDict

'''
KNN Layer Implementation
'''

class KNN(nn.Module):
    def __init__(self, k_neighbors=5):
        super(KNN, self).__init__()

        self.k_neighbors = k_neighbors

    # return the k nearest neighbors to x in X
    def forward(self, x, X):
        neighbors = []
        for (k_embedding, k_class, *args) in X:

            k_difference = torch.sub(k_embedding, x)
            #print(torch.norm(k_difference, dim=1).item())
            neighbors.append((torch.norm(k_difference, dim=1).item(), (k_embedding, k_class, *args)))

            # remove largest difference
            if len(neighbors) > self.k_neighbors:
                neighbors.sort(key=lambda x:x[0], reverse=False)
                neighbors.pop()

        return [k for (_,k) in neighbors]
