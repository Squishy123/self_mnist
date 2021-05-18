from models.kmeans import KMeans
import torch

model = KMeans()
model.fit(torch.rand(100,1))