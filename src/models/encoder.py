import torch.nn as nn
import torch

from collections import OrderedDict

'''
Encoder Network takes in a 1 channel b/w image
and encodes feature embeddings
'''
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
            ('encoder_conv1', nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)),
            ('encoder_relu1', nn.LeakyReLU()),
            ('encoder_conv2',  nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)),
            ('encoder_relu2', nn.LeakyReLU()),
            ('encoder_conv3', nn.Conv2d(32, 64, kernel_size=7)),
            ('encoder_relu3', nn.LeakyReLU()),
        ]))

    def forward(self, x):
        x = self.encoder(x)
        return x
