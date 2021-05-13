import torch.nn as nn
import torch

from collections import OrderedDict

'''
MNIST_Classifier Network takes in a 1 channel b/w image
and encodes feature embeddings
'''
class MNIST_Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(MNIST_Classifier, self).__init__()

        self.encoder = nn.Sequential(OrderedDict([
            ('encoder_conv1', nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)),
            ('encoder_relu1', nn.LeakyReLU()),
            ('encoder_conv2',  nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)),
            ('encoder_relu2', nn.LeakyReLU()),
            ('encoder_conv3', nn.Conv2d(32, 64, kernel_size=7)),
            ('encoder_relu3', nn.LeakyReLU()),
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('decoder_Tconv1', nn.ConvTranspose2d(64, 32, kernel_size=7)),
            ('decoder_relu1', nn.LeakyReLU()),
            ('decoder_Tconv2', nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)),
            ('decoder_relu2', nn.LeakyReLU()),
            ('decoder_Tconv3', nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)),
            ('decoder_relu3', nn.Sigmoid()),
        ]))

        self.head = nn.Sequential(OrderedDict([
            ('encoder_flatten', nn.Flatten()),
            ('classifier_linear1', nn.Linear(64, 32)),
            ('classifier_relu1', nn.LeakyReLU()),
            ('classifier_linear2', nn.Linear(32, num_classes)),
            ('classifier_softmax1', nn.Softmax(dim=1))
        ]))
        

    def forward(self, x):
        x1 = self.encoder(x)
        #print(x1.shape)
        x2 = self.decoder(x1.detach())
        x3 = self.head(x1.detach())
        #print(x2.shape)
        return x1, x2, x3
