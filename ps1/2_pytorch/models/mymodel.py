import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.modules.upsampling import Upsample

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(-1, 100352)
        return x

class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        self.channels, self.height, self.width = im_size
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.n_classes = n_classes

        # Resnet18
        self.pretrained = models.resnet18(pretrained=True)
        pretrained_num_features = self.pretrained.fc.in_features
        self.pretrained.fc = nn.Linear(in_features = pretrained_num_features, out_features = 10, bias = True)

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass.
        #############################################################################
        images = F.upsample(images, size=(224, 224), mode='bilinear')
        scores = self.pretrained(images)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores
