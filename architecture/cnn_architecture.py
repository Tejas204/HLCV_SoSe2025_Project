# -------------------------------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

# -------------------------------------------------------------------------------------------------------------
# @CLASS: CNN_ARCHITECTURE
# -------------------------------------------------------------------------------------------------------------
class CNN_ARCHITECTURE(nn.Module):
    def __init__(self, input_size, hidden_layers, activation, norm_layer, max_pooling, drop_prob=0.0):
        super(CNN_ARCHITECTURE, self).__init__()

        # Initialize variables
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.norm_layer = norm_layer
        self.drop_prob = drop_prob
        self.max_pooling = max_pooling

        self.build_model()

    """-------------------------------------------------------------------------------------------------------------
    @Function: 
        build_model
    @Args: 
        self
    @Returns: 
        model
            The model layers for performing the convolution operation
    @Description: 
        Create a CNN architecture that extracts the features from the input images
    -------------------------------------------------------------------------------------------------------------"""
    def build_model(self):

        layers = []
        input_dim = self.input_size
        for i in range(len(self.hidden_layers)):
            # Add the convoluiton layer
            # Conv2D: inout_dim := num of channels, self.hidden_layers = channels of filter of size 3, with padding 2, stride 1
            layers.append(nn.Conv2d(input_dim, self.hidden_layers[i], 3, stride=1, padding=1))

            # Add batch normalization
            if self.norm_layer:
                layers.append(self.norm_layer(self.hidden_layers[i]))
            
            # Add maxpooling, change this condition
            if self.max_pooling:
                layers.append(nn.MaxPool2d(2,2))
            
            # Add activation
            layers.append(self.activation())

            # Add dropout
            if self.drop_prob:
                layers.append(nn.Dropout(self.drop_prob))

            input_dim = self.hidden_layers[i]
        
        self.features = nn.Sequential(*layers)

    """-------------------------------------------------------------------------------------------------------------
    @Function: 
        forward
    @Args: 
        self: object
        x: torch.Tensor
            The input tensor
    @Returns: 
        torch.Tensor
            The output tensor containing the features
    @Description: 
        Passes the input to the CNN architecture for feature extraction
    -------------------------------------------------------------------------------------------------------------"""
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        return features
    

    """-------------------------------------------------------------------------------------------------------------
    @Function: image_denoising
    @Args: 
        self: object
    @Returns: denoised images
    @Description: A denoising filter to remove the noise from the input images
    -------------------------------------------------------------------------------------------------------------"""
    def image_denoising(self):
        print("I am denoising")