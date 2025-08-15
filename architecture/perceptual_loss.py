# -------------------------------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
from architecture.cnn_architecture import CNN_ARCHITECTURE
from architecture.vit_architectire import VIT_ARCHITECTURE
from architecture.reconstruction_head import RECONSTRUCTION_HEAD
from configs.cnn_branch_config import cnn_experiment_1, cnn_experiment_2


# -------------------------------------------------------------------------------------------------------------
# Perceptual Loss Class
# -------------------------------------------------------------------------------------------------------------
class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15, 22], resize=True):
        """
        Args:
            layer_ids: VGG16 feature layer indices for perceptual loss
            resize: whether to resize input to 224x224 for VGG
        """
        super(PerceptualLoss, self).__init__()
        
        vgg = torchvision.models.vgg16(pretrained=True).features
        self.selected_layers = layer_ids
        self.vgg_layers = nn.ModuleList(vgg).eval()  # freeze weights
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.resize = resize
        self.transform = transforms.Resize((224, 224)) if resize else nn.Identity()

        self.criterion = nn.MSELoss()