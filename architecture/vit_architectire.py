# -------------------------------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models import ViT_B_16_Weights

from transformers import ViTFeatureExtractor, ViTModel

class VIT_ARCHITECTURE(nn.Module):
    def __init__(self, model_name):
        super(VIT_ARCHITECTURE, self).__init__()

        # Initialize variables
        self.model_name = model_name
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)

        # self.model.eval()

    """-------------------------------------------------------------------------------------------------------------
    @Function: 
        preprocess
    @Args: 
        self
        image_path
            Path to the image
    @Returns: 
        features
            Extracted final attention layer
    @Description: 
        Preprocesses the image in a format required by the feature extractor
    -------------------------------------------------------------------------------------------------------------"""
    def preprocess(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return {k: v for k, v in inputs.items()}
    
    """-------------------------------------------------------------------------------------------------------------
    @Function: 
        extract_features
    @Args: 
        self
        image_path
            Path to the image
    @Returns: 
        features
            Extracted final attention layer
    @Description: 
        Extracts the last layer features and returns
    -------------------------------------------------------------------------------------------------------------"""
    def forward(self, x, return_cls=True, return_all=True):
        # inputs = self.preprocess(image_path)
        # with torch.no_grad():
        #     outputs = self.model(**inputs)

        # features = {}
        # if return_cls:
        #     features['cls'] = outputs.last_hidden_state[:, 0]  # [1, 768]
        # if return_all:
        #     features['all'] = outputs.last_hidden_state       # [1, 197, 768]
        # return features
        outputs = self.model(x, output_attentions=True)
        last_attention_layer = outputs.attentions[-1]
        return last_attention_layer
