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

class VIT_ARCHITECTURE():
    def __init__(self, model_name):
        super(VIT_ARCHITECTURE, self).__init__()

        # Initialize variables
        self.model_name = model_name
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
        self.model = ViTModel.from_pretrained(model_name)

        self.model.eval()

    def preprocess(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items()}

    def extract_features(self, image_path, return_cls=True, return_all=True):
        inputs = self.preprocess(image_path)
        with torch.no_grad():
            outputs = self.model(**inputs)

        features = {}
        if return_cls:
            features['cls'] = outputs.last_hidden_state[:, 0]  # [1, 768]
        if return_all:
            features['all'] = outputs.last_hidden_state       # [1, 197, 768]
        return features
