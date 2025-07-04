# -------------------------------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from architecture.cnn_architecture import CNN_ARCHITECTURE
from architecture.vit_architectire import VIT_ARCHITECTURE
from architecture.reconstruction_head import RECONSTRUCTION_HEAD
from configs.cnn_branch_config import cnn_experiment_1

# -------------------------------------------------------------------------------------------------------------
# @CLASS: CNN_VIT_HYBRID_ARCHITECTURE
# -------------------------------------------------------------------------------------------------------------
class CNN_VIT_HYBRID_ARCHITECTURE(nn.Module):
    def __init__(self):
        super(CNN_VIT_HYBRID_ARCHITECTURE, self).__init__()

        # Initialize the architecture
        self.cnn = CNN_ARCHITECTURE(cnn_experiment_1['model_args']['input_size'], cnn_experiment_1['model_args']['hidden_layers'], cnn_experiment_1['model_args']['activation'], cnn_experiment_1['model_args']['norm_layer'], cnn_experiment_1['model_args']['drop_prob'])
        self.vit = VIT_ARCHITECTURE(cnn_experiment_1['model_args']['model_name'])
        self.reconstructor = RECONSTRUCTION_HEAD(512,3)

    
    """-------------------------------------------------------------------------------------------------------------
    @Function: 
        forward
    @Args: 
        self: object
        x: torch.Tensor
            The input tensor
    @Returns: 
        output_image
    @Description: 
        Creates the CNN architecture, combined the features, decodes them and reconstructs the deblurred image
    -------------------------------------------------------------------------------------------------------------"""
    def forward(self, x):
        cnn_features = self.cnn(x)
        print(x.shape)
        H, W = cnn_features.shape

        if x.dim() == 3:
            x = x.unsqueeze(0)

        vit_attention = self.vit(x)
        # vit_attention_map = vit_attention.mean(dim=1).reshape(1, 1, int(H), int(W))

        # Fir visual
        vit_attention_map = vit_attention[:, 1, :] #remove CLS token
        print(f"vit_attention_map.shape: {vit_attention_map.shape}")
        vit_attention_map = vit_attention_map.reshape(1, 1, 197, 197)
        

        vit_attention_resized = F.interpolate(vit_attention_map, size=(H, W), mode='bilinear', align_corners=False)
        vit_attention_resized = vit_attention_resized.expand(-1, 512, -1, -1)

        combined_features = cnn_features * vit_attention_resized

        output_image = self.reconstructor(combined_features)

        # Visualization
        img = output_image[0].detach().cpu()
        img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        img_np = img.permute(1, 2, 0).numpy()  # (H, W, 3)

        # Clamp to [0, 1] in case of overshoot
        img_np = img_np.clip(0, 1)

        # Plot image
        plt.figure(figsize=(4, 4))
        plt.title("Deblurred Output")
        plt.imshow(img_np)
        plt.axis("off")
        plt.show()


        return output_image