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
from configs.cnn_branch_config import cnn_experiment_1, cnn_experiment_2

# -------------------------------------------------------------------------------------------------------------
# @CLASS: CNN_VIT_HYBRID_ARCHITECTURE
# -------------------------------------------------------------------------------------------------------------
class CNN_VIT_HYBRID_ARCHITECTURE(nn.Module):
    def __init__(self):
        super(CNN_VIT_HYBRID_ARCHITECTURE, self).__init__()

        # Initialize the architecture
        self.cnn = CNN_ARCHITECTURE(cnn_experiment_2['model_args']['input_size'], cnn_experiment_2['model_args']['hidden_layers'], cnn_experiment_2['model_args']['activation'], cnn_experiment_2['model_args']['norm_layer'], cnn_experiment_2['model_args']['drop_prob'], cnn_experiment_2['model_args']['max_pooling'])
        self.vit = VIT_ARCHITECTURE(cnn_experiment_2['model_args']['model_name'])
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
        print(f"CNN Features: {cnn_features.shape} = {cnn_features.shape[0]} channels, {np.sqrt(cnn_features.shape[1])} height, width")
        H, W = cnn_features.shape

        if x.dim() == 3:
            x = x.unsqueeze(0)

        vit_attention = self.vit(x)
        # vit_attention_map = vit_attention.mean(dim=1).reshape(1, 1, int(H), int(W))
        print(f"Attention shape: {vit_attention.shape}")

        # Fir visual
        vit_attention_map = vit_attention[:, 1, :] #remove CLS token
        print(f"Attention Map after CLS removal: {vit_attention_map.shape}")
        vit_attention_map = vit_attention_map.reshape(1, 1, 197, 197)
        print(f"Attention Map reshaped: {vit_attention_map.shape}")
        

        vit_attention_resized = F.interpolate(vit_attention_map, size=(H, W), mode='bilinear', align_corners=False)
        print(f"Attention Map after interpolation: {vit_attention_resized.shape}")
        vit_attention_resized = vit_attention_resized.expand(-1, 512, -1, -1)
        print(f"Attention Map resized: {vit_attention_resized.shape}")

        combined_features = cnn_features * vit_attention_resized

        output_image = self.reconstructor(vit_attention_resized)

        # ---------------------------------------------------------------------------------------------------
        # Output Visualization
        # ---------------------------------------------------------------------------------------------------
        img = output_image[0].detach().cpu()
        img = (img + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        img_np = img.permute(1, 2, 0).numpy() * 255.0  # (H, W, 3)

        # Clamp to [0, 1] in case of overshoot
        img_np = img_np.clip(0, 255).astype("uint8")

        # Plot image
        plt.figure(figsize=(4, 4))
        plt.title("Deblurred Output")
        plt.imshow(img_np)
        plt.axis("off")
        plt.show()


        # ---------------------------------------------------------------------------------------------------
        # Attention Visualization
        # ---------------------------------------------------------------------------------------------------
        attention_map = vit_attention.squeeze(0)  # -> [12, 197, 197]

        # Extract attention from CLS token (index 0) to all others
        # Shape: [12, 197]
        cls_attn = attention_map[:, 0, :]  # [num_heads, tokens]

        # Drop the CLS token itself for visualization (optional)
        cls_attn = cls_attn[:, 1:]  # [12, 196]

        # Reshape each head's attention to a 14x14 grid (if using 16×16 patches on 224×224 image)
        num_heads = cls_attn.shape[0]
        cls_attn_map = cls_attn.reshape(num_heads, 14, 14)

        # Normalize for visualization
        cls_attn_map = cls_attn_map / cls_attn_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        # Plot each head
        fig, axs = plt.subplots(3, 4, figsize=(12, 9))
        for i, ax in enumerate(axs.flat):
            ax.imshow(cls_attn_map[i].detach().numpy(), cmap='viridis')
            ax.set_title(f'Head {i}')
            ax.axis('off')
        plt.suptitle("CLS Token Attention Maps (14x14) - One per Head")
        plt.tight_layout()
        plt.show()


        # return output_image