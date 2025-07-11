import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

import torchvision
import torchvision.transforms as transforms
from architecture.cnn_vit_hybrid_architecture import CNN_VIT_HYBRID_ARCHITECTURE


class TRAINER_EVALUATOR():
    def __init__(self):
        super(TRAINER_EVALUATOR, self).__init__()

    """-------------------------------------------------------------------------------------------------------------
    @Function: 
        train_pretrained_cnn_vit_no_patch
    @Args: 
        self: object
        model
        dataloader
        optimizer
        epochs
        log_file
        best_model_dir
    @Returns: 
        none
    @Description: 
        training loop for the CNN-ViT hybrid with no patching and pre-trained weights
    -------------------------------------------------------------------------------------------------------------"""
    def train_pretrained_cnn_vit_no_patch(model, dataloader, optimizer, criterion, epochs=100, log_file='training_log.txt', best_model_dir='/best_models/cnn_vit_dot_product_no_patch/'):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        criterion = nn.MSELoss()

        os.makedirs(best_model_dir, exist_ok=True)
        best_loss = float('inf')

        # Define structures for capturing loss and epochs
        history = {
            "train_loss": [],
            "epochs": []
        }

        # Open log file
        with open(log_file, 'w') as log:
            for epoch in range(epochs):
                model.train()
                total_loss = 0.0

                for blurry, sharp in dataloader:
                    blurry = blurry.to(device)
                    sharp = sharp.to(device)

                    optimizer.zero_grad()
                    output = model(blurry)
                    loss = criterion(output, sharp)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(dataloader)
                history['train_loss'].append(avg_loss)
                history['epochs'].append(epoch)

                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(model.state_dict(), os.path.join(best_model_dir, 'best_model.pth'))

                # Logging
                log_line = f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}\n"
                print(log_line.strip())
                log.write(log_line)
                log.flush()

            # Visualization
            plt.plot(np.array(history['epochs']), np.array(history['train_loss']))
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Training Loss vs Epoch for Pretrained CNN-ViT Hybrid with no patching")
            plt.show()

    """-------------------------------------------------------------------------------------------------------------
    @Function: 
        train_pretrained_cnn_vit_no_patch
    @Args: 
        self: object
        model
        dataloader
        optimizer
        epochs
        log_file
        best_model_dir
    @Returns: 
        none
    @Description: 
        training loop for the CNN-ViT hybrid with no patching and pre-trained weights
    -------------------------------------------------------------------------------------------------------------"""
    def eval_pretrained_cnn_vit_no_patch(model, val_loader, criterion, device):
        model.eval()
        total_loss = 0.0
        psnr_total = 0.0
        ssim_total = 0.0
        image_count = 0

        with torch.no_grad():
            for blur, sharp in val_loader:
                blur = blur.to(device)
                sharp = sharp.to(device)

                outputs = model(blur)
                loss = criterion(outputs, sharp)
                total_loss += loss.item()

                # Convert output and ground truth to numpy arrays
                outputs_cpu = np.array(outputs)
                targets_cpu = np.array(sharp)

                
                out_img = np.clip(outputs_cpu, 0, 1)
                tgt_img = np.clip(targets_cpu, 0, 1)

                psnr_total += compute_psnr(tgt_img, out_img, data_range=1.0)
                ssim_total += compute_ssim(tgt_img, out_img, data_range=1.0, channel_axis=2)
                image_count += 1

        avg_loss = total_loss / len(val_loader.dataset)
        avg_psnr = psnr_total / image_count
        avg_ssim = ssim_total / image_count

        return avg_loss, avg_psnr, avg_ssim