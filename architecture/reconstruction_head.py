import torch
import torch.nn as nn

class RECONSTRUCTION_HEAD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RECONSTRUCTION_HEAD, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Assuming image values are normalized between [-1, 1]
        )

    def forward(self, x):
        return self.decoder(x)