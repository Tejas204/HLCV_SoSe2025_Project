# HLCV_SoSe2025_Project

### Introduction
Images are affected by various artifacts that distort or degrade their quality. An image can experience distortion due to blur, noise or, low resolution. These artifacts degrade the quality of the image rendering it useless for further applications. We propose a light-weight, hybrid CNN-ViT architecture that helps remove these distortion artifacts from an image, thus making degraded images useful for other high-level computer vision or post-processing tasks.

### Model
This repository contains the code for CNN-ViT hybrid architecture. an image containing multiple artifacts is input to both the convolutional block and the ViT block, as shown in figure 2. The CNN is expected to capture local features such as edges and textures, whereas the ViT focuses on global features like object shapes. The features from both branches are then fused through an element-wise product and fed to the decoder. Thus, the less significant features captured by the attention mechanism gain some significance while the features that are already significant receive a boost.

![](/assets/Cnn-ViT-Dot-Product.drawio.png)
