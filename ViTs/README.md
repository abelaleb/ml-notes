# MobileViT-XXS for CIFAR-10 Classification

This notebook implements a lightweight MobileViT-XXS model for image classification on the CIFAR-10 dataset. It combines mobile-friendly convolutional blocks with a vision transformer architecture to achieve an efficient yet performant model.

## Table of Contents
1.  [Introduction](#introduction)
2.  [Model Architecture](#model-architecture)
3.  [Dataset](#dataset)
4.  [Training Pipeline](#training-pipeline)
5.  [Performance Analysis](#performance-analysis)
6.  [Usage](#usage)

## 1. Introduction
This project demonstrates the implementation and training of a MobileViT-XXS model, a variant of MobileViT designed for mobile and edge devices, on the CIFAR-10 dataset. MobileViT effectively integrates the strengths of convolutions (local processing) and transformers (global processing) within a compact structure.

## 2. Model Architecture
The model is built from several key components:

*   **Inverted Residual Block**: A MobileNetV2-style block used for initial feature extraction and downsampling.
*   **Transformer Encoder**: A lightweight transformer encoder block with MultiheadAttention and an MLP.
*   **MobileViT Block**: The core block that efficiently combines local (convolutional) and global (transformer) feature learning. It uses a unique unfold/fold operation to feed image patches into a transformer while maintaining spatial information.
*   **MobileViT_XXS_CIFAR10**: The full model incorporating these blocks into a network suitable for CIFAR-10, including a stem, three stages (with Inverted Residuals and MobileViT Blocks), and a classification head.

## 3. Dataset

The model is trained and evaluated on the **CIFAR-10 dataset**, which consists of 60,000 32x32 colour images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

Data augmentation techniques such as random cropping and horizontal flipping are applied during training.

## 4. Training Pipeline

The training process involves:

*   **Optimizer**: AdamW with a learning rate of `1e-3` and weight decay of `1e-4`.
*   **Loss Function**: Cross-Entropy Loss.
*   **Epochs**: The model was trained for `20` epochs.
*   **Device**: Training is performed on `cuda` if available, otherwise `cpu`.

## 5. Performance Analysis
After training for 20 epochs, the model achieved the following performance:

*   **Final Training Accuracy**: 86.94%

### Efficiency Metrics

The MobileViT-XXS model's efficiency was analyzed and compared against a reference ResNet-50 model. The results are as follows:

| Model                  | Accuracy (%) | FPS    | Params (M) |
| :--------------------- | :----------- | :----- | :--------- |
| MobileViT-XXS (Yours)  | 86.94        | 162.20 | 0.44       |
| ResNet-50 (Ref)        | 93.0         | 45.0   | 23.5       |

*(Note: FPS measured on CPU/GPU depending on availability during execution)*

### Visual Comparison

The notebook also includes a scatter plot visualizing the trade-off between Inference Speed (FPS) and Accuracy, with bubble size representing the number of parameters, comparing MobileViT-XXS with ResNet-50.

## 6. Usage

To run this notebook and train the MobileViT-XXS model:

1.  Ensure you have a Colab environment or a local Python environment with the necessary PyTorch and torchvision libraries installed.
2.  Execute the cells sequentially.
3.  The notebook will download the CIFAR-10 dataset, define the model architecture, set up the training pipeline, and then train the model.
4.  Performance metrics, including training loss, accuracy, FPS, parameters, and FLOPs, will be displayed upon completion.
