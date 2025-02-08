# ECG Atrial Fibrillation Detection using SwinMobileNet

This project implements a deep learning model for atrial fibrillation (AF) detection using ECG signals. The approach involves transforming ECG signals into time-frequency diagrams, which are then used to train a SwinMobileNet-based model for classification.

## Overview

1. **Time-Frequency Transformation**: The first step in the process is converting raw ECG signals into time-frequency diagrams using techniques like wavelet transforms or short-time Fourier transform (STFT). These diagrams capture both the time and frequency characteristics of the ECG signals, allowing the model to better understand the underlying patterns.

2. **SwinMobileNet Architecture**: The model is based on the SwinMobileNet network architecture, a lightweight yet powerful model designed for mobile and edge devices. This architecture builds upon the ideas of Swin Transformer, a model that utilizes vision transformers for image processing tasks, which have proven effective in capturing both local and global dependencies in visual data.

3. **Atrial Fibrillation Detection**: The transformed time-frequency diagrams are used as input to the SwinMobileNet model, which is trained to classify the signals as either normal sinus rhythm (NSR) or atrial fibrillation (AF).

## Dependencies

Make sure to install the following libraries to run the project:

```bash
pip install -r requirements.txt
