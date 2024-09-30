## Galaxy Morphology Classification

## Overview

A Machine Learning model built with `PyTorch`, trained on several datasets to classify galaxies by their morphological shapes (Elliptical, Spiral, & Irregular). Currently the model reaches a training accuracy of 99% and a validational accuracy of around 91%.

### Pre-Processing
- `Image Resizing` | To ensure consistency across the entire dataset, all images were resized to a fixed 224 x 224.
- `Augmentation` | Increasing the datasets variability & robustness by adjusting rotation, brightness, and other factors.
- `Normalization` | Pixel sizes were normalized to ensure a more stable gradient descent, and faster convergence.

### Model Architecture
- `Convolutional Layers` | To clear up background noise and spatial features.
- `Pooling Layers` | Further reducing size to only highlight the primary subject.
- `Connected Layers` | Classify each data piece based on the extracted features.

### Training Process
- `Loss Function` | Leveraging categorical cross-entropy, to determine how close the models predictions were.
- `Adam Optimizer` | An Adam optimizer was utilized to tweak model settings for more efficient convergence during training.
- `Relevant Metrics` | Tracking Accuracy and Loss to determine the effeciveness of each Epoch.
