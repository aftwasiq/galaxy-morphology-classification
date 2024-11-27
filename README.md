## Galaxy Morphology Classification
 
### Project Overview

A Machine Learning model built with `PyTorch`, trained on several datasets to classify galaxies by their morphological shapes (Elliptical, Spiral, & Irregular). Currently the model reaches a training accuracy of 99% and a validational accuracy of around 91%.

<img src="https://github.com/user-attachments/assets/367e20c0-6767-4762-b95a-7646ba135d69" width="1200">

*I'm having issues with deploying yhe demo site, so if you'd like to test this out yourself, download the contents, open it in an IDE, and run it locally via `python main.py`. Make sure you have all dependancies and libraries installed.*

### Pre-Processing
Preprocessing the selected images to ensure a non-biased training sequence.
- `Image Resizing` | To ensure consistency across the entire dataset, all images were resized to a fixed 224 x 224.
- `Augmentation` | Increasing the datasets variability & robustness by adjusting rotation, brightness, etc.
- `Normalization` | Pixel sizes were normalized to ensure a more stable gradient descent, and faster convergence.

### Model Architecture
Utilizing a Convolutional Neural Network model, comprising of,
- `Convolutional Layers` | To clear up background noise and spatial features.
- `Pooling Layers` | Further reducing size to only highlight the primary focus of each image (in this case, the galaxy)
- `Fully Connected Layers` | Classify each data piece based on the extracted features.

### Training Process
Trained on around 30 epochs using a training-validation split
- `Loss Function` | Leveraging categorical cross-entropy, to determine how close the models predictions were.
- `Adam Optimizer` | An Adam optimizer was to minimize losss for more efficient convergence during training.
- `Relevant Metrics` | tracking Accuracy and Loss to determine the effeciveness of each Epoch.


### Directory Structure
```
galaxy-morphology-classification/
│
├── dataset/
│   ├── train/                 # training dataset
|     ├── elliptical 
|     ├── irregular
|     └── spiral
│   └── val/                   # validation dataset
|     ├── elliptical 
|     ├── irregular
|     └── spiral
│
├── static/
│   └── background.jpg          # stuff for the demo frontend
│
├── templates/
│   └── index.html              # quick frontend demo page if you wanna test out the model          
│
├── app.py                      # training + flask route to demo
├── main.py                     # cnn model
├── README.md                   # documentation
└── requirements.txt            # packages
```
