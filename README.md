# Enhanced U-Net Architecture Combining SE Blocks and Adaptive Attention Layers

## Overview
This repository contains the implementation of **SESA-UNet**, an enhanced deep learning architecture based on U-Net. The model integrates **Squeeze-and-Excitation (SE) blocks** and a **spatial attention mechanism** to improve feature extraction, refine object localization, and enhance segmentation performance in complex environments.

Additionally, this repository includes evaluation scripts to benchmark **SESA-UNet** against standard segmentation models such as **U-Net, SegNet, DeepLabV3+, and FCN8s**.

## Repository Structure
```
📂 SESA-UNet
│
├── 📂 SESAUNET
│   ├── DataAug.py  # Data augmentation utilities
│   ├── metrics.py 
│   ├── SESAUnet.py   # Implementation of SESA-UNet architecture
│   ├── Train.py   # Training script for SESA-UNet
    
│
├── 📂 evaluation
│   ├── unet.py      # Standard U-Net implementation
│   ├── segnet.py    # SegNet implementation
│   ├── fcn8s.py     # Fully Convolutional Network (FCN8s) implementation
│   ├── deeplabv3.py # DeepLabV3+ implementation
│   ├── Test.py      # Evaluation script for model comparison
│
├── README.md        # Project documentation
```

## Installation
### Prerequisites
- Python 3.8+
- TensorFlow (2.17.0)
- Keras (3.4.1)
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- Albumentations (for data augmentation)

### Setup
Clone the repository:
```bash
git clone https://github.com/Chahinechahine123/SESA-UNet.git
cd SESA-UNet
```

## Contact
For any questions, feel free to contact **Chahine BOUAZIZ** at **bouazizchahine7@gmail.com** or open an issue in this repository.

---

Enjoy using SESA-UNet for your research!

"# SESAUnet-Enhanced-Unet-Architecture-Combining-SE-Blocks-and-Adaptive-Attention-Layers" 
