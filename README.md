# SESAUnet: Enhanced U-Net Architecture Combining SE Blocks and Adaptive Attention Layers

## Overview
This repository implements **SESAUnet**, a novel deep learning segmentation model designed for early-stage pest detection in precision agriculture, with a focus on green aphids (*Aphis gossypii*). Built upon the U-Net architecture, SESAUnet enhances feature extraction and localization by integrating **Squeeze-and-Excitation (SE) blocks** in the encoder to recalibrate channel-wise features and a **spatial attention mechanism** at the decoder output to refine segmentation in complex, cluttered environments. This addresses key challenges in detecting small, low-contrast pests that blend with foliage, enabling timely interventions to reduce crop losses, pesticide usage, and environmental impact.

The model is tailored for real-time deployment in smart greenhouses, offering superior performance over traditional object detection methods (e.g., YOLO, Faster R-CNN) by providing pixel-level granularity. Experimental results demonstrate that SESAUnet outperforms state-of-the-art segmentation models like U-Net, DeepLabV3+, FCN8s, and SegNet, achieving an Intersection over Union (IoU) of 0.804 and a precision of 0.912 on a custom dataset of early aphid infestations.

This work contributes to sustainable agriculture by providing a scalable, AI-based solution for automated pest monitoring. The repository includes the core model implementation, training scripts, data augmentation utilities, and evaluation tools for benchmarking against baseline models.

## Key Features
- **Enhanced Architecture**: Combines SE blocks for channel attention and spatial attention for precise localization of small objects in noisy agricultural scenes.
- **Data Augmentation**: Domain-specific augmentations (e.g., noise simulation, lighting variations, scale changes) to improve model robustness.
- **Evaluation Suite**: Scripts to compare SESAUnet with U-Net, SegNet, DeepLabV3+, and FCN8s using metrics like IoU, precision, recall, and F1-score.
- **Real-World Focus**: Validated on high-resolution images from pepper greenhouses, capturing early infestation phases rarely addressed in existing datasets.
- **Open-Source Resources**: Full code and a public dataset to promote reproducibility and further research in precision agriculture.

## Dataset
A core contribution of this project is a high-resolution dataset specifically curated for early-stage green aphid detection. The dataset consists of images manually collected from real pepper greenhouses during the initial infestation phase, where aphids are sparse, small (1-2 mm), and visually similar to plant foliage. This underrepresented scenario includes complex backgrounds, variable illumination, and visual clutter, making it ideal for training robust AI models.

- **Size and Composition**: The dataset includes [1,838 images] with pixel-level annotations for aphid segmentation.
- **Annotations**: Fine-grained masks created using tools  VGG Image Annotator, focusing on precise boundaries.
- **Augmentation**: Applied techniques simulate real-world variations, such as sensor noise, distance-related scaling, and lighting changes.
- **Access**: The dataset is publicly available on Kaggle for download and use in research.  
  Link: [https://www.kaggle.com/datasets/chahinebouaziz/green-aphid-early-infestation-dataset](https://www.kaggle.com/datasets/chahinebouaziz/green-aphid-early-infestation-dataset) (Note: Replace with actual link if different; based on project details).

This dataset enables reliable training for early-warning systems and can be extended for other pest detection tasks in precision agriculture.

## Repository Structure
```
📂 SESA-UNet
│
├── 📂 SESAUNET
│   ├── DataAug.py       # Utilities for data augmentation (e.g., flips, rotations, noise, brightness adjustments)
│   ├── metrics.py       # Custom metrics for evaluation (IoU, precision, recall, F1-score)
│   ├── SESAUnet.py      # Core implementation of the SESAUnet architecture with SE blocks and spatial attention
│   ├── Train.py         # Script for training SESAUnet on the aphid dataset
│
├── 📂 evaluation
│   ├── unet.py          # Baseline U-Net implementation for comparison
│   ├── segnet.py        # Baseline SegNet implementation
│   ├── fcn8s.py         # Baseline FCN8s implementation
│   ├── deeplabv3.py     # Baseline DeepLabV3+ implementation
│   ├── Test.py          # Script for evaluating and comparing models on test data
│
├── README.md            # This documentation file
```

## Installation
### Prerequisites
- Python 3.8 or higher
- TensorFlow 2.17.0
- Keras 3.4.1
- NumPy
- OpenCV (cv2)
- Matplotlib
- Scikit-learn
- Albumentations (for advanced data augmentation)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Chahinechahine123/SESA-UNet.git
   cd SESA-UNet
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install tensorflow==2.17.0 keras==3.4.1 numpy opencv-python matplotlib scikit-learn albumentations
   ```

## Usage
### Training SESAUnet
1. Download the dataset from Kaggle and place it in a `data/` directory (create if needed).
2. Run the training script:
   ```bash
   python SESAUNET/Train.py --data_path data/ --epochs 50 --batch_size 16
   ```
   - Customize hyperparameters like learning rate, epochs, or batch size via command-line arguments.
   - The script applies data augmentation from `DataAug.py` and saves the trained model.

### Evaluation and Comparison
1. Use the evaluation script to test models:
   ```bash
   python evaluation/Test.py --model sesaunet --test_path data/test/
   ```
   - Options: `--model` can be `sesaunet`, `unet`, `segnet`, `fcn8s`, or `deeplabv3`.
   - Outputs metrics and visualizations (e.g., segmentation masks overlaid on images).


## Results
SESAUnet was evaluated on the custom aphid dataset, outperforming baselines:
- **IoU**: 0.804 (vs. U-Net: 0.752, DeepLabV3+: 0.781, FCN8s: 0.723, SegNet: 0.698)
- **Precision**: 0.912
- **Recall**: 0.845
- **F1-Score**: 0.878

An ablation study confirms the benefits of SE blocks (improving feature sensitivity) and spatial attention (enhancing localization in cluttered scenes). For full details, refer to the accompanying manuscript.

## Citation
If you use SESAUnet or the dataset in your research, please cite:
```
@article{bouaziz2025sesaunet,
  title={SESAUnet: A Deep Attention-Based Segmentation Model for Early Detection of Green Aphids in Smart Greenhouse Environments},
  author={Bouaziz, Chahine},
  year={2025},
  journal={TBD}
}
```

## Contact
For questions, issues, or collaborations, contact **Chahine Bouaziz** at **bouazizchahine7@gmail.com**. Feel free to open an issue on GitHub.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (add if not present).

---

Thank you for using SESAUnet! Contributions are welcome to advance AI in sustainable agriculture.

