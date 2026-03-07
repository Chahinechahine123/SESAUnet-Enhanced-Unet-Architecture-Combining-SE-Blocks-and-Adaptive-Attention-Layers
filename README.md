# SESAUnet: Attention-Enhanced U-Net for Early Aphid Segmentation in Precision Agriculture

## Overview
This repository provides the official implementation of **SESAUnet**, an attention-enhanced U-Net architecture designed for the **early detection of green aphids (*Aphis gossypii*) in greenhouse environments**.

Detecting aphids during early infestation is extremely challenging due to:
- their **small size (≈1–2 mm)**,
- **low contrast** with plant foliage,
- **sparse distribution** across leaf surfaces.

To address these challenges, **SESAUnet** integrates:

- **Squeeze-and-Excitation (SE) blocks** in the encoder to recalibrate channel-wise feature responses.
- **Spatial Attention (SA)** at the decoder output to refine spatial localization of small pests.

This architecture improves segmentation accuracy in cluttered agricultural scenes while preserving the computational efficiency of standard U-Net.

Extensive experiments demonstrate that **SESAUnet achieves superior performance compared with CNN-based, attention-based, transformer-based, and lightweight segmentation models**, reaching:

- **IoU:** 0.836 ± 0.013  
- **Precision:** 0.921 ± 0.011  
- **Recall:** 0.847 ± 0.017  

All results are reported as **mean ± standard deviation across five independent runs**.

The repository also includes **extensive experimental studies**, including:
- architectural ablation experiments,
- sensitivity analyses,
- comparison with 15+ segmentation architectures,
- robustness analysis under noise and illumination variations,
- cross-dataset generalization experiments.

---

# Key Features

### Enhanced Segmentation Architecture
SESAUnet combines:
- **SE channel attention** to strengthen discriminative feature channels.
- **Spatial attention** to focus on aphid regions within cluttered backgrounds.

### Comprehensive Experimental Evaluation
The repository includes experiments covering:
- **CNN-based segmentation models**
- **Attention-enhanced architectures**
- **Transformer-based segmentation networks**
- **Lightweight real-time segmentation models**

### Extensive Ablation Studies
Experiments validate:
- attention module placement
- kernel size selection
- SE reduction ratio
- data augmentation sensitivity

### Robustness and Generalization Analysis
Additional experiments analyze model behavior under:
- Gaussian noise
- extreme illumination changes
- cross-dataset transfer scenarios

### Reproducibility
The repository includes:
- training scripts
- testing pipelines
- evaluation metrics
- FLOPs and parameter calculations
- augmentation pipelines

---

# Dataset

A core contribution of this work is a **high-resolution dataset for early-stage green aphid detection** collected in real greenhouse conditions.

### Dataset Characteristics
- **1,838 images**
- pixel-level aphid segmentation masks
- captured during **early infestation stages**
- real greenhouse illumination conditions
- complex foliage backgrounds

### Annotation
Annotations were generated using **VGG Image Annotator**, producing high-quality semantic segmentation masks.

### Dataset Access
The dataset is publicly available on Kaggle:

https://www.kaggle.com/datasets/chahinebouaziz/green-aphid-early-infestation-dataset

This dataset supports research on **precision agriculture, pest monitoring, and small-object segmentation**.

---

# Repository Structure

```
📂 SESA-UNet
│
├── 📂 Ablation_Studies
│
│ ├── 📂 Architectural_Ablation_and_Components
│ │ ├── Unet_sa_decoder_ALLstages
│ │ ├── Unet_sa_decoder_out
│ │ ├── Unet_sa_skip_connection
│ │ ├── Unet_SE_encoder_decoder
│ │ └── Unet_SE_only_encoder
│ │
│ ├── 📂 Sensitivity_Analysis_of_SA_Kernel_Size
│ │ ├── SESAUnet_3_kernel
│ │ ├── SESAUnet_5_kernel
│ │ └── SESAUnet_7_kernel
│ │
│ └── 📂 Sensitivity_Analysis_of_SE_Reduction_Ratio
│ ├── SESAUnet_SE_ratio_8
│ ├── SESAUnet_SE_ratio_16
│ └── SESAUnet_SE_ratio_32
│
├── 📂 Comparaison_Models
│
│ ├── 📂 Attention_enhanced_models
│ │ ├── Attention_unet.py
│ │ ├── cbam_unet.py
│ │ ├── transunet.py
│ │ ├── swin_unet.py
│ │ └── psnet.py
│ │
│ ├── 📂 CNN_Based_segmentation_Models
│ │ ├── unet.py
│ │ ├── segnet.py
│ │ ├── fcn8s.py
│ │ ├── deeplabv3_resnet50.py
│ │ ├── deeplabv3_mobilenetv2.py
│ │ └── convnext_unet.py
│ │
│ └── 📂 Lightweight_Models
│ ├── mobileunet.py
│ ├── fastscnn.py
│ └── espnetv2.py
│
├── 📂 Data_Augmentation
│ ├── DataAug.py
│ └── online_augmentation.py
│
├── 📂 Generalization_and_Robustness_Analysis
│ ├── from_cocoJSON_to_binarymask.py
│ └── perturbations_on_testSet.py
│
├── 📂 Metrics_and_FLOPS
│ ├── metrics.py
│ ├── flops_calculation.py
│ └── prms_memoryFootprint_calculation.py
│
├── 📂 SESAUnet
│ └── SESAUnet.py
│
├── 📂 Train_and_Test
│ ├── train.py
│ ├── test.py
│ └── train_with_online_augmentation.py
│
└── README.md
```


---

# Installation

### Requirements

- Python ≥ 3.8
- TensorFlow 2.17
- Keras 3.4
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn
- Albumentations

### Setup

Clone the repository:

```bash
git clone https://github.com/Chahinechahine123/SESA-UNet.git
cd SESA-UNet

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
SESAUnet was evaluated on the custom aphid dataset, outperforming baselines
An ablation study confirms the benefits of SE blocks (improving feature sensitivity) and spatial attention (enhancing localization in cluttered scenes). For full details, refer to the accompanying manuscript.

## Citation
If you use SESAUnet or the dataset in your research, please cite:
```
@article{bouaziz2025sesaunet,
  title={SESAUnet: A Deep Attention-Based Segmentation Model for Early Detection of Green Aphids in Smart Greenhouse Environments},
  author={Bouaziz, Chahine},
  year={2026},
  journal={TBD}
}
```

## Contact
For questions, issues, or collaborations, contact **Chahine Bouaziz** at **bouazizchahine7@gmail.com**. Feel free to open an issue on GitHub.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (add if not present).

---

Thank you for using SESAUnet! Contributions are welcome to advance AI in sustainable agriculture.

