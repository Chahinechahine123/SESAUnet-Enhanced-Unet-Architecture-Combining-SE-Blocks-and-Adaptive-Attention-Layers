# SESAUnet: A Deep Attention-Based Segmentation Model for Early Detection of Green Aphids in Smart Greenhouse Environments

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

# 🚀 Optimization & Edge Deployment Pipeline

This section describes the full pipeline to convert, optimize, and deploy the trained U-Net model on NVIDIA edge devices (Jetson Nano, Jetson AGX Orin) using TensorRT FP16 inference.

---

## 📁 Folder Structure

```
Optimization and Containerization/
├── from_h5_to_onnx.py        # Step 1 — Keras H5 → ONNX
├── optimize_onnx.py          # Step 2 — Graph simplification + FP16 quantization
├── inference_tensort.py          # Step 4 — TensorRT inference module
└── Dockerfile                # Containerized deployment
```

> **Step 3 (ONNX → TensorRT `.engine`)** must be executed **directly on the target Jetson device** — TensorRT engines are hardware-specific and cannot be cross-compiled.

---

## ⚙️ Pipeline Overview

| Step | Script | Input | Output | Size |
|------|--------|-------|--------|------|
| 1 | `from_h5_to_onnx.py` | `model.h5` (364.2 MB) | `model.onnx` | 121.3 MB |
| 2 | `optimize_onnx.py` | `model.onnx` | `model_fp16.onnx` | 60.7 MB |
| 3 | `trtexec` on Jetson | `model_fp16.onnx` | `model_tensort.engine` | 29.6 MB |
| 4 | `inference_tensort.py` | `model_tensort.engine` | Predictions + metrics | — |

**Total compression: 364.2 MB → 29.6 MB (12.3× reduction, 1.4% IoU degradation)**

---

## 🔧 Step-by-Step Instructions

### Step 1 — Export Keras model to ONNX

```bash
pip install tensorflow tf2onnx onnx
python from_h5_to_onnx.py
```

### Step 2 — Graph simplification + FP16 quantization

```bash
pip install onnxsim onnxconverter-common
python optimize_onnx.py
```

This produces `model_fp16.onnx` (60.7 MB), ready for TensorRT compilation.

### Step 3 — TensorRT engine compilation *(on Jetson only)*

Copy `model_fp16.onnx` to your Jetson device, then run:

```bash
trtexec \
  --onnx=model_fp16.onnx \
  --saveEngine=model_tensort.engine \
  --fp16 \
  --workspace=1024     # MB — use 4096 for Jetson AGX Orin
```

> ⚠️ This step takes 5–15 minutes on first run. The resulting `.engine` file is specific to the Jetson device it was compiled on.

### Step 4 — Run inference

```bash
pip install pycuda opencv-python scikit-learn pandas tqdm

# Single image
python inference_tensort.py \
  --engine model_tensort.engine \
  --input  image.jpg \
  --output results/

# Batch folder
python inference_tensort.py \
  --engine model_tensort.engine \
  --input  ./test/images/ \
  --output ./results/

# Batch with metrics (requires ground-truth masks)
python inference_tensort.py \
  --engine model_tensort.engine \
  --input  ./test/images/ \
  --output ./results/ \
  --mask   ./test/masks/
```

**Output per image:** side-by-side PNG `[original | ground-truth | prediction]`
**With `--mask`:** additional `metrics.csv` with per-image Accuracy, F1, IoU, Recall, Precision.

---

## 🐳 Docker Deployment

A Docker image encapsulating the TensorRT engine and inference module is publicly available, compatible with **NVIDIA Jetson (JetPack 5.x, TensorRT 8+)** and **standard x86 GPU environments**.

### Prerequisites

- NVIDIA Container Toolkit installed
- `model_tensort.engine` compiled on the target device (see Step 3)

### Build the image

```bash
# Place your compiled model_unet.engine in this folder first
cp /path/to/model_unet.engine .

docker build -t sesaunet-inference .
```

### Run inference with Docker

```bash
# Single image
docker run --rm --runtime=nvidia \
  -v /path/to/image.jpg:/data/image.jpg \
  -v /path/to/results:/data/results \
  unet-inference \
  --engine /app/model_tensort.engine \
  --input  /data/image.jpg \
  --output /data/results/

# Batch folder with metrics
docker run --rm --runtime=nvidia \
  -v /path/to/test/images:/data/images \
  -v /path/to/test/masks:/data/masks \
  -v /path/to/results:/data/results \
  unet-inference \
  --engine /app/model_tensort.engine \
  --input  /data/images \
  --output /data/results \
  --mask   /data/masks
```

### Switch to x86 GPU

Edit the first line of `Dockerfile`:
```dockerfile
# Replace:
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime
# With:
ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:23.10-py3
```

Then rebuild the image.

---

## 📊 Expected Performance

| Device | GPU | FPS (512×512) | Latency |
|--------|-----|---------------|---------|
| Jetson Nano | 128 CUDA cores (Maxwell) | 8–15 FPS | 67–125 ms |
| Jetson AGX Orin | 2048 CUDA cores (Ampere) | 80–120 FPS | 8–12 ms |

> Results are estimates based on TensorRT FP16 benchmarks for equivalent architectures. Empirical measurements on physical hardware will be reported upon availability.

---

## 📦 Model Size Summary

| Artifact | Size | Compression vs FP32 |
|----------|------|----------------------|
| `model.h5` (FP32) | 364.2 MB | 1× (baseline) |
| `model.onnx` | 121.3 MB | 3.0× |
| `model_fp16.onnx` | 60.7 MB | 6.0× |
| `model_tensort.engine` | 29.6 MB | **12.3×** |

**IoU degradation (FP32 → FP16 TensorRT): 1.4%**

---



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

