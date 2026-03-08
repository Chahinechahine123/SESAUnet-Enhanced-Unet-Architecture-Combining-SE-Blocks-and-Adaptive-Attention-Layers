# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — SESAUNet TensorRT Inference
# Compatible with:
#   • NVIDIA Jetson (JetPack 5.x, TensorRT 8+)
#   • x86 GPU (CUDA 11.8+, TensorRT 8+)
#
# Build:
#   docker build -t sesaunet-inference .
#
# Run (single image):
#   docker run --rm --runtime=nvidia \
#     -v /path/to/images:/data/images \
#     -v /path/to/results:/data/results \
#     unet-inference \
#     python inference_tensort.py --input /data/images --output /data/results
# ─────────────────────────────────────────────────────────────────────────────

# Base image — NVIDIA L4T for Jetson (JetPack 5.x) / swap for x86 if needed
# Jetson (ARM64):  nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime
# x86 GPU:        nvcr.io/nvidia/tensorrt:23.10-py3
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime
FROM ${BASE_IMAGE}

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL maintainer="bouazizchahine7@gmail.com"
LABEL description="SESAUNet segmentation inference — TensorRT FP16"
LABEL version="1.0"

# ── System dependencies ───────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
RUN pip3 install --no-cache-dir \
    numpy==1.23.5 \
    opencv-python-headless==4.8.1.78 \
    pycuda==2022.2.2 \
    scikit-learn==1.3.2 \
    pandas==2.0.3 \
    tqdm==4.66.1

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Copy model + inference script ────────────────────────────────────────────
# ⚠️  model_tensort.engine must be compiled ON the target Jetson device first
#     then placed in this directory before building the image
COPY model_tensort.engine   /app/model_tensort.engine
COPY inference_tensort.py    /app/inference_tensort.py

# ── Environment ───────────────────────────────────────────────────────────────
ENV ENGINE_PATH=/app/model_tensort.engine
ENV PYTHONUNBUFFERED=1

# ── Default command ───────────────────────────────────────────────────────────
ENTRYPOINT ["python3", "inference_tensort.py"]
CMD ["--help"]
