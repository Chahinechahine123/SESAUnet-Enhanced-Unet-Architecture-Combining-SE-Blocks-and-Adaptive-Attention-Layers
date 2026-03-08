"""
inference_trt.py
─────────────────────────────────────────────────────────────
TensorRT FP16 inference script for U-Net segmentation model.
Compatible with NVIDIA Jetson (JetPack 5.x) and x86 GPU.

Usage:
    # Single image
    python inference_tensort.py --input image.jpg --output result.png

    # Folder (batch)
    python inference_tensort.py --input ./images/ --output ./results/

    # With metrics (requires mask folder)
    python inference_tensort.py --input ./images/ --output ./results/ --mask ./masks/
─────────────────────────────────────────────────────────────
"""

import os
import argparse
import time
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

# ── TensorRT + PyCUDA ─────────────────────────────────────────────────────────
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# ── Optional metrics ──────────────────────────────────────────────────────────
try:
    from sklearn.metrics import jaccard_score, f1_score, accuracy_score, \
                                precision_score, recall_score
    import pandas as pd
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
H, W          = 512, 512
ENGINE_PATH   = os.environ.get("ENGINE_PATH", "model_tensort.engine")
TRT_LOGGER    = trt.Logger(trt.Logger.WARNING)


# ─────────────────────────────────────────────────────────────────────────────
# Engine loader
# ─────────────────────────────────────────────────────────────────────────────
def load_engine(engine_path: str):
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())


# ─────────────────────────────────────────────────────────────────────────────
# Inference session (allocate buffers once, reuse)
# ─────────────────────────────────────────────────────────────────────────────
class TRTSession:
    def __init__(self, engine_path: str):
        print(f"[INFO] Loading TensorRT engine: {engine_path}")
        self.engine  = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()

        # Allocate host + device buffers
        self.h_input  = cuda.pagelocked_empty((1, 3, H, W), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty((1, 1, H, W), dtype=np.float32)
        self.d_input  = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        print(f"[INFO] Engine ready — input {self.h_input.shape}, output {self.h_output.shape}")

    def preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        """Resize, normalize, CHW conversion."""
        img = cv2.resize(bgr_image, (W, H))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))        # HWC → CHW
        return np.expand_dims(img, axis=0)         # (1, 3, H, W)

    def infer(self, bgr_image: np.ndarray) -> np.ndarray:
        """Run single-image inference. Returns binary mask (H, W) uint8."""
        np.copyto(self.h_input, self.preprocess(bgr_image))

        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v2(
            bindings=[int(self.d_input), int(self.d_output)],
            stream_handle=self.stream.handle
        )
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()

        mask = self.h_output[0, 0]                 # (H, W) float32
        return (mask > 0.5).astype(np.uint8) * 255


# ─────────────────────────────────────────────────────────────────────────────
# Result saving (image | mask | prediction)
# ─────────────────────────────────────────────────────────────────────────────
def save_result(image: np.ndarray, gt_mask: np.ndarray,
                pred_mask: np.ndarray, save_path: str):
    sep = np.ones((H, 10, 3), dtype=np.uint8) * 128

    def to3ch(m):
        m = cv2.resize(m, (W, H))
        return np.stack([m, m, m], axis=-1)

    row = [image]
    if gt_mask is not None:
        row += [sep, to3ch(gt_mask)]
    row += [sep, to3ch(pred_mask)]

    cv2.imwrite(save_path, np.concatenate(row, axis=1))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="U-Net TensorRT Inference")
    parser.add_argument("--input",   required=True,  help="Image file or folder")
    parser.add_argument("--output",  required=True,  help="Output folder")
    parser.add_argument("--mask",    default=None,    help="Ground-truth mask folder (optional, enables metrics)")
    parser.add_argument("--engine",  default=ENGINE_PATH, help="Path to .engine file")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Collect inputs
    if os.path.isdir(args.input):
        images = sorted(glob(os.path.join(args.input, "*")))
        images = [p for p in images if p.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff'))]
    else:
        images = [args.input]

    print(f"[INFO] Found {len(images)} image(s)")

    # Load masks if provided
    masks = None
    if args.mask and os.path.isdir(args.mask):
        masks = sorted(glob(os.path.join(args.mask, "*")))
        masks = [p for p in masks if p.lower().endswith(('.jpg','.jpeg','.png','.tif','.tiff'))]
        assert len(masks) == len(images), \
            f"Image/mask count mismatch: {len(images)} vs {len(masks)}"

    # Init TRT session
    session = TRTSession(args.engine)

    SCORE  = []
    times  = []

    for idx, img_path in enumerate(tqdm(images, desc="Inference")):
        name  = os.path.splitext(os.path.basename(img_path))[0]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] Cannot read {img_path}, skipping.")
            continue

        # ── Inference ────────────────────────────────────────────────────────
        t0   = time.time()
        pred = session.infer(image)
        times.append((time.time() - t0) * 1000)

        # ── Load GT mask ──────────────────────────────────────────────────────
        gt = None
        if masks:
            gt = cv2.imread(masks[idx], cv2.IMREAD_GRAYSCALE)

        # ── Save visual result ────────────────────────────────────────────────
        save_result(
            cv2.resize(image, (W, H)), gt, pred,
            os.path.join(args.output, f"{name}_result.png")
        )

        # ── Metrics ───────────────────────────────────────────────────────────
        if gt is not None and METRICS_AVAILABLE:
            y_true = (cv2.resize(gt, (W, H)) > 127).astype(np.int32).flatten()
            y_pred = (pred > 127).astype(np.int32).flatten()
            SCORE.append([
                name,
                accuracy_score(y_true, y_pred),
                f1_score(y_true, y_pred, average="binary", zero_division=1),
                jaccard_score(y_true, y_pred, average="binary", zero_division=1),
                recall_score(y_true, y_pred, average="binary", zero_division=1),
                precision_score(y_true, y_pred, average="binary", zero_division=1),
            ])

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_ms = np.mean(times)
    fps    = 1000.0 / avg_ms

    print(f"\n{'─'*50}")
    print(f"  Images processed : {len(times)}")
    print(f"  Avg latency      : {avg_ms:.2f} ms")
    print(f"  Throughput       : {fps:.1f} FPS")
    print(f"  Results saved to : {args.output}")

    if SCORE:
        arr   = np.mean([s[1:] for s in SCORE], axis=0)
        print(f"\n  Accuracy  : {arr[0]:.5f}")
        print(f"  F1        : {arr[1]:.5f}")
        print(f"  IoU       : {arr[2]:.5f}")
        print(f"  Recall    : {arr[3]:.5f}")
        print(f"  Precision : {arr[4]:.5f}")

        df = pd.DataFrame(SCORE, columns=["Image","Accuracy","F1","IoU","Recall","Precision"])
        csv_path = os.path.join(args.output, "metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Metrics CSV      : {csv_path}")

    print(f"{'─'*50}\n")


if __name__ == "__main__":
    main()
