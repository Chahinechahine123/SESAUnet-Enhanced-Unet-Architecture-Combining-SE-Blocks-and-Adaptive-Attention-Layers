# optimize_onnx.py

import onnx
from onnxsim import simplify
from onnxconverter_common import float16
import os

input_path  = "model.onnx"
sim_path    = "model_simplified.onnx"
fp16_path   = "model_fp16.onnx"

# ─────────────────────────────────────────
# ÉTAPE 1 : Graph Simplification
# ─────────────────────────────────────────
print("🔧 Étape 1 : Simplification du graphe...")
model = onnx.load(input_path)
model_simplified, check = simplify(model)

if not check:
    print("⚠️  La simplification a échoué, on garde le modèle original")
    model_simplified = model

onnx.save(model_simplified, sim_path)
print(f"✅ Graphe simplifié sauvegardé : {sim_path}")
print(f"   Taille : {os.path.getsize(sim_path) / (1024*1024):.2f} MB")

# ─────────────────────────────────────────
# ÉTAPE 2 : Conversion FP16
# ─────────────────────────────────────────
print("\n🔧 Étape 2 : Conversion FP16...")
model_fp16 = float16.convert_float_to_float16(
    model_simplified,
    keep_io_types=True  # ✅ Garde les inputs/outputs en FP32 pour compatibilité
)

onnx.save(model_fp16, fp16_path)
print(f"✅ Modèle FP16 sauvegardé : {fp16_path}")
print(f"   Taille : {os.path.getsize(fp16_path) / (1024*1024):.2f} MB")

# ─────────────────────────────────────────
# RÉSUMÉ
# ─────────────────────────────────────────
original_size = os.path.getsize(input_path)   / (1024*1024)
final_size    = os.path.getsize(fp16_path)    / (1024*1024)

print(f"""
╔══════════════════════════════════════╗
║           RÉSUMÉ OPTIMISATION        ║
╠══════════════════════════════════════╣
║  Original  (FP32) : {original_size:>7.2f} MB        ║
║  Optimisé  (FP16) : {final_size:>7.2f} MB        ║
║  Réduction        : {((1 - final_size/original_size)*100):>6.1f} %         ║
╚══════════════════════════════════════╝

📌 Prochaine étape sur le Jetson Nano :
   → Convertir model_fp16.onnx en TensorRT (.engine)
      trtexec --onnx=model_fp16.onnx \\
              --saveEngine=model.engine \\
              --fp16
""")