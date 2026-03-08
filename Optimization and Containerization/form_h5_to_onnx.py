# convert_to_onnx.py

import tensorflow as tf
import tf2onnx
import onnx
import os
from tensorflow.keras.layers import Conv2DTranspose

# ✅ Patch : surcharger Conv2DTranspose pour ignorer l'argument 'groups'
class PatchedConv2DTranspose(Conv2DTranspose):
    def __init__(self, *args, **kwargs):
        kwargs.pop('groups', None)  # Supprimer 'groups' si présent
        super().__init__(*args, **kwargs)

# Charger le modèle avec le patch
model_path = "model.h5"
model = tf.keras.models.load_model(
    model_path,
    custom_objects={"Conv2DTranspose": PatchedConv2DTranspose},
    compile=False
)

print("✅ Modèle U-Net chargé avec succès")
print(f"📥 Input shape  : {model.input_shape}")
print(f"📤 Output shape : {model.output_shape}")

# Convertir en ONNX
output_path = "model.onnx"

input_signature = [tf.TensorSpec(model.input_shape, tf.float32, name="input")]

onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13,
    output_path=output_path
)

print(f"✅ Modèle ONNX sauvegardé : {output_path}")

# Validation
onnx_model_loaded = onnx.load(output_path)
onnx.checker.check_model(onnx_model_loaded)
print("✅ Modèle ONNX validé avec succès")
print(f"📦 Taille : {os.path.getsize(output_path) / (1024*1024):.2f} MB")