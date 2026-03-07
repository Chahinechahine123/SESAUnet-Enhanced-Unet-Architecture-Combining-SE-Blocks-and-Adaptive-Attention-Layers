from keras import layers
import tensorflow as tf

def swin_patch_embedding(x, patch_size=4, embed_dim=96):
    # Patch embedding
    x = layers.Conv2D(embed_dim, patch_size, strides=patch_size, padding='same')(x)
    x = layers.LayerNormalization()(x)
    return x

def swin_transformer_block(x, num_heads=4, mlp_ratio=4, drop=0.0):
    input_x = x
    # LayerNorm
    x = layers.LayerNormalization()(x)
    # Multi-head self-attention
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
    x = layers.Add()([x, input_x])
    # MLP
    y = layers.LayerNormalization()(x)
    y = layers.Dense(x.shape[-1]*mlp_ratio, activation='gelu')(y)
    y = layers.Dense(x.shape[-1])(y)
    x = layers.Add()([x, y])
    return x

def Swin_UNet(image_size=512, num_classes=1):
    inputs = keras.Input((image_size, image_size, 3))

    # Patch embedding
    x = swin_patch_embedding(inputs, patch_size=4, embed_dim=96)

    # Transformer stages (simplifié)
    x = swin_transformer_block(x)
    x = swin_transformer_block(x)

    # Simple decoder (upsample to input size)
    x = layers.UpSampling2D(size=(4,4), interpolation='bilinear')(x)
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(x)

    return keras.Model(inputs, outputs)

model = Swin_UNet()
model.summary()