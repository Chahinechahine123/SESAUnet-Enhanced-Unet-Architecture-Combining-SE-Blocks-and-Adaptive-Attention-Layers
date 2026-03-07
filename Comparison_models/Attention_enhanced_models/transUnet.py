# ==============================
# Original-like TransUNet
# ==============================

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# ---------------------------------
# CNN Encoder (ResNet-like)
# ---------------------------------
def conv_block(x, filters):
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def encoder_block(x, filters):
    s = conv_block(x, filters)
    p = MaxPool2D((2,2))(s)
    return s, p


# ---------------------------------
# Patch Embedding Layer
# ---------------------------------
class PatchEmbedding(Layer):
    def __init__(self, patch_size=1, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.projection = Dense(embed_dim)

    def call(self, x):
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        C = tf.shape(x)[3]

        x = tf.reshape(x, [B, H*W, C])
        x = self.projection(x)
        return x


# ---------------------------------
# Transformer Encoder Block
# ---------------------------------
def transformer_block(x, embed_dim, num_heads, ff_dim):

    attn = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim
    )(x, x)

    x = Add()([x, attn])
    x = LayerNormalization()(x)

    ffn = Dense(ff_dim, activation='relu')(x)
    ffn = Dense(embed_dim)(ffn)

    x = Add()([x, ffn])
    x = LayerNormalization()(x)

    return x


# ---------------------------------
# Decoder Block
# ---------------------------------
def decoder_block(x, skip, filters):
    x = Conv2DTranspose(filters, 2, strides=2, padding='same')(x)
    x = Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


# ---------------------------------
# Build Full TransUNet
# ---------------------------------
def build_transunet(input_shape=(512,512,3),
                    embed_dim=768,
                    num_heads=12,
                    ff_dim=3072,
                    num_transformer_layers=8):

    inputs = Input(input_shape)

    # CNN Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bottleneck CNN
    b = conv_block(p4, 1024)

    # Patch Embedding (flatten spatial)
    x = PatchEmbedding(embed_dim=embed_dim)(b)

    # Transformer Encoder Stack
    for _ in range(num_transformer_layers):
        x = transformer_block(x, embed_dim, num_heads, ff_dim)

    # Reshape tokens back to feature map
    H = input_shape[0] // 16
    W = input_shape[1] // 16
    x = Reshape((H, W, embed_dim))(x)

    # Decoder with CNN skip connections
    d1 = decoder_block(x, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, activation='sigmoid')(d4)

    model = Model(inputs, outputs, name="TransUNet_OriginalLike")

    return model


# Test
model = build_transunet()
model.summary()