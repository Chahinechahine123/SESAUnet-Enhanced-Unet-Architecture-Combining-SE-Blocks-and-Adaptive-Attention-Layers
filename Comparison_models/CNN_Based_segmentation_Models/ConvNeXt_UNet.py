import keras
from keras import layers
from keras import ops


# -------------------------
# Conv Block
# -------------------------
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


# -------------------------
# CN-UNet
# -------------------------
def ConvNeXt_UNet(image_size=512, num_classes=1):

    inputs = keras.Input((image_size, image_size, 3))

    backbone = keras.applications.ConvNeXtTiny(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )

    # Encoder features
    s1 = backbone.get_layer("convnext_tiny_stem_activation").output
    s2 = backbone.get_layer("convnext_tiny_stage_1_block_2_output").output
    s3 = backbone.get_layer("convnext_tiny_stage_2_block_2_output").output
    s4 = backbone.get_layer("convnext_tiny_stage_3_block_8_output").output

    bridge = backbone.get_layer("convnext_tiny_stage_3_block_8_output").output

    # Decoder
    d1 = layers.UpSampling2D()(bridge)
    d1 = layers.Concatenate()([d1, s3])
    d1 = conv_block(d1, 512)

    d2 = layers.UpSampling2D()(d1)
    d2 = layers.Concatenate()([d2, s2])
    d2 = conv_block(d2, 256)

    d3 = layers.UpSampling2D()(d2)
    d3 = layers.Concatenate()([d3, s1])
    d3 = conv_block(d3, 128)

    d4 = layers.UpSampling2D()(d3)
    d4 = conv_block(d4, 64)

    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid")(d4)

    return keras.Model(inputs, outputs)


model_cn = ConvNeXt_UNet()
model_cn.summary()