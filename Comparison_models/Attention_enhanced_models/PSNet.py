import keras
from keras import layers
import tensorflow as tf


# -----------------------------
# Pyramid Pooling Module (SAFE)
# -----------------------------
def pyramid_pooling_module(input_tensor, bin_sizes):

    concat_list = [input_tensor]

    h = input_tensor.shape[1]
    w = input_tensor.shape[2]

    for bin_size in bin_sizes:

        # Adaptive pooling
        x = layers.Resizing(bin_size, bin_size)(input_tensor)
        x = layers.Conv2D(512, 1, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        # Resize back safely
        x = layers.Resizing(h, w)(x)

        concat_list.append(x)

    return layers.Concatenate()(concat_list)


# -----------------------------
# PSPNet
# -----------------------------
def PSNet(image_size=512, num_classes=1):

    inputs = keras.Input((image_size, image_size, 3))

    base_model = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs
    )

    # 32x32 feature map for 512 input
    x = base_model.get_layer("conv4_block6_out").output

    x = pyramid_pooling_module(x, [1, 2, 3, 6])

    x = layers.Conv2D(512, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Final upsample
    x = layers.Resizing(image_size, image_size)(x)

    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs)


model = PSNet()
model.summary()