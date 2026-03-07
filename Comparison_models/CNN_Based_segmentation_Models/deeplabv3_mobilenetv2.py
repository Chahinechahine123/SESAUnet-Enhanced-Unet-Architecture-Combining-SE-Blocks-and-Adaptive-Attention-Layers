import keras
from keras import layers
from keras import ops


# ----------------------------
# Convolution Block
# ----------------------------
def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return ops.nn.relu(x)


# ----------------------------
# ASPP
# ----------------------------
def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape

    x = layers.AveragePooling2D(pool_size=(dims[1], dims[2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)

    out_pool = layers.UpSampling2D(
        size=(dims[1] // x.shape[1], dims[2] // x.shape[2]),
        interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)

    return output


# ----------------------------
# DeepLabV3+ MobileNetV2
# ----------------------------
def DeeplabV3Plus_MobileNet(image_size, num_classes):

    model_input = keras.Input(shape=(image_size, image_size, 3))
    preprocessed = keras.applications.mobilenet_v2.preprocess_input(model_input)

    backbone = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_tensor=preprocessed,
    )

    # High-level feature map
    x = backbone.get_layer("block_13_expand_relu").output  # stride 16

    x = DilatedSpatialPyramidPooling(x)

    # Upsample ASPP output
    x = layers.UpSampling2D(
        size=(4, 4), interpolation="bilinear"
    )(x)

    # Low-level features
    low_level = backbone.get_layer("block_3_expand_relu").output  # stride 4
    low_level = convolution_block(low_level, num_filters=48, kernel_size=1)

    # Concatenate
    x = layers.Concatenate(axis=-1)([x, low_level])
    x = convolution_block(x)
    x = convolution_block(x)

    # Final upsampling
    x = layers.UpSampling2D(
        size=(4, 4), interpolation="bilinear"
    )(x)

    # Output
    model_output = layers.Conv2D(
        num_classes,
        kernel_size=(1, 1),
        padding="same",
        activation="sigmoid"
    )(x)

    return keras.Model(inputs=model_input, outputs=model_output)


# ----------------------------
# Build Model
# ----------------------------
IMAGE_SIZE = 512
NUM_CLASSES = 1

model = DeeplabV3Plus_MobileNet(image_size=IMAGE_SIZE, num_classes=NUM_CLASSES)
model.summary()