from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPool2D,
    Conv2DTranspose, Concatenate, Input,
    GlobalAveragePooling2D, Reshape, Multiply, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def spatial_attention(input_tensor, name_prefix="sa"):
    
    # Channel-wise average pooling
    avg_pool = Lambda(
        lambda x: K.mean(x, axis=-1, keepdims=True),
        name=f"{name_prefix}_avgpool"
    )(input_tensor)

    # Channel-wise max pooling
    max_pool = Lambda(
        lambda x: K.max(x, axis=-1, keepdims=True),
        name=f"{name_prefix}_maxpool"
    )(input_tensor)

    # Concatenate along channel axis
    concat = Concatenate(axis=-1, name=f"{name_prefix}_concat")([avg_pool, max_pool])

    # 7x7 convolution to generate attention map
    attention = Conv2D(
        filters=1,
        kernel_size=(7, 7),
        padding="same",
        activation="sigmoid",
        name=f"{name_prefix}_conv"
    )(concat)

    # Apply attention
    output = Multiply(name=f"{name_prefix}_multiply")([input_tensor, attention])

    return output

def conv_block(input, num_filters, block_name):
    x = Conv2D(num_filters, 3, padding="same", name=f"{block_name}_conv1")(input)
    x = BatchNormalization(name=f"{block_name}_bn1")(x)
    x = Activation("relu", name=f"{block_name}_relu1")(x)

    x = Conv2D(num_filters, 3, padding="same", name=f"{block_name}_conv2")(x)
    x = BatchNormalization(name=f"{block_name}_bn2")(x)
    x = Activation("relu", name=f"{block_name}_relu2")(x)

    return x


def encoder_block(input, num_filters, block_name):
    x = conv_block(input, num_filters, block_name)
    p = MaxPool2D((2, 2), name=f"{block_name}_pool")(x)
    return x, p


def decoder_block(input, skip_features, num_filters, block_name):

    x = Conv2DTranspose(
        num_filters,
        (2, 2),
        strides=2,
        padding="same",
        name=f"{block_name}_transconv"
    )(input)

    x = Concatenate(name=f"{block_name}_concat")([x, skip_features])

    x = conv_block(x, num_filters, block_name)

    # 🔥 Apply SA after each decoder stage
    x = spatial_attention(x, name_prefix=f"{block_name}_sa")

    return x


def build_unet_sa_decoder_out(input_shape=(512, 512, 3)):

    inputs = Input(input_shape, name="input_layer")

    # Encoder
    s1, p1 = encoder_block(inputs, 64, "encoder1")
    s2, p2 = encoder_block(p1, 128, "encoder2")
    s3, p3 = encoder_block(p2, 256, "encoder3")
    s4, p4 = encoder_block(p3, 512, "encoder4")

    # Bottleneck
    b1 = conv_block(p4, 1024, "bottleneck")

    # Decoder
    d1 = decoder_block(b1, s4, 512, "decoder1")
    d2 = decoder_block(d1, s3, 256, "decoder2")
    d3 = decoder_block(d2, s2, 128, "decoder3")
    d4 = decoder_block(d3, s1, 64, "decoder4") 
    
    d4 = spatial_attention(d4, name_prefix="output_sa")
    outputs = Conv2D(1, 1, activation="sigmoid", padding="same")(d4)

    model = Model(inputs, outputs, name="UNet_SA_decoderout")

    return model

if __name__ == "__main__":
    model = build_unet_sa_decoder_out()
    model.summary() 