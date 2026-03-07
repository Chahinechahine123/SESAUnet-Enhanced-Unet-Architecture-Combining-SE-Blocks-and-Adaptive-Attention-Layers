from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input,
    GlobalAveragePooling2D, Reshape, Multiply, Lambda)
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

def squeeze_and_excitation(input, ratio=16, name_prefix="se"):
    filters = input.shape[-1]
    se = GlobalAveragePooling2D(name=f"{name_prefix}_gap")(input)
    se = Reshape((1, 1, filters), name=f"{name_prefix}_reshape")(se)
    se = Conv2D(filters // ratio, (1, 1), activation='relu', padding='same', name=f"{name_prefix}_excite1")(se)
    se = Conv2D(filters, (1, 1), activation='sigmoid', padding='same', name=f"{name_prefix}_excite2")(se)
    return Multiply(name=f"{name_prefix}_multiply")([input, se])

def conv_block(input, num_filters, block_name):
    x = Conv2D(num_filters, 3, padding="same", name=f"{block_name}_conv1")(input)
    x = BatchNormalization(name=f"{block_name}_bn1")(x)
    x = Activation("relu", name=f"{block_name}_relu1")(x)
    x = Conv2D(num_filters, 3, padding="same", name=f"{block_name}_conv2")(x)
    x = BatchNormalization(name=f"{block_name}_bn2")(x)
    x = Activation("relu", name=f"{block_name}_relu2")(x)
    x = squeeze_and_excitation(x, name_prefix=f"{block_name}_se")
    return x

def encoder_block(input, num_filters, block_name):
    x = conv_block(input, num_filters, block_name)
    p = MaxPool2D((2, 2), name=f"{block_name}_pool")(x)
    return x, p

def decoder_block(input, skip_features, num_filters, block_name):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same", name=f"{block_name}_transconv")(input)
    x = Concatenate(name=f"{block_name}_concat")([x, skip_features])
    x = conv_block(x, num_filters, block_name)
    return x

def adaptive_attention(input_tensor, name_prefix="attention"):
    avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True), name=f"{name_prefix}_avg_pool")(input_tensor)
    max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True), name=f"{name_prefix}_max_pool")(input_tensor)
    concat = Concatenate(axis=-1, name=f"{name_prefix}_concat")([avg_pool, max_pool])
    # Dans la fonction adaptive_attention
    attention = Conv2D(filters=1, kernel_size=(7, 7), padding='same', activation='sigmoid', name=f"{name_prefix}_conv")(concat)
    return Multiply(name=f"{name_prefix}_multiply")([input_tensor, attention])

def build_unet_sa_7x7(input_shape):
    inputs = Input(input_shape)
    s1, p1 = encoder_block(inputs, 64, "encoder1")
    s2, p2 = encoder_block(p1, 128, "encoder2")
    s3, p3 = encoder_block(p2, 256, "encoder3")
    s4, p4 = encoder_block(p3, 512, "encoder4")
    b1 = conv_block(p4, 1024, "bottleneck")
    d1 = decoder_block(b1, s4, 512, "decoder1")
    d2 = decoder_block(d1, s3, 256, "decoder2")
    d3 = decoder_block(d2, s2, 128, "decoder3")
    d4 = decoder_block(d3, s1, 64, "decoder4")
    d4 = adaptive_attention(d4, "output_attention")
    outputs = Conv2D(1, 1, activation="sigmoid", name="output_layer")(d4)
    return Model(inputs, outputs, name="UNet_SA_7x7")

# Test
if __name__ == "__main__":
    model = build_unet_sa_7x7((512, 512, 3))
    model.summary()