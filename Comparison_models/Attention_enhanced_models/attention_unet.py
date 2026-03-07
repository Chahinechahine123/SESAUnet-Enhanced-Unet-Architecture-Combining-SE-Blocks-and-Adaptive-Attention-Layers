# Attention U-Net
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Multiply, Add, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def attention_gate(x, g, inter_channels):
    # x : skip connection
    # g : gating signal
    theta_x = Conv2D(inter_channels, 1, strides=1, padding='same')(x)
    phi_g = Conv2D(inter_channels, 1, strides=1, padding='same')(g)
    add = Add()([theta_x, phi_g])
    act = Activation('relu')(add)
    psi = Conv2D(1, 1, strides=1, padding='same')(act)
    psi = Activation('sigmoid')(psi)
    return Multiply()([x, psi])

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding='same')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2,2))(x)
    return x, p

def decoder_block(input, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(input)
    attn = attention_gate(skip, x, num_filters // 2)
    x = Concatenate()([x, attn])
    x = conv_block(x, num_filters)
    return x

def build_attention_unet(input_shape=(512,512,3)):
    inputs = Input(input_shape)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b = conv_block(p4, 1024)

    d1 = decoder_block(b, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, activation='sigmoid')(d4)
    model = Model(inputs, outputs, name="Attention_UNet")
    return model

# Test
model = build_attention_unet()
model.summary()