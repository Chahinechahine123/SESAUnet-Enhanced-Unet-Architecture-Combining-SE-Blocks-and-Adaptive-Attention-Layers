import keras
from keras import layers

def mobile_conv_block(x, filters, kernel_size=3):
    x = layers.SeparableConv2D(filters, kernel_size, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def Mobile_UNet(image_size=512, num_classes=1):
    inputs = keras.Input((image_size, image_size, 3))

    # Encoder
    c1 = mobile_conv_block(inputs, 32)
    p1 = layers.MaxPool2D()(c1)

    c2 = mobile_conv_block(p1, 64)
    p2 = layers.MaxPool2D()(c2)

    c3 = mobile_conv_block(p2, 128)
    p3 = layers.MaxPool2D()(c3)

    c4 = mobile_conv_block(p3, 256)
    p4 = layers.MaxPool2D()(c4)

    # Bottleneck
    b = mobile_conv_block(p4, 512)

    # Decoder
    d4 = layers.UpSampling2D()(b)
    d4 = layers.Concatenate()([d4, c4])
    d4 = mobile_conv_block(d4, 256)

    d3 = layers.UpSampling2D()(d4)
    d3 = layers.Concatenate()([d3, c3])
    d3 = mobile_conv_block(d3, 128)

    d2 = layers.UpSampling2D()(d3)
    d2 = layers.Concatenate()([d2, c2])
    d2 = mobile_conv_block(d2, 64)

    d1 = layers.UpSampling2D()(d2)
    d1 = layers.Concatenate()([d1, c1])
    d1 = mobile_conv_block(d1, 32)

    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(d1)

    return keras.Model(inputs, outputs)

model = Mobile_UNet()
model.summary()