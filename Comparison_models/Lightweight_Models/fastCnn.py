def fast_sc_conv(x, filters, kernel_size=3, stride=1):
    x = layers.DepthwiseConv2D(kernel_size, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def Fast_SCNN(image_size=512, num_classes=1):
    inputs = keras.Input((image_size, image_size, 3))

    # Learning to Downsample
    lds = fast_sc_conv(inputs, 32, 3, stride=2)  # 512->256
    lds = fast_sc_conv(lds, 48, 3, stride=2)    # 256->128
    lds = fast_sc_conv(lds, 64, 3, stride=2)    # 128->64

    # Global Feature Extractor
    gfe = fast_sc_conv(lds, 128)
    gfe = fast_sc_conv(gfe, 128)

    # Upsample global features to lds size (64x64)
    gfe_up = layers.UpSampling2D(
        size=(lds.shape[1] // gfe.shape[1], lds.shape[2] // gfe.shape[2]),
        interpolation='bilinear'
    )(gfe)

    # Feature Fusion
    ff = layers.Concatenate()([gfe_up, lds])  # maintenant les dimensions correspondent
    ff = fast_sc_conv(ff, 128)

    # Upsample à la taille d'entrée
    x = layers.UpSampling2D(
        size=(image_size // ff.shape[1], image_size // ff.shape[2]),
        interpolation='bilinear'
    )(ff)

    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid')(x)

    return keras.Model(inputs, outputs)

model_fast_scnn = Fast_SCNN()
model_fast_scnn.summary()