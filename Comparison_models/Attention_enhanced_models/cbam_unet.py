# CBAM-UNet
from tensorflow.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Reshape, Activation, Multiply, Add

def cbam_block(input_feature, ratio=8):
    # Channel attention
    channel = input_feature.shape[-1]
    avg_pool = GlobalAveragePooling2D()(input_feature)
    max_pool = GlobalMaxPooling2D()(input_feature)
    dense = Dense(channel//ratio, activation='relu')
    dense_out = Dense(channel, activation='sigmoid')

    avg_out = dense_out(dense(avg_pool))
    max_out = dense_out(dense(max_pool))
    channel_att = Add()([avg_out, max_out])
    channel_att = Reshape((1,1,channel))(channel_att)
    x = Multiply()([input_feature, channel_att])

    # Spatial attention
    avg_pool_sp = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(x)
    max_pool_sp = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(x)
    concat = Concatenate(axis=-1)([avg_pool_sp, max_pool_sp])
    spatial_att = Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')(concat)
    x = Multiply()([x, spatial_att])
    return x

def conv_block_cbam(input, num_filters):
    x = Conv2D(num_filters, 3, padding='same', activation='relu')(input)
    x = Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    x = cbam_block(x)
    return x

def encoder_block_cbam(input, num_filters):
    x = conv_block_cbam(input, num_filters)
    p = MaxPool2D((2,2))(x)
    return x, p

def decoder_block_cbam(input, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2,2), strides=2, padding='same')(input)
    x = Concatenate()([x, skip])
    x = conv_block_cbam(x, num_filters)
    return x

def build_cbam_unet(input_shape=(512,512,3)):
    inputs = Input(input_shape)
    s1, p1 = encoder_block_cbam(inputs, 64)
    s2, p2 = encoder_block_cbam(p1, 128)
    s3, p3 = encoder_block_cbam(p2, 256)
    s4, p4 = encoder_block_cbam(p3, 512)

    b = conv_block_cbam(p4, 1024)

    d1 = decoder_block_cbam(b, s4, 512)
    d2 = decoder_block_cbam(d1, s3, 256)
    d3 = decoder_block_cbam(d2, s2, 128)
    d4 = decoder_block_cbam(d3, s1, 64)

    outputs = Conv2D(1,1,activation='sigmoid')(d4)
    model = Model(inputs, outputs, name="CBAM_UNet")
    return model

# Test
model = build_cbam_unet()
model.summary()