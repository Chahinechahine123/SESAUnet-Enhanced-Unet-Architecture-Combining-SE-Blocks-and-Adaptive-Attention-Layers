import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16

def build_fcn_8s(input_shape=(512, 512, 3), num_classes=1):
    # Base VGG16 model without fully connected layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Extract relevant feature maps for skip connections
    f3 = base_model.get_layer("block3_pool").output  # 1/8 resolution
    f4 = base_model.get_layer("block4_pool").output  # 1/16 resolution
    f5 = base_model.get_layer("block5_pool").output  # 1/32 resolution

    # Fully convolutional layers
    x = layers.Conv2D(4096, (7, 7), activation='relu', padding='same')(f5)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
    x = layers.Dropout(0.5)(x)

    # Convolution to match the number of classes
    x = layers.Conv2D(num_classes, (1, 1), kernel_initializer='he_normal')(x)

    # Upsampling with skip connections (1/16 resolution)
    x = layers.Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = layers.Add()([x, f4])

    # Upsampling with skip connections (1/8 resolution)
    x = layers.Conv2DTranspose(num_classes, kernel_size=(4, 4), strides=(2, 2), padding='same')(x)
    x = layers.Add()([x, f3])

    # Final upsampling to original image size
    x = layers.Conv2DTranspose(num_classes, kernel_size=(16, 16), strides=(8, 8), padding='same', activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=x)
    return model

# Instantiate and summarize the model
fcn_8s_model = build_fcn_8s()
fcn_8s_model.summary()
