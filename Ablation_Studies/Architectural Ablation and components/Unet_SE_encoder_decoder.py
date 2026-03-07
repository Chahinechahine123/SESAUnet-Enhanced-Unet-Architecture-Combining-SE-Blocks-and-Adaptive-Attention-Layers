from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# ---------- SE ----------
def squeeze_and_excitation(input, ratio=16, name_prefix="se"):
    filters = input.shape[-1]
    se = GlobalAveragePooling2D(name=f"{name_prefix}_gap")(input)
    se = Reshape((1, 1, filters))(se)
    se = Conv2D(filters // ratio, 1, activation='relu', padding='same')(se)
    se = Conv2D(filters, 1, activation='sigmoid', padding='same')(se)
    return Multiply()([input, se])

# ---------- Blocks ----------
def conv_block(input, num_filters, block_name, use_se=False):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if use_se:
        x = squeeze_and_excitation(x, name_prefix=f"{block_name}_se")
    return x

def encoder_block(input, num_filters, block_name):
    x = conv_block(input, num_filters, block_name, use_se=True)
    p = MaxPool2D((2,2))(x)
    return x, p

def decoder_block(input, skip, num_filters, block_name):
    x = Conv2DTranspose(num_filters, 2, strides=2, padding="same")(input)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters, block_name, use_se=True)
    return x

def build_model():
    inputs = Input((512,512,3))
    s1,p1 = encoder_block(inputs,64,"enc1")
    s2,p2 = encoder_block(p1,128,"enc2")
    s3,p3 = encoder_block(p2,256,"enc3")
    s4,p4 = encoder_block(p3,512,"enc4")
    b1 = conv_block(p4,1024,"bottleneck",use_se=False)
    d1 = decoder_block(b1,s4,512,"dec1")
    d2 = decoder_block(d1,s3,256,"dec2")
    d3 = decoder_block(d2,s2,128,"dec3")
    d4 = decoder_block(d3,s1,64,"dec4")
    outputs = Conv2D(1,1,activation="sigmoid")(d4)
    return Model(inputs,outputs,name="UNet_SE_Encoder_Decoder")

if __name__ == "__main__":
    model = build_model()
    model.summary()