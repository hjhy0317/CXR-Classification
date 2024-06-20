# models/unet.py
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, BatchNormalization
from tensorflow.keras.applications import EfficientNetB3

def unet_model_with_efficientnet(input_shape):
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)

    inputs = base_model.input
    enc1 = base_model.get_layer('block1a_activation').output
    enc2 = base_model.get_layer('block2b_expand_activation').output
    enc3 = base_model.get_layer('block3b_expand_activation').output
    enc4 = base_model.get_layer('block4a_expand_activation').output
    enc5 = base_model.get_layer('block6a_expand_activation').output

    up6 = UpSampling2D((2, 2))(enc5)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = BatchNormalization()(conv6)
    merge6 = Concatenate()([tf.image.resize(enc4, tf.shape(conv6)[1:3]), conv6])

    up7 = UpSampling2D((2, 2))(merge6)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = BatchNormalization()(conv7)
    merge7 = Concatenate()([tf.image.resize(enc3, tf.shape(conv7)[1:3]), conv7])

    up8 = UpSampling2D((2, 2))(merge7)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = BatchNormalization()(conv8)
    merge8 = Concatenate()([tf.image.resize(enc2, tf.shape(conv8)[1:3]), conv8])

    up9 = UpSampling2D((2, 2))(merge8)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = BatchNormalization()(conv9)
    merge9 = Concatenate()([tf.image.resize(enc1, tf.shape(conv9)[1:3]), conv9])

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(merge9)

    return Model(inputs, outputs)
