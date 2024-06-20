import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Concatenate, Flatten, LayerNormalization
from tensorflow.keras.regularizers import l2

# Build hybrid model combining U-Net and ViT
def build_hybrid_model(unet_model, vit_model, input_shape, num_classes):
    # U-Net path
    unet_input = tf.keras.Input(shape=input_shape)
    unet_output = unet_model(unet_input)
    unet_output = GlobalAveragePooling2D()(unet_output)

    # ViT path
    vit_input = tf.keras.Input(shape=input_shape)
    resized_input = tf.image.resize(vit_input, (224, 224))
    vit_output = vit_model(resized_input)
    vit_output = Flatten()(vit_output)

    # Combine intermediate layers
    combined = Concatenate()([unet_output, vit_output])
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(combined)
    x = Dropout(0.6)(x)  # Increased dropout rate
    x = LayerNormalization()(x)

    # Final output
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[unet_input, vit_input], outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model
