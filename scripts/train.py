import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from models.unet import unet_model_with_efficientnet
from models.vit import vit_model
from utils.model_utils import build_hybrid_model

class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.accuracies = []
        self.val_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracies.append(logs.get('accuracy'))
        self.val_accuracies.append(logs.get('val_accuracy'))
        print(f'Epoch {epoch+1}, Loss: {logs.get("loss")}, Validation Loss: {logs.get("val_loss")}, Accuracy: {logs.get("accuracy")}, Validation Accuracy: {logs.get("val_accuracy")}')

def run(data_dir, input_shape, num_classes):
    train_data = np.load(os.path.join(data_dir, 'train.npz'))
    val_data = np.load(os.path.join(data_dir, 'val.npz'))
    X_train, y_train = train_data['X_train'], train_data['y_train']
    X_val, y_val = val_data['X_val'], val_data['y_val']

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    loss_history = LossHistory()

    unet = unet_model_with_efficientnet(input_shape)
    vit = vit_model(input_shape)
    model = build_hybrid_model(unet, vit, input_shape, num_classes)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        validation_data=(X_val, y_val),
        epochs=50,
        callbacks=[loss_history, reduce_lr, EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )

    # Plot loss and accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history.losses, label='Training Loss')
    plt.plot(loss_history.val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss_history.accuracies, label='Training Accuracy')
    plt.plot(loss_history.val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.show()

    # Save the trained model
    model.save('hybrid_model.h5')
