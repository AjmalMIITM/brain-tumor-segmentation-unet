import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
# import matplotlib.pyplot as plt # Uncomment if you want to plot history at the end

# --- Dice Coefficient Metric Function ---
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# --- Dice Loss Function ---
def dice_loss(y_true, y_pred, smooth=1e-6):
    return 1 - dice_coef(y_true, y_pred, smooth)

# --- Define the U-Net Model Architecture ---
def unet_model(input_size=(256, 256, 1)):
    inputs = keras.Input(shape=input_size)

    # Encoder Path
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Decoder Path
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model

# --- Main part of the script ---
if __name__ == '__main__':
    
    processed_data_path = r'C:\AI_Projects\BrainTumorProject\brain-tumor-segmentation-unet\data_processed'
    model_save_path = r'C:\AI_Projects\BrainTumorProject\brain-tumor-segmentation-unet\trained_models'
    os.makedirs(model_save_path, exist_ok=True)

    print("Loading preprocessed data...")
    try:
        X_train = np.load(os.path.join(processed_data_path, 'X_train.npy'))
        y_train = np.load(os.path.join(processed_data_path, 'y_train.npy'))
        X_val = np.load(os.path.join(processed_data_path, 'X_val.npy'))
        y_val = np.load(os.path.join(processed_data_path, 'y_val.npy'))
    except FileNotFoundError:
        print("ERROR: Preprocessed data files not found. Please run preprocess_data.py first.")
        exit()

    print(f"Training data shapes: X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"Validation data shapes: X_val: {X_val.shape}, y_val: {y_val.shape}")

    IMG_HEIGHT = X_train.shape[1]
    IMG_WIDTH = X_train.shape[2]
    IMG_CHANNELS = X_train.shape[3]
    input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    print(f"Input shape for U-Net: {input_shape}")

    checkpoint_filepath_to_load = os.path.join(model_save_path, 'unet_brain_tumor_best.keras')
    initial_epoch = 0

    if os.path.exists(checkpoint_filepath_to_load):
        print(f"\n--- Found existing best model at: {checkpoint_filepath_to_load} ---")
        print("--- Loading this model. It will be RE-COMPILED with Dice Loss for further training. ---")
        try:
            model = keras.models.load_model(checkpoint_filepath_to_load, custom_objects={'dice_coef': dice_coef, 'dice_loss': dice_loss})
            print("--- Model loaded successfully. Re-compiling with Dice Loss... ---")
            optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
            model.compile(optimizer=optimizer, loss=dice_loss, metrics=['accuracy', dice_coef])
            print("--- Model re-compiled with Dice Loss. ---")
        except Exception as e:
            print(f"Error loading or re-compiling model: {e}")
            print("Starting training from scratch with a new model using Dice Loss.")
            model = unet_model(input_size=input_shape)
            optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
            model.compile(optimizer=optimizer, loss=dice_loss, metrics=['accuracy', dice_coef])
    else:
        print(f"\n--- No existing best model found at: {checkpoint_filepath_to_load} ---")
        print("--- Creating a new model and starting training from scratch with Dice Loss. ---")
        model = unet_model(input_size=input_shape)
        optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=dice_loss, metrics=['accuracy', dice_coef])
    
    model.summary()

    print("\nStarting model training...")
    EPOCHS = 25 
    BATCH_SIZE = 16

    checkpoint_filepath = os.path.join(model_save_path, 'unet_brain_tumor_best.keras')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_dice_coef', 
        mode='max',             
        save_best_only=True)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_dice_coef',
        mode='max',
        patience=7,         
        verbose=1,
        restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        validation_data=(X_val, y_val),
        callbacks=[model_checkpoint_callback, early_stopping_callback]
    )

    print("\nTraining finished.")
    if os.path.exists(checkpoint_filepath):
         print(f"Best model (based on val_dice_coef) should be at: {checkpoint_filepath}")
    else:
        print(f"Warning: Best model checkpoint file not found at {checkpoint_filepath}.")

    print("\nScript finished.")
