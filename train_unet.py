import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
# Import Albumentations
import albumentations as A
import cv2 # Ensure cv2 is imported at the top

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

# --- Combined BCE + Dice Loss Function ---
def combined_bce_dice_loss(y_true, y_pred, bce_weight=0.5, dice_weight=0.5, smooth_dice=1e-6):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred, smooth_dice)
    return bce_weight * bce + dice_weight * d_loss

# --- Albumentations Augmentation Pipeline ---
transform_alb = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.GridDistortion(p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    # A.GaussNoise(var_limit=(10.0, 50.0), p=0.5) # Optional
])

# --- Corrected Albumentations Wrapper for tf.data ---
def augment_data_with_albumentations(image_tensor, mask_tensor):
    # Convert TensorFlow Tensors to NumPy arrays
    image_np = image_tensor.numpy()
    mask_np = mask_tensor.numpy()

    # Scale to uint8 [0, 255] for Albumentations
    image_np_uint8 = (image_np * 255.0).astype(np.uint8)
    mask_np_uint8 = (mask_np * 255.0).astype(np.uint8) # Assuming masks are 0 or 1, scaling makes them 0 or 255

    # Apply Albumentations transformations
    augmented = transform_alb(image=image_np_uint8, mask=mask_np_uint8)
    aug_img_uint8 = augmented['image']
    aug_mask_uint8 = augmented['mask']

    # Convert back to float32 [0, 1]
    aug_img_float32 = aug_img_uint8.astype(np.float32) / 255.0
    aug_mask_float32 = aug_mask_uint8.astype(np.float32) / 255.0

    # Ensure mask is binary {0, 1} after potential interpolations
    aug_mask_float32 = np.round(aug_mask_float32)
    
    # Ensure channel dimension if Albumentations removed it for single-channel
    if aug_img_float32.ndim == 2:
        aug_img_float32 = np.expand_dims(aug_img_float32, axis=-1)
    if aug_mask_float32.ndim == 2:
        aug_mask_float32 = np.expand_dims(aug_mask_float32, axis=-1)
            
    return aug_img_float32, aug_mask_float32

def tf_augment_data_with_albumentations(image, mask):
    aug_img, aug_mask = tf.py_function(
        func=augment_data_with_albumentations,
        inp=[image, mask],
        Tout=[tf.float32, tf.float32]
    )
    
    img_shape = image.shape # Original shape from the dataset
    aug_img.set_shape(img_shape)
    aug_mask.set_shape(img_shape)
    
    return aug_img, aug_mask

# --- Define the U-Net Model Architecture (remains the same) ---
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

    BATCH_SIZE = 16
    BUFFER_SIZE = len(X_train)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
    train_dataset = train_dataset.map(tf_augment_data_with_albumentations, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    checkpoint_filepath_to_load = os.path.join(model_save_path, 'unet_brain_tumor_best.keras')
    initial_epoch = 0
    optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)

    if os.path.exists(checkpoint_filepath_to_load):
        print(f"\n--- Found existing best model at: {checkpoint_filepath_to_load} ---")
        print("--- Loading this model. It will be RE-COMPILED with Combined BCE-Dice Loss for further training. ---")
        try:
            model = keras.models.load_model(
                checkpoint_filepath_to_load, 
                custom_objects={
                    'dice_coef': dice_coef, 
                    'dice_loss': dice_loss, 
                    'combined_bce_dice_loss': combined_bce_dice_loss
                }
            )
            print("--- Model loaded successfully. Re-compiling with Combined BCE-Dice Loss... ---")
            model.compile(optimizer=optimizer, loss=combined_bce_dice_loss, metrics=['accuracy', dice_coef])
            print("--- Model re-compiled with Combined BCE-Dice Loss. ---")
        except Exception as e:
            print(f"Error loading or re-compiling model: {e}")
            print("Starting training from scratch with a new model using Combined BCE-Dice Loss.")
            model = unet_model(input_size=input_shape)
            model.compile(optimizer=optimizer, loss=combined_bce_dice_loss, metrics=['accuracy', dice_coef])
    else:
        print(f"\n--- No existing best model found at: {checkpoint_filepath_to_load} ---")
        print("--- Creating a new model and starting training from scratch with Combined BCE-Dice Loss. ---")
        model = unet_model(input_size=input_shape)
        model.compile(optimizer=optimizer, loss=combined_bce_dice_loss, metrics=['accuracy', dice_coef])
    
    model.summary()

    print("\nStarting model training...")
    EPOCHS = 50
    checkpoint_filepath = os.path.join(model_save_path, 'unet_brain_tumor_best.keras')
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_dice_coef', 
        mode='max',             
        save_best_only=True)

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_dice_coef',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_dice_coef',
        mode='max',
        patience=15,      
        verbose=1,
        restore_best_weights=True)

    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        initial_epoch=initial_epoch,
        validation_data=val_dataset,
        callbacks=[model_checkpoint_callback, early_stopping_callback, reduce_lr_callback]
    )

    print("\nTraining finished.")
    if os.path.exists(checkpoint_filepath):
         print(f"Best model (based on val_dice_coef) should be at: {checkpoint_filepath}")
    else:
        print(f"Warning: Best model checkpoint file not found at {checkpoint_filepath}.")

    print("\nScript finished.")

