import os
import numpy as np
import tensorflow as tf # Make sure this is imported
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2 # We might not need cv2 if just loading .npy, but good to have for other image ops

# --- Configuration ---
processed_data_path = r'C:\AI_Projects\BrainTumorProject\brain-tumor-segmentation-unet\data_processed'
# Path to your saved best model
model_path = r'C:\AI_Projects\BrainTumorProject\brain-tumor-segmentation-unet\trained_models\unet_brain_tumor_best.keras'

NUM_SAMPLES_TO_DISPLAY = 5 # How many samples to show visually

# --- Keras Dice Coefficient Metric Function (must match the one used in training) ---
# This is the function Keras needs to know about when loading the model
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# --- NumPy Dice Coefficient Function (for evaluation AFTER prediction) ---
# This function takes NumPy arrays as input.
def dice_coefficient_numpy(y_true_np, y_pred_np, smooth=1e-6):
    """
    Calculates the Dice coefficient between true and predicted binary masks (NumPy version).
    y_true_np: True binary mask (0 or 1), as a NumPy array.
    y_pred_np: Predicted binary mask (0 or 1, after thresholding), as a NumPy array.
    smooth: A small constant to avoid division by zero.
    """
    y_true_f = y_true_np.flatten()
    y_pred_f = y_pred_np.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

# --- Main part of the script ---
if __name__ == '__main__':
    print("Loading the trained U-Net model...")
    try:
        # MODIFIED: Pass custom_objects to load_model
        model = keras.models.load_model(model_path, custom_objects={'dice_coef': dice_coef})
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load the model from {model_path}")
        print(f"Error details: {e}")
        exit()

    print("\nLoading validation data...")
    try:
        X_val = np.load(os.path.join(processed_data_path, 'val_images.npy'))
        y_val = np.load(os.path.join(processed_data_path, 'val_masks.npy'))
    except FileNotFoundError:
        print(f"ERROR: Validation data files not found in '{processed_data_path}'.")
        print("Please ensure preprocess_data.py was run successfully.")
        exit()
    
    print(f"Validation data shapes: X_val: {X_val.shape}, y_val: {y_val.shape}")

    if len(X_val) == 0:
        print("Validation data is empty. Cannot proceed.")
        exit()

    # --- Make Predictions on Validation Samples ---
    print(f"\nMaking predictions on {NUM_SAMPLES_TO_DISPLAY} validation samples...")
    indices_to_display = np.arange(min(NUM_SAMPLES_TO_DISPLAY, len(X_val)))
    
    sample_images = X_val[indices_to_display]
    sample_true_masks = y_val[indices_to_display]
    
    predicted_masks_probs = model.predict(sample_images)
    predicted_masks_binary = (predicted_masks_probs > 0.5).astype(np.uint8)

    # --- Display Results and Calculate Dice Scores for these samples ---
    plt.figure(figsize=(15, NUM_SAMPLES_TO_DISPLAY * 4)) 
    
    individual_dice_scores = []

    for i in range(len(indices_to_display)):
        original_image = sample_images[i, :, :, 0] 
        true_mask = sample_true_masks[i, :, :, 0]
        pred_mask = predicted_masks_binary[i, :, :, 0]
        
        # Use the NumPy version of Dice for evaluation here
        dice_score = dice_coefficient_numpy(true_mask, pred_mask) 
        individual_dice_scores.append(dice_score)
        
        plt.subplot(NUM_SAMPLES_TO_DISPLAY, 3, i * 3 + 1)
        plt.imshow(original_image, cmap='gray')
        plt.title(f"Sample {indices_to_display[i]}: Original Image")
        plt.axis('off')
        
        plt.subplot(NUM_SAMPLES_TO_DISPLAY, 3, i * 3 + 2)
        plt.imshow(true_mask, cmap='gray')
        plt.title(f"True Mask")
        plt.axis('off')
        
        plt.subplot(NUM_SAMPLES_TO_DISPLAY, 3, i * 3 + 3)
        plt.imshow(pred_mask, cmap='gray')
        plt.title(f"Predicted Mask\nDice: {dice_score:.4f}")
        plt.axis('off')

    plt.tight_layout()
    # Save the figure before showing it
    figure_save_path = os.path.join(os.path.dirname(model_path), 'evaluation_plot.png') # Save in trained_models folder
    try:
        plt.savefig(figure_save_path)
        print(f"\nEvaluation plot saved to: {figure_save_path}")
    except Exception as e:
        print(f"Error saving evaluation plot: {e}")
    plt.show()


    print("\nDice scores for displayed samples:")
    for i in range(len(indices_to_display)):
        print(f"  Sample {indices_to_display[i]} Dice: {individual_dice_scores[i]:.4f}")
    
    # --- Calculate Average Dice Score over a larger subset or all validation data ---
    num_samples_for_avg_dice = min(100, len(X_val))
    print(f"\nCalculating average Dice score over the first {num_samples_for_avg_dice} validation samples...")
    
    # Ensure we use enough memory for prediction, but avoid OOM on very large X_val
    # For 100 samples, direct prediction is usually fine.
    # If X_val were huge, you might predict in batches.
    if num_samples_for_avg_dice > 0:
        all_val_pred_probs = model.predict(X_val[:num_samples_for_avg_dice])
        all_val_pred_binary = (all_val_pred_probs > 0.5).astype(np.uint8)
        
        avg_dice_scores = []
        for i in range(num_samples_for_avg_dice):
            # Use the NumPy version of Dice for evaluation
            score = dice_coefficient_numpy(y_val[i, :, :, 0], all_val_pred_binary[i, :, :, 0])
            avg_dice_scores.append(score)
            
        if avg_dice_scores:
            print(f"Average Dice Coefficient over {num_samples_for_avg_dice} samples: {np.mean(avg_dice_scores):.4f}")
        else:
            print("Could not calculate average Dice score (no samples processed for average).")
    else:
        print("No samples available to calculate average Dice score.")


    print("\nEvaluation script finished.")