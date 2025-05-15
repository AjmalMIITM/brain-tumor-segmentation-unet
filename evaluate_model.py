import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix # For sensitivity, specificity, precision

# --- Configuration ---
processed_data_path = r'C:\AI_Projects\BrainTumorProject\brain-tumor-segmentation-unet\data_processed'
model_path = r'C:\AI_Projects\BrainTumorProject\brain-tumor-segmentation-unet\trained_models\unet_brain_tumor_best.keras' # Loads the latest best model

NUM_SAMPLES_TO_DISPLAY = 5
THRESHOLD = 0.5 # For converting probabilities to binary mask

# --- Keras Dice Coefficient Metric Function (for loading the model) ---
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# --- NumPy Metrics Functions (for evaluation AFTER prediction) ---
def dice_coefficient_numpy(y_true_np, y_pred_np, smooth=1e-6):
    y_true_f = y_true_np.flatten()
    y_pred_f = y_pred_np.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return dice

def jaccard_index_numpy(y_true_np, y_pred_np, smooth=1e-6):
    y_true_f = y_true_np.flatten()
    y_pred_f = y_pred_np.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def calculate_pixel_metrics(y_true_np, y_pred_np):
    y_true_f = y_true_np.flatten()
    y_pred_f = y_pred_np.flatten()
    
    if len(y_true_f) == 0 or len(y_pred_f) == 0: # Handle empty masks if necessary
        return 0,0,0,0 # Or some other appropriate default

    tn, fp, fn, tp = confusion_matrix(y_true_f, y_pred_f, labels=[0, 1]).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Less critical for tumor seg
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    return sensitivity, specificity, precision, tp # also return tp for context

# --- Main part of the script ---
if __name__ == '__main__':
    print("Loading the trained U-Net model...")
    try:
        model = keras.models.load_model(model_path, custom_objects={'dice_coef': dice_coef})
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load the model from {model_path}")
        print(f"Error details: {e}")
        exit()

    print("\nLoading TEST data...") # MODIFIED
    try:
        X_test = np.load(os.path.join(processed_data_path, 'X_test.npy')) # MODIFIED
        y_test = np.load(os.path.join(processed_data_path, 'y_test.npy')) # MODIFIED
    except FileNotFoundError:
        print(f"ERROR: TEST data files not found in '{processed_data_path}'.")
        print("Please ensure preprocess_data.py was run successfully with patient-level splits.")
        exit()
    
    print(f"Test data shapes: X_test: {X_test.shape}, y_test: {y_test.shape}") # MODIFIED

    if len(X_test) == 0:
        print("Test data is empty. Cannot proceed.")
        exit()

    # --- Make Predictions on a few Test Samples for display ---
    print(f"\nMaking predictions on {NUM_SAMPLES_TO_DISPLAY} test samples for display...")
    indices_to_display = np.random.choice(len(X_test), NUM_SAMPLES_TO_DISPLAY, replace=False) # Random samples
    
    sample_images = X_test[indices_to_display]
    sample_true_masks = y_test[indices_to_display]
    
    predicted_probs_display = model.predict(sample_images)
    predicted_binary_display = (predicted_probs_display > THRESHOLD).astype(np.uint8)

    plt.figure(figsize=(15, NUM_SAMPLES_TO_DISPLAY * 4)) 
    for i in range(len(indices_to_display)):
        original_image = sample_images[i, :, :, 0] 
        true_mask = sample_true_masks[i, :, :, 0]
        pred_mask = predicted_binary_display[i, :, :, 0]
        
        dice_val = dice_coefficient_numpy(true_mask, pred_mask)
        
        plt.subplot(NUM_SAMPLES_TO_DISPLAY, 3, i * 3 + 1)
        plt.imshow(original_image, cmap='gray')
        plt.title(f"Test Sample {indices_to_display[i]}: Original")
        plt.axis('off')
        
        plt.subplot(NUM_SAMPLES_TO_DISPLAY, 3, i * 3 + 2)
        plt.imshow(true_mask, cmap='gray')
        plt.title(f"True Mask (Tumor Pixels: {np.sum(true_mask)})")
        plt.axis('off')
        
        plt.subplot(NUM_SAMPLES_TO_DISPLAY, 3, i * 3 + 3)
        plt.imshow(pred_mask, cmap='gray')
        plt.title(f"Predicted Mask (Dice: {dice_val:.4f})")
        plt.axis('off')

    plt.tight_layout()
    figure_save_path = os.path.join(os.path.dirname(model_path), 'test_set_evaluation_plot.png')
    try:
        plt.savefig(figure_save_path)
        print(f"\nTest set evaluation plot saved to: {figure_save_path}")
    except Exception as e:
        print(f"Error saving evaluation plot: {e}")
    plt.show()
    
    # --- Calculate Average Metrics over the ENTIRE TEST SET ---
    print(f"\nCalculating metrics over the entire test set ({len(X_test)} samples)...")
    
    # Predict on the entire test set (might need to do in batches if memory is an issue)
    # For ~500-700 images of 256x256, direct prediction should be okay on CPU with 16GB RAM
    all_test_pred_probs = model.predict(X_test, batch_size=16) # Added batch_size for predict
    all_test_pred_binary = (all_test_pred_probs > THRESHOLD).astype(np.uint8)
    
    all_dice_scores = []
    all_jaccard_scores = []
    all_sensitivities = []
    all_specificities = [] # Not as critical for foreground but good to see
    all_precisions = []
    
    num_slices_with_tumor_in_gt = 0
    num_slices_where_tumor_predicted = 0
    
    for i in range(len(X_test)):
        true_m = y_test[i, :, :, 0]
        pred_m = all_test_pred_binary[i, :, :, 0]
        
        # Only calculate segmentation metrics for slices that actually contain a tumor in ground truth
        # This is a common practice to avoid being skewed by many true negatives (empty masks)
        if np.sum(true_m) > 0: # If there is a tumor in the ground truth
            num_slices_with_tumor_in_gt += 1
            dice = dice_coefficient_numpy(true_m, pred_m)
            jaccard = jaccard_index_numpy(true_m, pred_m)
            sensitivity, _, precision, tp = calculate_pixel_metrics(true_m, pred_m) # ignore specificity from this func
            
            all_dice_scores.append(dice)
            all_jaccard_scores.append(jaccard)
            all_sensitivities.append(sensitivity)
            all_precisions.append(precision)

            if tp > 0 : # If model predicted any tumor pixels for this slice that has GT tumor
                num_slices_where_tumor_predicted +=1
        else: # If ground truth mask is empty
            # If prediction is also empty, it's a perfect true negative for this slice.
            # If prediction is not empty, it's a false positive for this slice.
            # These cases are handled by overall pixel specificity if calculated on all pixels.
            # For tumor-focused metrics, we often only score slices with GT tumors.
            pass 


    print("\n--- Test Set Metrics (calculated ONLY on slices with ground truth tumors) ---")
    if num_slices_with_tumor_in_gt > 0:
        print(f"  Number of test slices with ground truth tumor: {num_slices_with_tumor_in_gt} / {len(X_test)}")
        print(f"  Number of these slices where model predicted some tumor: {num_slices_where_tumor_predicted} / {num_slices_with_tumor_in_gt}")
        print(f"  Average Dice Coefficient: {np.mean(all_dice_scores):.4f} (Std: {np.std(all_dice_scores):.4f})")
        print(f"  Average Jaccard Index (IoU): {np.mean(all_jaccard_scores):.4f} (Std: {np.std(all_jaccard_scores):.4f})")
        print(f"  Average Sensitivity (Recall): {np.mean(all_sensitivities):.4f} (Std: {np.std(all_sensitivities):.4f})")
        print(f"  Average Precision: {np.mean(all_precisions):.4f} (Std: {np.std(all_precisions):.4f})")
    else:
        print("No slices with ground truth tumors found in the test set to calculate metrics.")

    print("\nEvaluation script finished.")