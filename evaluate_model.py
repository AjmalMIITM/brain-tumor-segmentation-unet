import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix # For sensitivity, specificity, precision

# --- Configuration ---
processed_data_path = r'C:\AI_Projects\BrainTumorProject\brain-tumor-segmentation-unet\data_processed'
model_path = r'C:\AI_Projects\BrainTumorProject\brain-tumor-segmentation-unet\trained_models\unet_brain_tumor_best.keras' 

NUM_SAMPLES_TO_DISPLAY = 5
THRESHOLD = 0.5 

# --- Keras Custom Functions (must match those used in training for model loading) ---
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred, smooth=1e-6): # This is needed by combined_bce_dice_loss
    return 1 - dice_coef(y_true, y_pred, smooth)

def combined_bce_dice_loss(y_true, y_pred, bce_weight=0.5, dice_weight=0.5, smooth_dice=1e-6): # This is the loss function
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    d_loss = dice_loss(y_true, y_pred, smooth_dice)
    return bce_weight * bce + dice_weight * d_loss

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
    
    if len(y_true_f) == 0 or len(y_pred_f) == 0:
         # Handle cases with no foreground pixels if necessary, or ensure labels=[0,1] handles it.
         # For now, assuming binary masks and standard confusion matrix.
        pass

    # Ensure labels are [0,1] for binary segmentation to get tn, fp, fn, tp correctly
    cm = confusion_matrix(y_true_f, y_pred_f, labels=[0, 1])
    if cm.size == 4: # Ensure we get a 2x2 matrix
        tn, fp, fn, tp = cm.ravel()
    else: # Handle cases where one class might be missing in a slice (e.g. all background or all tumor)
          # This might happen if a slice is all tumor or all background and model predicts similarly
        if np.all(y_true_f == 0) and np.all(y_pred_f == 0): # All true negatives
            tn = len(y_true_f); fp = 0; fn = 0; tp = 0;
        elif np.all(y_true_f == 1) and np.all(y_pred_f == 1): # All true positives
            tn = 0; fp = 0; fn = 0; tp = len(y_true_f);
        # Add other edge cases if necessary, or rely on behavior of metrics for zero denominators
        else: # Fallback if confusion_matrix doesn't return 4 values as expected
            tp = np.sum(np.logical_and(y_pred_f == 1, y_true_f == 1))
            tn = np.sum(np.logical_and(y_pred_f == 0, y_true_f == 0))
            fp = np.sum(np.logical_and(y_pred_f == 1, y_true_f == 0))
            fn = np.sum(np.logical_and(y_pred_f == 0, y_true_f == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    return sensitivity, specificity, precision, tp

# --- Main part of the script ---
if __name__ == '__main__':
    print("Loading the trained U-Net model...")
    try:
        # MODIFIED: Pass all necessary custom objects
        model = keras.models.load_model(
            model_path, 
            custom_objects={
                'dice_coef': dice_coef, 
                'dice_loss': dice_loss, 
                'combined_bce_dice_loss': combined_bce_dice_loss
            }
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load the model from {model_path}")
        print(f"Error details: {e}")
        exit()

    print("\nLoading TEST data...")
    try:
        X_test = np.load(os.path.join(processed_data_path, 'X_test.npy'))
        y_test = np.load(os.path.join(processed_data_path, 'y_test.npy'))
    except FileNotFoundError:
        print(f"ERROR: TEST data files not found in '{processed_data_path}'.")
        print("Please ensure preprocess_data.py was run successfully with patient-level splits.")
        exit()
    
    print(f"Test data shapes: X_test: {X_test.shape}, y_test: {y_test.shape}")

    if len(X_test) == 0:
        print("Test data is empty. Cannot proceed.")
        exit()

    print(f"\nMaking predictions on {NUM_SAMPLES_TO_DISPLAY} test samples for display...")
    indices_to_display = np.random.choice(len(X_test), NUM_SAMPLES_TO_DISPLAY, replace=False)
    
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
    
    print(f"\nCalculating metrics over the entire test set ({len(X_test)} samples)...")
    
    all_test_pred_probs = model.predict(X_test, batch_size=16)
    all_test_pred_binary = (all_test_pred_probs > THRESHOLD).astype(np.uint8)
    
    all_dice_scores = []
    all_jaccard_scores = []
    all_sensitivities = []
    all_specificities = []
    all_precisions = []
    
    num_slices_with_tumor_in_gt = 0
    num_slices_where_tumor_predicted_for_gt_tumor_slices = 0 # Renamed for clarity
    
    for i in range(len(X_test)):
        true_m = y_test[i, :, :, 0]
        pred_m = all_test_pred_binary[i, :, :, 0]
        
        if np.sum(true_m) > 0: 
            num_slices_with_tumor_in_gt += 1
            dice = dice_coefficient_numpy(true_m, pred_m)
            jaccard = jaccard_index_numpy(true_m, pred_m)
            sensitivity, _, precision, tp_count = calculate_pixel_metrics(true_m, pred_m) # Using the updated name
            
            all_dice_scores.append(dice)
            all_jaccard_scores.append(jaccard)
            all_sensitivities.append(sensitivity)
            all_precisions.append(precision)

            if tp_count > 0 : 
                num_slices_where_tumor_predicted_for_gt_tumor_slices +=1
        # We don't calculate specificity per tumor-containing slice in this loop, 
        # but overall pixel specificity could be calculated once on all_y_test_flat and all_pred_flat if needed.

    print("\n--- Test Set Metrics (calculated ONLY on slices with ground truth tumors) ---")
    if num_slices_with_tumor_in_gt > 0:
        print(f"  Number of test slices with ground truth tumor: {num_slices_with_tumor_in_gt} / {len(X_test)}")
        print(f"  Number of these slices where model predicted some tumor: {num_slices_where_tumor_predicted_for_gt_tumor_slices} / {num_slices_with_tumor_in_gt}") # Using updated name
        print(f"  Average Dice Coefficient: {np.mean(all_dice_scores):.4f} (Std: {np.std(all_dice_scores):.4f})")
        print(f"  Average Jaccard Index (IoU): {np.mean(all_jaccard_scores):.4f} (Std: {np.std(all_jaccard_scores):.4f})")
        print(f"  Average Sensitivity (Recall): {np.mean(all_sensitivities):.4f} (Std: {np.std(all_sensitivities):.4f})")
        print(f"  Average Precision: {np.mean(all_precisions):.4f} (Std: {np.std(all_precisions):.4f})")
    else:
        print("No slices with ground truth tumors found in the test set to calculate metrics.")

    print("\nEvaluation script finished.")
