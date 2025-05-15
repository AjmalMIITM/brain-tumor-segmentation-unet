import os
import sys  # For command-line arguments
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt

# --- Configuration ---
# Path to your trained model
model_path = r'C:\AI_Projects\BrainTumorProject\brain-tumor-segmentation-unet\trained_models\unet_brain_tumor_best.keras'

# Define image size (must match training size)
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# --- Dice Coefficient Metric Function (must match the one used in training) ---
def dice_coef(y_true, y_pred, smooth=1e-6): # This is needed to load the model
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# --- Main part of the script ---
if __name__ == '__main__':
    # --- Get image path from command line ---
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        image_path = input("Please enter the path to the MRI slice image: ") # Or prompt user
        if not image_path: # If user just presses Enter
            print("No image path provided. Exiting.")
            sys.exit(1) # Exit with an error code
    else:
        image_path = sys.argv[1] # Get path from command line argument

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at path: {image_path}")
        sys.exit(1)

    print(f"Loading image: {image_path}")

    # --- Load and preprocess the image ---
    try:
        # Load as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"ERROR: Could not load image using OpenCV. Ensure it's a valid image file.")
            sys.exit(1)

        # Resize
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))

        # Normalize
        img = img / 255.0

        # Add channel dimension: (256, 256) -> (256, 256, 1)
        img = np.expand_dims(img, axis=-1)

        # Add batch dimension: (256, 256, 1) -> (1, 256, 256, 1)
        img = np.expand_dims(img, axis=0)

        print(f"Preprocessed image shape: {img.shape}") # Should be (1, 256, 256, 1)
    except Exception as e:
        print(f"ERROR during image loading/preprocessing: {e}")
        sys.exit(1)

    # --- Load the trained model ---
    print("Loading the trained U-Net model...")
    try:
        # Load the model with the custom dice_coef function
        model = keras.models.load_model(model_path, custom_objects={'dice_coef': dice_coef})
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR: Could not load the model from {model_path}")
        print(f"Error details: {e}")
        sys.exit(1)

    # --- Make prediction ---
    print("Making prediction...")
    try:
        # Get the predicted probabilities
        predicted_mask_probs = model.predict(img)

        # Threshold to get a binary mask (0 or 1)
        predicted_mask = (predicted_mask_probs[0, ...] > 0.5).astype(np.uint8)  # Remove batch dim

        print(f"Predicted mask shape: {predicted_mask.shape}") # Should be (256, 256, 1)

        # Remove channel dimension for display purposes
        predicted_mask = predicted_mask[:, :, 0]
    except Exception as e:
        print(f"ERROR during prediction: {e}")
        sys.exit(1)

    # --- Display the results ---
    print("Displaying results...")
    try:
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        # Load original image again to display, as img is now preprocessed
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is not None:
             original_img = cv2.resize(original_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
             plt.imshow(original_img, cmap='gray') # Show original image
        else:
            plt.imshow(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH)), cmap='gray')
            print("Warning: Could not load original image for display.")
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(predicted_mask, cmap='gray') # Display predicted mask
        plt.title("Predicted Mask")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"ERROR during display: {e}")
        sys.exit(1)

    print("\nPrediction script finished.")
