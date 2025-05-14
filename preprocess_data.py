import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split # For splitting data

# Define the main path to your dataset
dataset_base_path = r'C:\AI_Projects\BrainTumorProject\brain-tumor-segmentation-unet\data\kaggle_3m'
# Define where to save the processed NumPy arrays
output_numpy_path = r'C:\AI_Projects\BrainTumorProject\brain-tumor-segmentation-unet\data_processed'

# Create the output directory if it doesn't exist
os.makedirs(output_numpy_path, exist_ok=True)

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

print(f"Scanning dataset at: {dataset_base_path}")
# --- Get file lists (same as before) ---
try:
    patient_folders = [
        folder_name for folder_name in os.listdir(dataset_base_path)
        if os.path.isdir(os.path.join(dataset_base_path, folder_name))
    ]
except FileNotFoundError:
    print(f"ERROR: The directory '{dataset_base_path}' was not found. Please check the path.")
    exit()

if not patient_folders:
    print(f"No patient folders found in '{dataset_base_path}'. Make sure the data is there.")
    exit()

all_image_filepaths = []
all_mask_filepaths = []

for patient_folder_name in patient_folders:
    patient_folder_path = os.path.join(dataset_base_path, patient_folder_name)
    all_files_in_patient_folder = glob.glob(os.path.join(patient_folder_path, '*.tif'))
    
    patient_image_files = []
    patient_mask_files = []
    
    for file_path in all_files_in_patient_folder:
        file_name = os.path.basename(file_path)
        if '_mask.tif' in file_name:
            patient_mask_files.append(file_path)
        else:
            patient_image_files.append(file_path)
    
    patient_image_files.sort()
    patient_mask_files.sort()
    
    for img_path in patient_image_files:
        img_basename_no_ext = os.path.splitext(os.path.basename(img_path))[0]
        expected_mask_filename = img_basename_no_ext + '_mask.tif'
        expected_mask_path = os.path.join(patient_folder_path, expected_mask_filename)
        
        if expected_mask_path in patient_mask_files:
            all_image_filepaths.append(img_path)
            all_mask_filepaths.append(expected_mask_path)

if not all_image_filepaths:
    print("No image-mask file paths found. Exiting.")
    exit()
    
print(f"Found {len(all_image_filepaths)} image-mask pairs.")

# --- Load, preprocess, and collect all images and masks ---
# Initialize lists to hold the NumPy arrays
images_data = []
masks_data = []

print("Starting to load and preprocess images and masks...")
for i in range(len(all_image_filepaths)):
    img_path = all_image_filepaths[i]
    mask_path = all_mask_filepaths[i]
    
    # Load image as grayscale
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Load mask as grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Warning: Could not load image {img_path}. Skipping.")
        continue
    if mask is None:
        print(f"Warning: Could not load mask {mask_path}. Skipping.")
        continue
        
    # Resize (should not be needed if all are 256x256, but good practice)
    if image.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    if mask.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
        mask = cv2.resize(mask, (IMAGE_WIDTH, IMAGE_HEIGHT))
        
    # Normalize image to [0, 1]
    image = image / 255.0
    # Normalize mask to [0, 1] (since its values are 0 and 255)
    mask = mask / 255.0
    
    images_data.append(image)
    masks_data.append(mask)
    
    if (i + 1) % 500 == 0: # Print progress every 500 images
        print(f"  Processed {i+1}/{len(all_image_filepaths)} pairs.")

print("Finished loading and preprocessing.")

# Convert lists of images/masks to NumPy arrays
images_np = np.array(images_data)
masks_np = np.array(masks_data)

# Add channel dimension for TensorFlow/Keras (height, width, channels)
# Images will be (num_samples, 256, 256, 1)
# Masks will be (num_samples, 256, 256, 1)
images_np = np.expand_dims(images_np, axis=-1)
masks_np = np.expand_dims(masks_np, axis=-1)

print(f"\nImages NumPy array shape: {images_np.shape}") # (num_samples, height, width, 1)
print(f"Masks NumPy array shape: {masks_np.shape}")   # (num_samples, height, width, 1)
print(f"Images data type: {images_np.dtype}")
print(f"Masks data type: {masks_np.dtype}")
print(f"Min/Max image values: {np.min(images_np)}, {np.max(images_np)}")
print(f"Unique mask values: {np.unique(masks_np)}")


# --- Split data into training and validation sets ---
# test_size=0.2 means 20% for validation, 80% for training
# random_state ensures the split is the same every time you run the script
X_train, X_val, y_train, y_val = train_test_split(
    images_np, masks_np, test_size=0.2, random_state=42
)

print(f"\nTraining images shape: {X_train.shape}")
print(f"Training masks shape: {y_train.shape}")
print(f"Validation images shape: {X_val.shape}")
print(f"Validation masks shape: {y_val.shape}")

# --- Save the processed NumPy arrays ---
print("\nSaving processed data to NumPy files...")
np.save(os.path.join(output_numpy_path, 'train_images.npy'), X_train)
np.save(os.path.join(output_numpy_path, 'train_masks.npy'), y_train)
np.save(os.path.join(output_numpy_path, 'val_images.npy'), X_val)
np.save(os.path.join(output_numpy_path, 'val_masks.npy'), y_val)

# Alternatively, save into a single compressed .npz file
# np.savez_compressed(os.path.join(output_numpy_path, 'processed_data.npz'),
#                     train_images=X_train, train_masks=y_train,
#                     val_images=X_val, val_masks=y_val)

print(f"\nData saved successfully in '{output_numpy_path}'.")
print("Preprocessing complete!")
