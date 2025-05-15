import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedGroupKFold # For splitting data
import random  # For shuffling

# Define the main path to your dataset
dataset_base_path = r'C:\AI_Projects\BrainTumorProject\brain-tumor-segmentation-unet\data\kaggle_3m'
# Define where to save the processed NumPy arrays and patient ID lists
output_numpy_path = r'C:\AI_Projects\BrainTumorProject\brain-tumor-segmentation-unet\data_processed'

# Create the output directory if it doesn't exist
os.makedirs(output_numpy_path, exist_ok=True)

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

# --- Seed for Reproducibility ---
SEED = 42  # You can change this, but keep it consistent for your experiments
random.seed(SEED)
np.random.seed(SEED)

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

# --- Extract Patient IDs ---
patient_ids = []
for img_path in all_image_filepaths:
    patient_id = os.path.basename(os.path.dirname(img_path))  # Extract folder name
    patient_ids.append(patient_id)

# --- Create Train/Validation/Test Splits of Patient IDs ---
# The strategy is simple splits with `train_test_split`.
# Could use StratifiedKFold instead if class imbalance at patient level is important.
unique_patient_ids = sorted(list(set(patient_ids)))  # Get unique IDs and sort for consistency

# First split: Train vs. (Val + Test)
train_patient_ids, temp_patient_ids = train_test_split(
    unique_patient_ids,
    test_size=0.3, # 30% for val+test combined
    random_state=SEED, # For reproducibility
    shuffle=True # Shuffle patient IDs
)

# Second split: Val vs. Test (split the remaining 30% into 50% val, 50% test)
val_patient_ids, test_patient_ids = train_test_split(
    temp_patient_ids,
    test_size=0.5, # 50% of the remaining for test
    random_state=SEED,
    shuffle=True
)

print(f"Number of training patients: {len(train_patient_ids)}")
print(f"Number of validation patients: {len(val_patient_ids)}")
print(f"Number of test patients: {len(test_patient_ids)}")

# --- Assign Images and Masks to the Correct Split ---
train_images_list = []
train_masks_list = []
val_images_list = []
val_masks_list = []
test_images_list = []
test_masks_list = []

for i in range(len(all_image_filepaths)):
    img_path = all_image_filepaths[i]
    mask_path = all_mask_filepaths[i]
    patient_id = patient_ids[i]  # Get the patient ID for this image/mask pair
    
    if patient_id in train_patient_ids:
        train_images_list.append(img_path)
        train_masks_list.append(mask_path)
    elif patient_id in val_patient_ids:
        val_images_list.append(img_path)
        val_masks_list.append(mask_path)
    elif patient_id in test_patient_ids:
        test_images_list.append(img_path)
        test_masks_list.append(mask_path)

print(f"Number of training images: {len(train_images_list)}")
print(f"Number of validation images: {len(val_images_list)}")
print(f"Number of test images: {len(test_images_list)}")

# --- Load, preprocess, and collect all images and masks ---
def load_and_preprocess(image_list, mask_list):
    images_data = []
    masks_data = []
    
    for i in range(len(image_list)):
        img_path = image_list[i]
        mask_path = mask_list[i]
        
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
    
    # Convert lists of images/masks to NumPy arrays
    images_np = np.array(images_data)
    masks_np = np.array(masks_data)
    
    # Add channel dimension for TensorFlow/Keras (height, width, channels)
    # Images will be (num_samples, 256, 256, 1)
    # Masks will be (num_samples, 256, 256, 1)
    images_np = np.expand_dims(images_np, axis=-1)
    masks_np = np.expand_dims(masks_np, axis=-1)
    
    return images_np, masks_np

print("\nLoading and preprocessing training data...")
X_train, y_train = load_and_preprocess(train_images_list, train_masks_list)
print("\nLoading and preprocessing validation data...")
X_val, y_val = load_and_preprocess(val_images_list, val_masks_list)
print("\nLoading and preprocessing test data...")
X_test, y_test = load_and_preprocess(test_images_list, test_masks_list)


print(f"\nTraining images shape: {X_train.shape}") # (num_samples, height, width, 1)
print(f"Training masks shape: {y_train.shape}")   # (num_samples, height, width, 1)
print(f"Validation images shape: {X_val.shape}")
print(f"Validation masks shape: {y_val.shape}")
print(f"Test images shape: {X_test.shape}")
print(f"Test masks shape: {y_test.shape}")

# --- Save the processed NumPy arrays ---
print("\nSaving processed data to NumPy files...")
np.save(os.path.join(output_numpy_path, 'X_train.npy'), X_train)
np.save(os.path.join(output_numpy_path, 'y_train.npy'), y_train)
np.save(os.path.join(output_numpy_path, 'X_val.npy'), X_val)
np.save(os.path.join(output_numpy_path, 'y_val.npy'), y_val)
np.save(os.path.join(output_numpy_path, 'X_test.npy'), X_test)
np.save(os.path.join(output_numpy_path, 'y_test.npy'), y_test)

# --- Save patient ID lists to text files ---
def save_patient_ids(patient_ids, filename):
    filepath = os.path.join(output_numpy_path, filename)
    with open(filepath, 'w') as f: # Open file in write mode ('w')
        for patient_id in patient_ids:
            f.write(f"{patient_id}\n") # Write each ID followed by a newline
    print(f"Patient IDs saved to: {filepath}")

save_patient_ids(train_patient_ids, 'train_patient_ids.txt')
save_patient_ids(val_patient_ids, 'val_patient_ids.txt')
save_patient_ids(test_patient_ids, 'test_patient_ids.txt')


print("\nData saved successfully in '{}'.".format(output_numpy_path))
print("Preprocessing complete!")
