import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

dataset_base_path = r'C:\AI_Projects\BrainTumorProject\brain-tumor-segmentation-unet\data\kaggle_3m'

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

all_image_files = []
all_mask_files = []

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
            all_image_files.append(img_path)
            all_mask_files.append(expected_mask_path)

if not all_image_files:
    print("No image-mask pairs were found by the script. Exiting.")
    exit()

print(f"Total image-mask pairs found: {len(all_image_files)}")

# --- MODIFIED PART: Loop to find and display a non-empty mask ---
found_non_empty_mask = False
# Let's check up to, say, the first 100 pairs to find a non-empty mask
# You can increase this number if needed.
# Or, to be more robust, you could shuffle all_image_files and all_mask_files first.
# For now, let's just iterate.

# Shuffle the pairs to get a random sample if you run it multiple times (optional)
# import random
# combined = list(zip(all_image_files, all_mask_files))
# random.shuffle(combined)
# all_image_files[:], all_mask_files[:] = zip(*combined)


for i in range(min(len(all_image_files), 200)): # Check up to 200 images
    sample_image_path = all_image_files[i]
    sample_mask_path = all_mask_files[i]

    # Load image as grayscale this time to simplify
    image = cv2.imread(sample_image_path, cv2.IMREAD_GRAYSCALE) 
    mask = cv2.imread(sample_mask_path, cv2.IMREAD_UNCHANGED) # Keep mask as is to see its original values

    if image is None:
        print(f"Warning: Could not load image at {sample_image_path} (sample {i+1})")
        continue # Skip to next image
    if mask is None:
        print(f"Warning: Could not load mask at {sample_mask_path} (sample {i+1})")
        continue # Skip to next image

    unique_mask_values = np.unique(mask)
    
    # Check if mask contains more than just 0 (or if it contains a value like 1 or 255)
    # We are looking for a mask that has at least two unique values (e.g., 0 and 1, or 0 and 255)
    # Or if it has only one unique value, that value is not 0.
    if len(unique_mask_values) > 1 or (len(unique_mask_values) == 1 and unique_mask_values[0] != 0):
        print(f"\nFound a non-empty mask at sample index {i}:")
        print(f"Loading sample image: {sample_image_path}")
        print(f"Loading sample mask:  {sample_mask_path}")
        
        print(f"\nImage Properties (Grayscale):")
        print(f"  Shape: {image.shape}")
        print(f"  Data type: {image.dtype}")
        print(f"  Min pixel value: {np.min(image)}")
        print(f"  Max pixel value: {np.max(image)}")

        print(f"\nMask Properties:")
        print(f"  Shape: {mask.shape}")
        print(f"  Data type: {mask.dtype}")
        print(f"  Unique pixel values: {unique_mask_values}")

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title(f"Sample Image (Grayscale)\nShape: {image.shape}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray') # Use 'gray' for now, or 'viridis' if values are diverse
        plt.title(f"Sample Mask\nShape: {mask.shape}\nUnique Vals: {unique_mask_values}")
        plt.axis('off')
        
        plt.suptitle(f"Patient: {os.path.basename(os.path.dirname(sample_image_path))}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        found_non_empty_mask = True
        break # Exit loop once a non-empty mask is found and displayed

if not found_non_empty_mask:
    print("\nCould not find a non-empty mask after checking several samples.")
    print("The first sample was all zeros. You might need to inspect more samples or check data.csv.")

