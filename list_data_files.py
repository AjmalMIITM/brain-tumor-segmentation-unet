import os
import glob # This module helps find files matching a pattern

# Define the main path to your dataset
# The 'r' before the string means it's a "raw string", which helps with backslashes in Windows paths.
dataset_base_path = r'C:\AI_Projects\BrainTumorProject\brain-tumor-segmentation-unet\data\kaggle_3m'

print(f"Scanning dataset at: {dataset_base_path}")

# Get a list of all patient folder names
# We only want directories, so we check with os.path.isdir()
try:
    patient_folders = [
        folder_name for folder_name in os.listdir(dataset_base_path)
        if os.path.isdir(os.path.join(dataset_base_path, folder_name))
    ]
except FileNotFoundError:
    print(f"ERROR: The directory '{dataset_base_path}' was not found. Please check the path.")
    exit() # Stop the script if the base path is wrong

if not patient_folders:
    print(f"No patient folders found in '{dataset_base_path}'. Make sure the data is there.")
    exit()

print(f"Found {len(patient_folders)} patient folders. First few: {patient_folders[:5]}")

all_image_files = []
all_mask_files = []

# Loop through each patient folder
for patient_folder_name in patient_folders:
    patient_folder_path = os.path.join(dataset_base_path, patient_folder_name)
    
    # Find all .tif files in the current patient folder
    # Using glob is easier for matching patterns
    all_files_in_patient_folder = glob.glob(os.path.join(patient_folder_path, '*.tif'))
    
    patient_image_files = []
    patient_mask_files = []
    
    for file_path in all_files_in_patient_folder:
        file_name = os.path.basename(file_path) # Get just the filename (e.g., TCGA_..._1.tif)
        if '_mask.tif' in file_name:
            patient_mask_files.append(file_path)
        else:
            patient_image_files.append(file_path)
    
    # Sort them to help with pairing (optional but good practice)
    patient_image_files.sort()
    patient_mask_files.sort()
    
    for img_path in patient_image_files:
        img_basename_no_ext = os.path.splitext(os.path.basename(img_path))[0] 
        expected_mask_filename = img_basename_no_ext + '_mask.tif'
        expected_mask_path = os.path.join(patient_folder_path, expected_mask_filename)
        
        if expected_mask_path in patient_mask_files:
            all_image_files.append(img_path)
            all_mask_files.append(expected_mask_path)

print(f"\nSuccessfully paired {len(all_image_files)} image files with mask files.")

print("\nFirst 5 image-mask pairs found:")
for i in range(min(5, len(all_image_files))): 
    print(f"Image: {all_image_files[i]}")
    print(f"Mask:  {all_mask_files[i]}\n")

if not all_image_files:
    print("No image-mask pairs were found. Check file naming and paths.")