import os
import numpy as np
import nibabel as nib
from glob import glob

# Define paths
ground_truth_dir = "L:/Results_Compare/Ground_NIFTI_MNI"
model_output_dir = "L:/Results_Compare/Model_NIFTI_MNI"
output_ground_suvr = "L:/Results_Compare/Ground_NIFTI_MNI_SUVR"
output_model_suvr = "L:/Results_Compare/Model_NIFTI_MNI_SUVR"
cerebellum_mask_path = "L:/Results_Compare/cerebellum_mask.nii"

# Ensure output directories exist
os.makedirs(output_ground_suvr, exist_ok=True)
os.makedirs(output_model_suvr, exist_ok=True)

# Load cerebellum mask
cerebellum_mask_img = nib.load(cerebellum_mask_path)
cerebellum_mask_data = cerebellum_mask_img.get_fdata()

# Identify cerebellum voxels
cerebellum_voxels = cerebellum_mask_data == 1
num_cerebellum_voxels = np.sum(cerebellum_voxels)  # N_cerebellum

def process_nifti_files(input_dir, output_dir):
    """ Process all NIfTI files in the input directory and compute SUVR. """
    for file_path in glob(os.path.join(input_dir, "*.nii")):
        # Load image
        img = nib.load(file_path)
        img_data = img.get_fdata()
        img_data = np.nan_to_num(img_data, nan=0.0)

        # Compute sum of voxel values in the cerebellum
        sum_cerebellum_values = np.sum(img_data[cerebellum_voxels])

        if sum_cerebellum_values == 0:
            print(f"Skipping {file_path} due to zero cerebellum sum.")
            continue

        # Compute SUVR
        suvr_data = img_data * (num_cerebellum_voxels / sum_cerebellum_values)

        # Save SUVR image
        output_path = os.path.join(output_dir, os.path.basename(file_path))
        suvr_img = nib.Nifti1Image(suvr_data, img.affine, img.header)
        nib.save(suvr_img, output_path)

        print(f"Saved SUVR image to {output_path}")

# Process both sets of NIfTI files
process_nifti_files(ground_truth_dir, output_ground_suvr)
process_nifti_files(model_output_dir, output_model_suvr)

print("Processing complete!")
