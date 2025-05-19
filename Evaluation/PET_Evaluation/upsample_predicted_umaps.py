"""
Upsample Model Output to Original Attenuation Template Dimensions
==================================================================

This script takes the model-generated attenuation maps and:
- Upsamples them to match the spatial dimensions of the original attenuation templates.
- Applies the affine transformation from the original templates to ensure spatial alignment.

Directory Configuration:
------------------------
- input_dir  : "FromModel"   # Directory containing model output attenuation maps.
- affine_dir : "ATT"         # Directory containing reference NIfTI files with desired affine matrices.
- output_dir : "upconverted" # Directory to save the upsampled and affine-aligned output.

Requirements:
-------------
- nibabel
- numpy
- scipy
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib

def crop_upsample_replace_values_with_affine(input_file, affine_file, output_file):
    """
    Crop, upsample, replace values, and use affine matrix from another NIfTI file.

    Args:
        input_file (str): Path to the input NIfTI file.
        affine_file (str): Path to the NIfTI file from which to extract the affine matrix.
        output_file (str): Path to save the processed NIfTI file.
    """
    # Load the input NIfTI file
    nii = nib.load(input_file)
    data = nii.get_fdata()  # Load the data as a NumPy array

    # Load the affine matrix from the specified NIfTI file
    affine_nii = nib.load(affine_file)
    affine = affine_nii.affine  # Extract the affine matrix

    # Step 1: Crop the third dimension (48 to 47)
    cropped_data = data[:, :, :47]  # Crop the last slice

    # Step 2: Upsample to 256x256x175
    input_size = cropped_data.shape  # (128, 128, 47)
    upsample_size = (256, 256, 175)  # Intermediate target size

    # Convert the data to a PyTorch tensor
    cropped_tensor = torch.tensor(cropped_data, dtype=torch.float32)

    # Get the unique classes (assumes segmentation map with discrete values)
    classes = torch.unique(cropped_tensor)

    # Create one-hot encoding (C x D x H x W)
    one_hot = torch.stack([(cropped_tensor == cls).float() for cls in classes], dim=0)

    # Reshape one-hot encoding for interpolation (N x C x D x H x W)
    one_hot = one_hot.unsqueeze(0)  # Add batch dimension

    # Perform interpolation for upsampling
    interpolated = F.interpolate(
        one_hot,
        size=upsample_size,  # Target size (D, H, W)
        mode="trilinear",  # Trilinear interpolation
        align_corners=True
    )

    # Merge the upsampled probabilities by taking argmax across class dimension
    upsampled_tensor = torch.argmax(interpolated, dim=1).squeeze(0)  # Remove batch dimension

    # Convert back to NumPy array
    upsampled_data = upsampled_tensor.numpy()

    # Step 3: Replace values
    value_mapping = {0: 0, 1: 4, 2: 3}
    replaced_data = np.zeros_like(upsampled_data, dtype=np.uint8)
    for original, new in value_mapping.items():
        replaced_data[upsampled_data == original] = new

    # Step 4: Embed in a 256 slice dimension with padding
    output_data = np.zeros((256, 256, 256), dtype=np.uint8)  # Use uint8
    output_data[:, :, 53:228] = replaced_data  # Embed the 175 slices

    # Save the processed data as a new NIfTI file with the provided affine matrix
    new_nii = nib.Nifti1Image(output_data, affine)
    nib.save(new_nii, output_file)

def process_all_files(input_dir, affine_dir, output_dir):
    """
    Iterate over all NIfTI files in the input directory, generate corresponding affine and output filenames, and process them.

    Args:
        input_dir (str): Directory containing input NIfTI files.
        affine_dir (str): Directory containing affine NIfTI files.
        output_dir (str): Directory to save processed NIfTI files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".nii"):
            input_file = os.path.join(input_dir, filename)
            
            # Extract first three characters for affine file
            base_name = filename[:3]  # Example: "316"
            affine_file = os.path.join(affine_dir, f"{base_name}_att_map_SimSET.nii")
            
            # Construct output file name
            output_filename = filename.replace(".nii", "_att.nii")
            output_file = os.path.join(output_dir, output_filename)
            
            if os.path.exists(affine_file):
                print(f"Processing: {input_file} -> {output_file}")
                crop_upsample_replace_values_with_affine(input_file, affine_file, output_file)
            else:
                print(f"Skipping {input_file}, missing affine file: {affine_file}")

# Define paths
input_dir = r"FromModel"
affine_dir = r"ATT"
output_dir = r"upconverted"

# Process all files
process_all_files(input_dir, affine_dir, output_dir)
