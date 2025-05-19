import os
import torch
import torch.nn.functional as F
import nibabel as nib
import pandas as pd
from piqa import SSIM

# Define paths
suvr_ground_dir = "Ground_NIFTI_MNI_SUVR"
suvr_model_dir = "Model_NIFTI_MNI_SUVR"
output_csv_path = "mse_ssim_results.csv"

# Initialize SSIM metric for 3D images
ssim_metric = SSIM(window_size=5, sigma=1.5, n_channels=1, reduction='mean', value_range=5.0).cuda()

# Get matching filenames
filenames = [f for f in os.listdir(suvr_ground_dir) if f.endswith(".nii")]

# Store results
data = []

# Process each pair of files
for filename in filenames:
    ground_path = os.path.join(suvr_ground_dir, filename)
    model_path = os.path.join(suvr_model_dir, filename)

    if os.path.exists(model_path):
        # Load NIfTI images
        ground_img = nib.load(ground_path).get_fdata()
        model_img = nib.load(model_path).get_fdata()

        # Convert to PyTorch tensors and move to GPU
        ground_tensor = torch.tensor(ground_img).float().cuda()
        model_tensor = torch.tensor(model_img).float().cuda()

        # Adjust shape for SSIM: (Batch=1, Channels=1, Depth, Height, Width)
        ground_tensor = ground_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        model_tensor = model_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        # Compute MSE
        mse_value = F.mse_loss(ground_tensor, model_tensor, reduction='mean').item()

        # Compute SSIM
        ssim_value = ssim_metric(ground_tensor, model_tensor).item()

        # Extract patient ID (first 3 characters of filename)
        patient_id = filename[:3]

        # Store results
        data.append([filename, patient_id, mse_value, ssim_value])

        # Print result
        print(f"{filename}: MSE = {mse_value:.6f}, SSIM = {ssim_value:.6f}")

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Filename", "Patient_ID", "MSE", "SSIM"])

# Save full per-image results without modification
df.to_csv(output_csv_path, index=False)

# Print final results
print("\nFinal Results Saved:")
print(f"Full data: {output_csv_path}")