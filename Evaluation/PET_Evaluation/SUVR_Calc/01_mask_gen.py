import nibabel as nib
import numpy as np

# Path to the AAL3 ROI file
aal3_path = r"K:\spm12\spm12\toolbox\AAL3\ROI_MNI_V7.nii"

# Load the AAL3 ROI image
aal3_image = nib.load(aal3_path)
aal3_data = aal3_image.get_fdata()

# Define cerebellum labels (95 to 112)
cerebellum_labels = list(range(95, 113))  # Include labels 95 to 112

# Create the cerebellum binary mask
cerebellum_mask = np.isin(aal3_data, cerebellum_labels).astype(np.int)

# Save the cerebellum mask as a new NIfTI file
cerebellum_mask_image = nib.Nifti1Image(cerebellum_mask, affine=aal3_image.affine)
output_path = r"cerebellum_mask.nii"
nib.save(cerebellum_mask_image, output_path)

print(f"Cerebellum mask saved at: {output_path}")
