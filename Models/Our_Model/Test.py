import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import nibabel as nib
import matplotlib.pyplot as plt
from torchsummary import summary
from torchviz import make_dot
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import make_grid


import wandb

from torch_radon import ParallelBeam, Volume2D

dropout_val_enc = 0.10
dropout_val_dec = 0.60
dropout_val_bn = 0.10

wd_val = 0.01
lr_rate = 0.00001
epochs_val = 100
batch_size_val = 8
model_arch_name = f"UNET-3D-Radon-upsample-{epochs_val}-{lr_rate}-{wd_val}-{batch_size_val}"

workers = 24

wandb.init(
    # set the wandb project where this run will be logged
    project="pseudo-uMap-recon-unet",

    # track hyperparameters and run metadata
    config={
    "learning_rate": lr_rate,
    "architecture": model_arch_name,
    "dataset": "Simulated",
    "epochs": epochs_val,
    }
)

# Automatically get the filename of the running script
script_name = os.path.basename(__file__)  # Get the current script's filename
wandb.save(script_name)  # Upload the script to WandB

print(f"Uploaded script: {script_name} to WandB.")

# Create directories for saving results 
# removed _{model_arch_name}
os.makedirs(rf"results/train", exist_ok=True)
os.makedirs(rf"results/val", exist_ok=True)
os.makedirs(rf"results/hole", exist_ok=True)
os.makedirs(rf"results/test", exist_ok=True)

# removed test_inputs_{model_arch_name} -> preprocessed_input
os.makedirs(rf"preprocessed_input", exist_ok=True)
# removed test_outputs_{model_arch_name} -> preprocessed_target
os.makedirs(rf"preprocessed_target", exist_ok=True)

# Define the CustomCrossEntropyLoss        
class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()
        
    def weight_func(self, targets):
        # Define the weights based on the target values
        weights = torch.ones_like(targets, dtype=torch.float)  # Initialize with ones and float type
        weights[targets == 0] = 0.0348
        weights[targets == 1] = 0.0614
        weights[targets == 2] = 0.9038
        


        return weights        

    def forward(self, predictions, targets):
        targets = targets.squeeze(1)
        
        # Convert targets to Long type
        targets = targets.long()
        # Calculate the weight for each class
        weights = self.weight_func(targets)
        
        # Use F.cross_entropy which combines LogSoftmax and NLLLoss in one single function
        # Apply weights to each element in the batch before taking the mean
        loss = F.cross_entropy(predictions, targets, reduction='none')
        weighted_loss = weights * loss
        
        return torch.mean(weighted_loss)        


# Define the 3D U-Net Model

### Define DoubleConv block
class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

### Define DownsampleBlock
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DownsampleBlock, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.downsample(x)


### Define Custom skip connection
class CustomSkipOperation(nn.Module):
    def __init__(self, in_channels, det_count, n_angles, width, filter_name="ramp"):
        super(CustomSkipOperation, self).__init__()
        self.filter_name = filter_name
        self.det_count = det_count
        self.n_angles = n_angles
        self.width = width

    def forward(self, x, encoder_input_dims, decoder_target_shape):
        b, c, d, h, w = x.shape
        device = x.device

        # Compute angles on the correct device
        angles = torch.linspace(0, np.pi, self.n_angles, requires_grad=False).float().to(device)

        # Initialize radon transform
        volume = Volume2D()
        volume.set_size(self.width, self.width)
        radon = ParallelBeam(self.det_count, angles, volume=volume)

        # Validate dimensions and transpose if necessary
        if h != self.n_angles or w != self.det_count:
            if w == self.n_angles and h == self.det_count:
                #print("Warning: Detected swapped dimensions. Transposing d and h.")
                x = x.permute(0, 1, 2, 4, 3)  # Transpose d and h
                h, w = w, h
            else:
                raise ValueError(
                    f"shape: {x.shape}"
                    f"Dimension mismatch: "
                    f"Expected d={self.n_angles}, h={self.det_count}, but got d={h}, h={w}."
                )

        #print(f"Dimensions after potential transpose: d={d}, h={h}, n_angles={self.n_angles}, det_count={self.det_count}")

        # Combine batch, channel, and slice dimensions for batched processing
        x_batched = x.permute(0, 1, 2, 3, 4).reshape(-1, h, w)  # Shape: (b*c*d, h, w)

        # Process in chunks to avoid GPU limits
        chunk_size = 1024  # Adjust based on your GPU capacity
        img_slices = []
        for i in range(0, x_batched.shape[0], chunk_size):
            chunk = x_batched[i:i+chunk_size].float().to(device)
            #print(f"Processing chunk {i // chunk_size + 1}, shape: {chunk.shape}")

            # Perform filtering and back-projection
            filtered_sinos = radon.filter_sinogram(chunk, filter_name=self.filter_name)
            img_slices_chunk = radon.backward(filtered_sinos)

            img_slices.append(img_slices_chunk)
            torch.cuda.empty_cache()


        # Concatenate processed slices and reshape
        img_slices = torch.cat(img_slices, dim=0)
        reconstructed_volume = img_slices.reshape(b, c, d, self.width, self.width).permute(0, 1, 2, 3, 4)

        # Re-transpose dimensions to original order if necessary
        if w == self.det_count:
            #print("Re-transposing dimensions to original order.")
            reconstructed_volume = reconstructed_volume.permute(0, 1, 2, 3, 4)  # Re-transpose h and d

        # Perform upsampling to match decoder target shape
        #x = F.interpolate(reconstructed_volume, size=decoder_target_shape, mode="trilinear", align_corners=False)

        corrected_volume = torch.flip(reconstructed_volume, dims=[-2, -1])
        return self.upsample_shift_crop(corrected_volume) # center_crop_and_interpolate(reconstructed_volume) #x

    def center_crop_and_interpolate(self, tensor, mode='trilinear'):
        """
        Center crops the tensor in H and W dimensions to half the size,
        then resizes it back to the original dimensions using trilinear interpolation.
        
        Args:
            tensor (torch.Tensor): The input tensor of shape (N, C, D, H, W).
            mode (str): Interpolation mode. Default is 'trilinear'.
        
        Returns:
            torch.Tensor: The processed tensor.
        """
        assert len(tensor.shape) == 5, "Input tensor must have 5 dimensions (N, C, D, H, W)"
        
        # Get original dimensions
        N, C, D, H, W = tensor.shape
        
        # Calculate cropping dimensions
        crop_h, crop_w = H // 2, W // 2
        start_h = (H - crop_h) // 2
        start_w = (W - crop_w) // 2
        
        # Perform center crop
        cropped_tensor = tensor[:, :, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Resize back to original dimensions using trilinear interpolation
        resized_tensor = F.interpolate(
            cropped_tensor,
            size=(D, H, W),
            mode=mode,
            align_corners=False  # Prevents boundary artifacts
        )
        
        return resized_tensor

    def upsample_shift_crop(self,tensor, upsample_factor=1.8, shift_h=0, shift_w=0, crop_h=None, crop_w=None, mode='trilinear'):
        """
        Upsamples the input tensor, applies shifting, and crops it to match original dimensions.
        
        Args:
            tensor (torch.Tensor): Input tensor of shape (N, C, D, H, W).
            upsample_factor (float): Factor for upsampling.
            shift_h (int): Vertical shift for the processed tensor.
            shift_w (int): Horizontal shift for the processed tensor.
            crop_h (int): Target height for cropping (default: original height).
            crop_w (int): Target width for cropping (default: original width).
            mode (str): Interpolation mode for resizing (default 'trilinear').
            
        Returns:
            torch.Tensor: Processed tensor with the same shape as the original tensor.
        """
        assert len(tensor.shape) == 5, "Input tensor must have 5 dimensions (N, C, D, H, W)"
        
        # Get original dimensions
        N, C, D, H, W = tensor.shape

        # Step 1: Upsample
        upsampled_h = int(H * upsample_factor)
        upsampled_w = int(W * upsample_factor)
        upsampled_tensor = F.interpolate(
            tensor,
            size=(D, upsampled_h, upsampled_w),
            mode=mode,
            align_corners=False
        )

        # Step 2: Shift (by padding or cropping)
        padded_tensor = F.pad(
            upsampled_tensor,
            pad=(max(0, shift_w), max(0, -shift_w), max(0, shift_h), max(0, -shift_h)),
            mode="constant",
            value=0
        )

        # Step 3: Crop to target dimensions
        # Set crop_h and crop_w to the original dimensions if not provided
        crop_h = crop_h or H
        crop_w = crop_w or W

        # Calculate cropping start and end positions
        start_h = (padded_tensor.shape[-2] - crop_h) // 2
        start_w = (padded_tensor.shape[-1] - crop_w) // 2

        # Ensure the cropping indices are within bounds
        start_h = max(0, start_h)
        start_w = max(0, start_w)

        cropped_tensor = padded_tensor[
            :, :, :, 
            start_h:start_h + crop_h, 
            start_w:start_w + crop_w
        ]

        # Ensure the final size matches the original height and width exactly
        assert cropped_tensor.shape[-2] == crop_h, "Final height does not match target crop height."
        assert cropped_tensor.shape[-1] == crop_w, "Final width does not match target crop width."

        return cropped_tensor

### Define 3D UNet
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob_enc=0.5,dropout_prob_dec=0.5,dropout_prob_bn=0.5):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_prob_enc = dropout_prob_enc
        self.dropout_prob_dec = dropout_prob_dec
        self.dropout_prob_bn = dropout_prob_bn

        # Encoder with varying kernel sizes
        self.encoder1 = DoubleConv(in_channels, 32, 64, kernel_size=7)
        self.pool1 = DownsampleBlock(64, 64, kernel_size=5)
        self.dropout1 = nn.Dropout3d(self.dropout_prob_enc)

        self.encoder2 = DoubleConv(64, 64, 128, kernel_size=5)
        self.pool2 = DownsampleBlock(128, 128, kernel_size=5)
        self.dropout2 = nn.Dropout3d(self.dropout_prob_enc)

        self.encoder3 = DoubleConv(128, 128, 256, kernel_size=5)
        self.pool3 = DownsampleBlock(256, 256, kernel_size=3)
        self.dropout3 = nn.Dropout3d(self.dropout_prob_enc)

        self.encoder4 = DoubleConv(256, 256, 512, kernel_size=3)
        self.pool4 = DownsampleBlock(512, 512, kernel_size=3)
        self.dropout4 = nn.Dropout3d(self.dropout_prob_enc)

        # Bottleneck with feature expansion and compression
        self.bottleneck_expand = DoubleConv(512, 1024, 1024, kernel_size=3)
        self.bottleneck = DoubleConv(1024, 1024, 512, kernel_size=3)
        self.dropout_bottleneck = nn.Dropout3d(self.dropout_prob_bn)

        # Bottleneck upsampling
        self.upsample_bottleneck = nn.Upsample(size=(3, 8, 8), mode='trilinear', align_corners=False)
        self.smooth_conv = nn.Conv3d(512, 512, kernel_size=1)

        # Custom Skip Operations
        self.skip_op1 = CustomSkipOperation(64, 96, 216, 128)
        self.skip_op2 = CustomSkipOperation(128, 48, 108, 64)
        self.skip_op3 = CustomSkipOperation(256, 24, 54, 32)
        self.skip_op4 = CustomSkipOperation(512, 12, 27, 16)

        # Decoder with fixed kernel size of 3
        self.upconv4 = nn.ConvTranspose3d(512, 1024, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024 + 512, 512, 512, kernel_size=3)
        self.dropout_dec4 = nn.Dropout3d(self.dropout_prob_dec)

        self.upconv3 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512 + 256, 256, 256, kernel_size=3)
        self.dropout_dec3 = nn.Dropout3d(self.dropout_prob_dec)

        self.upconv2 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256 + 128, 128, 128, kernel_size=3)
        self.dropout_dec2 = nn.Dropout3d(self.dropout_prob_dec)

        self.upconv1 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128 + 64, 64, 64, kernel_size=3)
        self.dropout_dec1 = nn.Dropout3d(self.dropout_prob_dec)

        # Output layer
        self.output_layer_0 = nn.Conv3d(64, 3, stride=(1, 1, 1), kernel_size=1)
        #self.output_layer_1 = Upscale3DModel()

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc2 = self.dropout1(enc2)

        enc3 = self.encoder3(self.pool2(enc2))
        enc3 = self.dropout2(enc3)

        enc4 = self.encoder4(self.pool3(enc3))
        enc4 = self.dropout3(enc4)

        # Bottleneck with feature expansion and compression
        bottleneck = self.bottleneck_expand(self.pool4(enc4))
        bottleneck = self.bottleneck(bottleneck)
        bottleneck = self.dropout_bottleneck(bottleneck)

        # Bottleneck upsampling
        bottleneck_upsampled = self.upsample_bottleneck(bottleneck)
        bottleneck_upsampled = self.smooth_conv(bottleneck_upsampled)

        # Decoder with custom skip connections
        dec4 = self.upconv4(bottleneck_upsampled)
        enc4_skip = self.skip_op4(enc4, enc4.shape[2:], dec4.shape[2:])
        #print(f"dec4 shape: {dec4.shape}, enc4_skip shape: {enc4_skip.shape}")

        dec4 = torch.cat((dec4, enc4_skip), dim=1)
        dec4 = self.decoder4(dec4)
        dec4 = self.dropout_dec4(dec4)
        #print("passed dec4")

        dec3 = self.upconv3(dec4)
        enc3_skip = self.skip_op3(enc3, enc3.shape[2:], dec3.shape[2:])
        dec3 = torch.cat((dec3, enc3_skip), dim=1)
        dec3 = self.decoder3(dec3)
        dec3 = self.dropout_dec3(dec3)
        #print("passed dec3")

        dec2 = self.upconv2(dec3)
        enc2_skip = self.skip_op2(enc2, enc2.shape[2:], dec2.shape[2:])
        dec2 = torch.cat((dec2, enc2_skip), dim=1)
        dec2 = self.decoder2(dec2)
        dec2 = self.dropout_dec2(dec2)
        #print("passed dec2")

        dec1 = self.upconv1(dec2)
        enc1_skip = self.skip_op1(enc1, enc1.shape[2:], dec1.shape[2:])
        dec1 = torch.cat((dec1, enc1_skip), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.dropout_dec1(dec1)

        out0 = self.output_layer_0(dec1)

        return out0  #self.output_layer_1(out0)


######## NOISE START

# Define the noise augmentation function
class AddRandomNoise:
    def __init__(self, mean=0.0, std=0.1, clip_min=0.0, clip_max=1.0):
        """
        Add random Gaussian noise to the input tensor.
        Args:
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.
            clip_min (float): Minimum value to clip the output to.
            clip_max (float): Maximum value to clip the output to.
        """
        self.mean = mean
        self.std = std
        self.clip_min = clip_min
        self.clip_max = clip_max

    def __call__(self, tensor):
        """
        Apply the transformation.
        Args:
            tensor (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: Tensor with added noise, clipped to the valid range.
        """
        noise = torch.randn_like(tensor) * self.std + self.mean
        augmented_tensor = tensor + noise
        # Clip values to ensure they stay within [0, 1]
        return torch.clamp(augmented_tensor, min=self.clip_min, max=self.clip_max)


# Wrapper dataset to apply Gaussian noise
class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        input_data, output_data, filename_only = self.base_dataset[idx]
        if self.transform:
            input_data = self.transform(input_data)
        return input_data, output_data, filename_only


######## NOISE END

def downsample_with_f_interpolate_fixed_sizes(segmentation_map):
    """
    Downsample a 3D segmentation map from 256x256x175 to 128x128x48 using trilinear interpolation.
    Args:
        segmentation_map (numpy.ndarray): Input segmentation map of shape (256, 256, 175).
    Returns:
        numpy.ndarray: Downsampled segmentation map of shape (128, 128, 48).
    """
    # Define fixed sizes
    input_size = (256, 256, 175)  # Original size
    output_size = (128, 128, 48)  # Target size

    # Ensure the input segmentation map matches the expected input size
    assert segmentation_map.shape == input_size, f"Input must have shape {input_size}, but got {segmentation_map.shape}"
    
    # Convert the segmentation map (numpy array) to a PyTorch tensor
    segmentation_map = torch.tensor(segmentation_map, dtype=torch.float32)
    
    # Get the unique classes
    classes = torch.unique(segmentation_map)  # e.g., [0, 1, 2]
    
    # Create one-hot encoding (C x D x H x W)
    one_hot = torch.stack([(segmentation_map == cls).float() for cls in classes], dim=0)
    
    # Reshape one-hot encoding for interpolation (N x C x D x H x W)
    one_hot = one_hot.unsqueeze(0)  # Add batch dimension
    
    # Perform interpolation
    interpolated = F.interpolate(
        one_hot, 
        size=output_size,  # Target size (D, H, W)
        mode="trilinear",  # Trilinear interpolation
        align_corners=True
    )
    
    # Merge the downsampled probabilities by taking argmax across class dimension
    downsampled_segmentation = torch.argmax(interpolated, dim=1).squeeze(0)  # Remove batch dimension
    
    # Convert back to NumPy
    return downsampled_segmentation.numpy()

# Custom Dataset for loading nifti files
class NiftiDataset(Dataset):
    def __init__(self, input_dir, output_dir, filenames):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        input_filename = os.path.join(self.input_dir, self.filenames[idx])
        output_filename = os.path.join(self.output_dir, f"{self.filenames[idx][:3]}_att_map_SimSET.nii")

        input_data = nib.load(input_filename).get_fdata(dtype=np.float32)
        # Crop input data
        input_data = input_data[77:173, :, :]  # Crop to 96 in the first dimension 

        # Pad input data with zeros
        input_data = np.pad(input_data, ((0, 0), (3, 3), (0, 1)), mode='constant', constant_values=0)  # Pad second dimension to 224 and third dimension to 48 (7, 7) (8,9)

      
        
        # Save preprocessed input data
        preprocessed_input_filename = f"preprocessed_input/preprocessed_{self.filenames[idx]}_stir_sinogram.nii"
        preprocessed_input_nifti = nib.Nifti1Image(input_data, affine=np.eye(4))
 ##       nib.save(preprocessed_input_nifti, preprocessed_input_filename)      

       
        
#        input_data = input_data/200
        # Normalize input data to range [0, 1]
        input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))

        
        output_data = nib.load(output_filename).get_fdata(dtype=np.float32)
        
        ### Attunuation maps preprocessing
        # Replace 0 with -1000
        output_data[output_data == 0] = -1000
        # Replace 4 with 30
        output_data[output_data == 4] = 30
        # Replace 3 with 600
        output_data[output_data == 3] = 600
        
        min_value = -1000
        max_value = 1000
        # Scaling the data
        output_data = (output_data - min_value) / (max_value - min_value)
        
        ### Attunuation maps preprocessing for classification
        # Replace 0 with class 0
        output_data[output_data == 0] = 0
        # Replace 0.515 with class 1
        output_data[output_data == 0.515] = 1
        # Replace 0.8 with class 2
        output_data[output_data == 0.8] = 2

        affine = np.eye(4)
        affine[0, :] *= -1  # Multiply srow_x by -1
        affine[1, :] *= -1  # Multiply srow_y by -1

        output_data = output_data[:, :, 53:228]
        output_data = downsample_with_f_interpolate_fixed_sizes(output_data) #torch.nn.functional.interpolate(torch.from_numpy(output_data).unsqueeze(0).unsqueeze(0).float(), size=(128, 128, 48), mode='nearest-exact').squeeze(0).squeeze(0).numpy()



        
        preprocessed_output_filename = f"preprocessed_target/{self.filenames[idx][:-4]}_preprocessed.nii"
        preprocessed_output_nifti = nib.Nifti1Image(output_data.astype(np.uint8), affine=affine) #np.eye(4)
        nib.save(preprocessed_output_nifti, preprocessed_output_filename)             
        

        input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        output_data = torch.tensor(output_data, dtype=torch.uint8).unsqueeze(0)

        filename_only = os.path.splitext(os.path.basename(self.filenames[idx]))[0]

        # Permute input_data from [C, H, W, D] to [C, D, H, W]
        input_data = input_data.permute(0, 3, 2, 1)  # [C, H=224, W=96, D=64] -> [C, D=64, H=224, W=96]

        # Permute output_data in the same way as input_data
        output_data = output_data.permute(0, 3, 2, 1)  # [C, H=256, W=256, D=256] -> [C, D=256, H=256, W=256]


        return input_data, output_data, filename_only




train_input_dir = r"/Final_Dataset_1/Training_Sino"
train_output_dir = r"/Final_Dataset_1/Training_Att"

validation_input_dir = r"/Final_Dataset_1/Validation_Sino"
validation_output_dir = r"/Final_Dataset_1/Validation_Att"

hole_input_dir = r"/Final_Dataset_1/Hole_Sino"
hole_output_dir = r"/Final_Dataset_1/Hole_Att"


# Create dataset and dataloader
train_dataset = NiftiDataset(train_input_dir, train_output_dir, sorted(os.listdir(train_input_dir)))
val_dataset = NiftiDataset(validation_input_dir, validation_output_dir, sorted(os.listdir(validation_input_dir)))



# Create the augmented dataset with Gaussian noise
noise_transform_10 = AddRandomNoise(mean=0.0, std=0.1)
train_dataset_augmented_10 = AugmentedDataset(train_dataset, transform=noise_transform_10)

# Create the augmented dataset with Gaussian noise
noise_transform_05 = AddRandomNoise(mean=0.0, std=0.05)
train_dataset_augmented_05 = AugmentedDataset(train_dataset, transform=noise_transform_05)


# Combine original and augmented datasets
combined_train_dataset = ConcatDataset([train_dataset]) #, train_dataset_augmented_10 train_dataset_augmented_05






train_loader = DataLoader(combined_train_dataset, batch_size=batch_size_val, shuffle=True,num_workers=workers) #train_dataset
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=workers)

# Assuming train_loader is your DataLoader
for i, (input_data, output_data, filename) in enumerate(train_loader):
    # Print shapes and filename for reference
    print(f"Sample {i + 1}:")
    print(f"Input Data Shape: {input_data.shape}")  # Expected: [B, C, D, H, W]
    print(f"Output Data Shape: {output_data.shape}")  # Expected: [B, C, D, H, W]
    print(f"Filename: {filename[0]}")  # Print the filename (assume batch_size=1 for simplicity)

    # Assuming data is in [B, C, D, H, W] format
    # Extract the first batch and first channel
    input_sample = input_data[0, 0]  # First batch, first channel
    output_sample = output_data[0, 0]  # First batch, first channel

    # Select a middle slice along the Depth (D) axis
    depth_index = input_sample.shape[0] // 2  # Middle slice index
    input_slice = input_sample[depth_index, :, :].cpu().numpy()  # Convert to numpy for visualization
    output_slice = output_sample[depth_index, :, :].cpu().numpy()

    # Plot and visualize the slices side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Input slice
    axs[0].imshow(input_slice, cmap="hot")
    axs[0].set_title(f"Input Slice - {filename[0]}")
    axs[0].set_xlabel(f"Width (pixels): {input_slice.shape[1]}")
    axs[0].set_ylabel(f"Height (pixels): {input_slice.shape[0]}")
    axs[0].axis("on")

    # Output slice
    axs[1].imshow(output_slice, cmap="hot")
    axs[1].set_title(f"Output Slice - {filename[0]}")
    axs[1].set_xlabel(f"Width (pixels): {output_slice.shape[1]}")
    axs[1].set_ylabel(f"Height (pixels): {output_slice.shape[0]}")
    axs[1].axis("on")

    # Save the visualization to a PNG file
    plt.tight_layout()
    plt.savefig(f"visualization_{filename[0].replace('.nii.gz', '')}_hot.png")
    plt.show()

    # Optional: Break after visualizing the first batch
    #hello = input("PAUSED")
    break

# Model, criterion, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet3D(in_channels=1, out_channels=3, dropout_prob_enc=dropout_val_enc, dropout_prob_dec=dropout_val_dec, dropout_prob_bn=dropout_val_bn)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
    model = torch.nn.DataParallel(model,device_ids=[0,1,2])

model.to(device)

# Function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the number of parameters in the model
print(f'Number of parameters in the model: {count_parameters(model):,}')

# Print model summary
input_shape = (48, 216, 96) #(96, 224, 64)
summary(model, input_size=(1, *input_shape))

# Create a dummy input tensor
x = torch.randn(1, 1, *input_shape).to(device)

# Forward pass through the model
y = model(x)

# Visualize the model architecture
#make_dot(y, params=dict(model.named_parameters())).render("unet3d.png", format="png")

#criterion = nn.PoissonNLLLoss()
criterion = CustomCrossEntropyLoss() #CustomMSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr_rate, weight_decay=wd_val)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,      # Reduce LR by half
    patience=3,      # Wait 5 epochs before reducing LR
    threshold=1e-4,  # Significant improvement threshold
    min_lr=1e-7,     # Minimum learning rate
    verbose=True
)


num_epochs = epochs_val

train_losses = []
val_losses = []

# Initialize GradScaler for mixed precision
scaler = GradScaler()

# Early stopping parameters
best_val_loss = float('inf')  # Initialize to a very high value
patience = 7                 # Number of epochs to wait before stopping
patience_counter = 0          # Counter for early stopping


# Global dictionary to store feature maps and counters
feature_maps = {}
layer_counter = {}  # Counter for each layer type to ensure unique numbering

# Global variables to control hook execution
current_epoch = 0
current_batch = 0

def hook_fn(module, input, output):
    global feature_maps, layer_counter, current_epoch, current_batch

    # Only store feature maps for the first batch of the first epoch
    if (current_epoch == 2) and current_batch == 0:
        layer_type = module.__class__.__name__

        # Increment counter for this layer type
        if layer_type not in layer_counter:
            layer_counter[layer_type] = 0
        layer_index = layer_counter[layer_type]
        layer_counter[layer_type] += 1

        # Generate a unique layer name
        layer_name = f"{layer_type}-{layer_index}"
        print(f"Storing feature map for layer: {layer_name}")  # Debugging
        feature_maps[layer_name] = output
    else:
        pass #print("Skipping feature map storage for subsequent batches.")  # Debugging


# Function to attach hooks to relevant layers
def attach_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d, nn.Upsample, nn.MaxPool3d, nn.Dropout3d, CustomSkipOperation)):
            layer_type = module.__class__.__name__
            print(f"Attaching hook to: {layer_type} ({name})")  # Debugging
            module.register_forward_hook(hook_fn)

def save_feature_maps(feature_maps, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    for layer_name, fmap in feature_maps.items():
        print(f"Processing feature map for layer: {layer_name}")  # Debugging

        # Get the first batch from the feature map (shape: [Batch, Channels, Depth, Height, Width])
        fmap = fmap[0].detach().cpu()  # Extract the first batch

        # Select the middle channel
        middle_channel_index = fmap.shape[0] // 2
        channel_map = fmap[middle_channel_index]  # [D, H, W]

        # Take the middle slice along the depth (D) axis
        middle_slice_d = channel_map[channel_map.shape[0] // 2, :, :].numpy()  # [H, W]

        # Save the image with the layer name and channel index
        filename = os.path.join(output_dir, f"{layer_name}_middle_channel_HxW.png")
        print(f"Saving feature map to: {filename}")  # Debugging
        plt.figure(figsize=(10, 10))
        plt.imshow(middle_slice_d, cmap="viridis", aspect="auto")
        plt.xlabel("Width (W)")
        plt.ylabel("Height (H)")
        plt.title(f"{layer_name} (Middle Channel: H x W)")
        plt.colorbar(label="Activation")
        plt.savefig(filename)
        plt.close()



# Load the saved best model weights
model.load_state_dict(torch.load(rf"{model_arch_name}-best.pth"))
model = model.to(device)


train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=workers)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers=workers)

# hole test
hole_dataset = NiftiDataset(hole_input_dir, hole_output_dir, sorted(os.listdir(hole_input_dir)))
hole_loader = DataLoader(hole_dataset, batch_size=1, shuffle=False,num_workers=workers)


# final test
test_input_dir = r"/Sino_Dataset_Test/Test_Sino"
test_output_dir = r"/Sino_Dataset_Test/Test_Att"

test_dataset = NiftiDataset(test_input_dir, test_output_dir, sorted(os.listdir(test_input_dir)))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=workers)


model.eval()
with torch.no_grad():
    # Save training results
    for i, (X_batch, y_batch, file_name) in enumerate(train_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        
        # Convert to single channel by taking the argmax (i.e., predicted class)
        outputs = torch.argmax(outputs, dim=1, keepdim=False)
        
        #swap back to original orientation
        outputs = outputs.permute(0, 3, 2, 1)

        # At this point, outputs should be [batch_size, depth, height, width]
        # Assuming batch_size=1, squeeze to remove batch dimension
        outputs = outputs.squeeze(0)
        
        # Convert outputs to int32 before saving
        outputs = outputs.cpu().numpy().astype(np.uint8)
        
        # Ensure the output has the correct shape (256, 256, 175)
        assert outputs.shape == (128, 128, 48), f"Unexpected output shape: {outputs.shape}"

        affine = np.eye(4)
        affine[0, :] *= -1  # Multiply srow_x by -1
        affine[1, :] *= -1  # Multiply srow_y by -1
        
        # Save the output as NIfTI file
        # removed _{model_arch_name}
        #output_filename = f"results_{model_arch_name}/train/patient_{i}_output.nii"
        output_filename = f"results/train/{file_name[0]}_output.nii"
        output_nifti = nib.Nifti1Image(outputs, affine=affine) #np.eye(4)
        nib.save(output_nifti, output_filename)

    # Save validation results
    for i, (X_batch, y_batch, file_name) in enumerate(val_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        
        # Convert to single channel by taking the argmax (i.e., predicted class)
        outputs = torch.argmax(outputs, dim=1, keepdim=False)

        #swap back to original orientation
        outputs = outputs.permute(0, 3, 2, 1)
        
        # At this point, outputs should be [batch_size, depth, height, width]
        # Assuming batch_size=1, squeeze to remove batch dimension
        outputs = outputs.squeeze(0)
        
        # Convert outputs to int32 before saving
        outputs = outputs.cpu().numpy().astype(np.uint8)
        
        # Ensure the output has the correct shape (256, 256, 175)
        assert outputs.shape == (128, 128, 48), f"Unexpected output shape: {outputs.shape}"

        affine = np.eye(4)
        affine[0, :] *= -1  # Multiply srow_x by -1
        affine[1, :] *= -1  # Multiply srow_y by -1  
        
        # Save the output as NIfTI file
        # removed _{model_arch_name}
        #output_filename = f"results_{model_arch_name}/val/patient_{i}_output.nii"
        output_filename = f"results/val/{file_name[0]}_output.nii"
        output_nifti = nib.Nifti1Image(outputs, affine=affine) #np.eye(4)
        nib.save(output_nifti, output_filename)

    # Save hole results
    for i, (X_batch, y_batch, file_name) in enumerate(hole_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        
        # Convert to single channel by taking the argmax (i.e., predicted class)
        outputs = torch.argmax(outputs, dim=1, keepdim=False)

        #swap back to original orientation
        outputs = outputs.permute(0, 3, 2, 1)
        
        # At this point, outputs should be [batch_size, depth, height, width]
        # Assuming batch_size=1, squeeze to remove batch dimension
        outputs = outputs.squeeze(0)
        
        # Convert outputs to int32 before saving
        outputs = outputs.cpu().numpy().astype(np.uint8)
        
        # Ensure the output has the correct shape (256, 256, 175)
        assert outputs.shape == (128, 128, 48), f"Unexpected output shape: {outputs.shape}"

        affine = np.eye(4)
        affine[0, :] *= -1  # Multiply srow_x by -1
        affine[1, :] *= -1  # Multiply srow_y by -1  
        
        # Save the output as NIfTI file
        # removed _{model_arch_name}
        #output_filename = f"results_{model_arch_name}/val/patient_{i}_output.nii"
        output_filename = f"results/hole/{file_name[0]}_output.nii"
        output_nifti = nib.Nifti1Image(outputs, affine=affine) #np.eye(4)
        nib.save(output_nifti, output_filename)   

    # Save test results
    for i, (X_batch, y_batch, file_name) in enumerate(test_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        
        # Convert to single channel by taking the argmax (i.e., predicted class)
        outputs = torch.argmax(outputs, dim=1, keepdim=False)

        #swap back to original orientation
        outputs = outputs.permute(0, 3, 2, 1)
        
        # At this point, outputs should be [batch_size, depth, height, width]
        # Assuming batch_size=1, squeeze to remove batch dimension
        outputs = outputs.squeeze(0)
        
        # Convert outputs to int32 before saving
        outputs = outputs.cpu().numpy().astype(np.uint8)
        
        # Ensure the output has the correct shape (256, 256, 175)
        assert outputs.shape == (128, 128, 48), f"Unexpected output shape: {outputs.shape}"

        affine = np.eye(4)
        affine[0, :] *= -1  # Multiply srow_x by -1
        affine[1, :] *= -1  # Multiply srow_y by -1  
        
        # Save the output as NIfTI file
        # removed _{model_arch_name}
        #output_filename = f"results_{model_arch_name}/val/patient_{i}_output.nii"
        output_filename = f"results/test/{file_name[0]}_output.nii"
        output_nifti = nib.Nifti1Image(outputs, affine=affine) #np.eye(4)
        nib.save(output_nifti, output_filename)        



# Plot training and validation loss
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig(rf'{model_arch_name}.png')  # Save the plot as an image file
#plt.show()
