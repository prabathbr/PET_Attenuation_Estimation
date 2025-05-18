import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import matplotlib.pyplot as plt
from torchsummary import summary
from torchviz import make_dot
from torch.cuda.amp import autocast, GradScaler

import wandb

dropout_val_enc = 0.10
dropout_val_dec = 0.10
dropout_val_bn = 0.10

wd_val = 0.01
lr_rate = 0.00001
epochs_val = 100
batch_size_val = 8
model_arch_name = f"NAC_UNET-3D_{epochs_val}-{lr_rate}-{wd_val}"

workers = 24

wandb.init(
    # set the wandb project where this run will be logged
    project="pseudo-uMap-NAC",

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
        

        # Apply weights to each element in the batch before taking the mean
        loss = F.cross_entropy(predictions, targets, reduction='none')
        weighted_loss = weights * loss
        
        return torch.mean(weighted_loss)        


# Define the 3D U-Net Model
class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)




        
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob_enc=0.5,dropout_prob_dec=0.5,dropout_prob_bn=0.5):
        super(UNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_prob_enc = dropout_prob_enc
        self.dropout_prob_dec = dropout_prob_dec
        self.dropout_prob_bn = dropout_prob_bn

        # Encoder
        self.encoder1 = DoubleConv(in_channels, 32, 64)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout3d(self.dropout_prob_enc)  # Dropout after first pooling

        self.encoder2 = DoubleConv(64, 64, 128)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout3d(self.dropout_prob_enc)  # Dropout after second pooling

        self.encoder3 = DoubleConv(128, 128, 256)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout3d(self.dropout_prob_enc)  # Dropout after third pooling

        # Bottleneck
        self.bottleneck = DoubleConv(256, 256, 256)
        self.dropout_bottleneck = nn.Dropout3d(self.dropout_prob_bn)  # Dropout in bottleneck

        # Decoder
        self.upconv3 = nn.ConvTranspose3d(256, 512, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512 + 256, 256, 256)
        self.dropout_dec3 = nn.Dropout3d(self.dropout_prob_dec)  # Dropout after third decoder

        self.upconv2 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256 + 128, 128, 128)
        self.dropout_dec2 = nn.Dropout3d(self.dropout_prob_dec)  # Dropout after second decoder

        self.upconv1 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128 + 64, 64, 64)
        self.dropout_dec1 = nn.Dropout3d(self.dropout_prob_dec)  # Dropout after first decoder

        # Output layer
        #self.output_layer_0 = nn.Conv3d(64, 32, stride=(3, 7, 2), kernel_size=1)
        self.output_layer_0 = nn.Conv3d(64, 3, stride=(1,1,1), kernel_size=1)
       

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc2 = self.dropout1(enc2)  # Apply dropout after first pooling

        enc3 = self.encoder3(self.pool2(enc2))
        enc3 = self.dropout2(enc3)  # Apply dropout after second pooling

        bottleneck = self.bottleneck(self.pool3(enc3))
        bottleneck = self.dropout3(bottleneck)  # Apply dropout after third pooling

        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec3 = self.dropout_dec3(dec3)  # Apply dropout after third decoder

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec2 = self.dropout_dec2(dec2)  # Apply dropout after second decoder

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.dropout_dec1(dec1)  # Apply dropout after first decoder

        out0 = self.output_layer_0(dec1)

        return out0  #self.output_layer_1(out0)

    def pad_to_match(self, enc, dec):
        enc_d, enc_h, enc_w = enc.shape[2], enc.shape[3], enc.shape[4]
        dec_d, dec_h, dec_w = dec.shape[2], dec.shape[3], dec.shape[4]

        pad_d = enc_d - dec_d
        pad_h = enc_h - dec_h
        pad_w = enc_w - dec_w

        dec = F.pad(dec, (0, pad_w, 0, pad_h, 0, pad_d))

        return dec

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
    
    def resize_and_crop(self, tensor):
        """
        Resizes the tensor in HxW dimensions and crops it back to the original HxW dimensions.

        Args:
            tensor (torch.Tensor): Input tensor to be resized and cropped with shape [C, D, H, W].

        Returns:
            torch.Tensor: Resized and cropped tensor with shape [C, D, H, W].
        """
        scale_factor_h_w = 1.15
        output_shape = tensor.shape  # [C, D, H, W]

        # Add batch dimension for F.interpolate
        tensor = tensor.unsqueeze(0)  # Shape becomes [1, C, D, H, W]

        # Resize in HxW dimensions
        resized_tensor = F.interpolate(
            tensor,
            size=(output_shape[1],  # Depth remains the same
                  int(output_shape[2] * scale_factor_h_w),  # Height scaled
                  int(output_shape[3] * scale_factor_h_w)),  # Width scaled
            mode='trilinear',
            align_corners=True
        )

        # Remove batch dimension
        resized_tensor = resized_tensor.squeeze(0)  # Shape becomes [C, D, H, W]

        # Crop to match original HxW dimensions
        original_h, original_w = output_shape[2], output_shape[3]
        new_h, new_w = resized_tensor.shape[2], resized_tensor.shape[3]

        start_h = (new_h - original_h) // 2
        start_w = (new_w - original_w) // 2
        end_h = start_h + original_h
        end_w = start_w + original_w

        cropped_tensor = resized_tensor[:, :, start_h:end_h, start_w:end_w]  # Crop H and W

        return cropped_tensor    

    def __getitem__(self, idx):
        input_filename = os.path.join(self.input_dir, self.filenames[idx])
        output_filename = os.path.join(self.output_dir, f"{self.filenames[idx][:3]}_att_map_SimSET.nii")

        input_data = nib.load(input_filename).get_fdata(dtype=np.float32)
        # Crop input data
        #input_data = input_data[77:173, :, :]  # Crop to 96 in the first dimension 

        # Create a circular mask
        x, y = np.meshgrid(np.arange(128), np.arange(128))
        center = (64, 64)
        radius = 60
        mask = ((x - center[0])**2 + (y - center[1])**2) <= radius**2

        # Apply the mask slice-by-slice and overwrite input_data
        for z in range(input_data.shape[2]):
                input_data[..., z] *= mask

        # Pad input data with zeros
        input_data = np.pad(input_data, ((0, 0), (0, 0), (0, 1)), mode='constant', constant_values=0)  # Pad second dimension to 224 and third dimension to 48

      
        
        # Save preprocessed input data
        preprocessed_input_filename = f"preprocessed_input/preprocessed_{self.filenames[idx]}_stir_sinogram.nii"
        preprocessed_input_nifti = nib.Nifti1Image(input_data, affine=np.eye(4))
       ## nib.save(preprocessed_input_nifti, preprocessed_input_filename)      

       
        
#        input_data = input_data/200
        # Normalize input data to range [0, 1]
        input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))

        
        output_data = nib.load(output_filename).get_fdata(dtype=np.float32)
        
        ### Attenuation maps preprocessing
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

        # Resize and crop the input data
        input_data = self.resize_and_crop(input_data)    

        input_data = input_data.flip(dims=[-1])    

        # Permute output_data in the same way as input_data
        output_data = output_data.permute(0, 3, 2, 1)  # [C, H=256, W=256, D=256] -> [C, D=256, H=256, W=256]
        #print({v.item(): c.item() for v, c in zip(*torch.unique(output_data, return_counts=True))})


        return input_data, output_data, filename_only


train_input_dir = r"/NAC_Dataset/Training_NAC"
train_output_dir = r"/NAC_Dataset/Training_Att"

validation_input_dir = r"/NAC_Dataset/Validation_NAC"
validation_output_dir = r"/NAC_Dataset/Validation_Att"



# List all filenames
#all_filenames = [f"{i}" for i in range(314, 339) if i not in [314,317,320,323,326,335, 336]]

# Set the split ratio
#split_ratio = 0.75  # 75% training, 25% validation

# Calculate the split index
#split_index = int(len(all_filenames) * split_ratio)

# Split the filenames
#train_filenames = [327,328,329,330,331,332,337] #all_filenames[:split_index]
#val_filenames = [322,323] #all_filenames[split_index:]

# Create dataset and dataloader
train_dataset = NiftiDataset(train_input_dir, train_output_dir, sorted(os.listdir(train_input_dir)))
val_dataset = NiftiDataset(validation_input_dir, validation_output_dir, sorted(os.listdir(validation_input_dir)))

train_loader = DataLoader(train_dataset, batch_size=batch_size_val, shuffle=True,num_workers=workers)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False,num_workers=workers)


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
    depth_index = 10 #input_sample.shape[0] // 2  # Middle slice index
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
model = UNet3D(in_channels=1, out_channels=3,  dropout_prob_enc=dropout_val_enc, dropout_prob_dec=dropout_val_dec, dropout_prob_bn=dropout_val_bn)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
    model = torch.nn.DataParallel(model,device_ids=[0])

model.to(device)

# Function to count the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the number of parameters in the model
print(f'Number of parameters in the model: {count_parameters(model):,}')

# Print model summary
input_shape = (48, 128, 128)
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


for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (X_batch, y_batch, file_name) in enumerate(train_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        # Use autocast for mixed precision
        with autocast():
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

        #outputs = model(X_batch)
        #loss = criterion(outputs, y_batch)
        #loss.backward()
        #optimizer.step()

        # Scale the loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()  # Update the scaler for the next iteration

        running_loss += loss.item() * X_batch.size(0)

        # Print training batch details
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Batch [{batch_idx+1}/{len(train_loader)}], Batch Loss: {loss.item():.4f}")


    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (X_batch, y_batch, file_name) in enumerate(val_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Use autocast for mixed precision
            with autocast():
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

            #outputs = model(X_batch)
            #loss = criterion(outputs, y_batch)


            val_loss += loss.item() * X_batch.size(0)

            # Print validation batch details
            print(f"Epoch [{epoch+1}/{num_epochs}], Val Batch [{batch_idx+1}/{len(val_loader)}], Batch Loss: {loss.item():.4f}")


    val_loss = val_loss / len(val_loader.dataset)
    val_losses.append(val_loss)

    scheduler.step(val_loss)    

    # Clear unused GPU memory at the end of each epoch
    torch.cuda.empty_cache()

    # print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    # wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss})
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Current learning rate: {current_lr:.2e}')
    wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss,"lr": current_lr})

# Add this at the end of your training loop, after calculating val_loss
    if val_loss < best_val_loss - 1e-4:  # Improvement threshold
        best_val_loss = val_loss         # Update the best validation loss
        patience_counter = 0             # Reset patience counter
        print(f"Validation loss improved to {val_loss:.4f}.")
        torch.save(model.module.state_dict(), rf"{model_arch_name}_best.pth")
    else:
        patience_counter += 1
        print(f"No improvement in validation loss for {patience_counter} epochs.")

        if patience_counter >= patience:
            print("Early stopping triggered. Stopping training.")
            break    

# After the training loop, save the final epoch results
wandb.finish()

# Save the model
torch.save(model.module.state_dict(), rf"{model_arch_name}.pth")

# Load the saved best model weights
model.module.load_state_dict(torch.load(rf"{model_arch_name}_best.pth"))
model = model.to(device)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,num_workers=workers)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers=workers)

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
        
        # Ensure the output has the correct shape (256, 256, 256)
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
        
        # Ensure the output has the correct shape (256, 256, 256)
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



# Plot training and validation loss
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig(rf'{model_arch_name}.png')  # Save the plot as an image file
#plt.show()
