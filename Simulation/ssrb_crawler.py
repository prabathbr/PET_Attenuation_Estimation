import os
import subprocess
import stir
import stirextra
import numpy as np
import nibabel as nib

def update_hs_file(hs_file_path):
    """
    Updates the line in the .hs file where "name of data file :=" is defined.
    
    Parameters:
    hs_file_path (str): The path to the .hs file.
    """
    try:
        with open(hs_file_path, 'r') as file:
            lines = file.readlines()
        
        # Modify the line starting with "name of data file :="
        with open(hs_file_path, 'w') as file:
            for line in lines:
                if line.startswith("name of data file :="):
                    file.write("name of data file := stir_sinogram.s\n")
                else:
                    file.write(line)
        
        print(f"Updated the .hs file: {hs_file_path}")
    except Exception as e:
        print(f"An error occurred while updating the .hs file: {e}")
        raise

def run_ssrb_command(ssrb_output_file, stir_input_file):
    """
    Runs the SSRB command with the given output and input files.

    Parameters:
    ssrb_output_file (str): The output filename for the SSRB command.
    stir_input_file (str): The input filename for the SSRB command.
    """
    try:
        # Construct the command
        command = ["SSRB", ssrb_output_file, stir_input_file, "23", "1", "1"]
        
        # Run the command
        subprocess.run(command, check=True)
        
        print(f"Command executed successfully: {' '.join(command)}")
        
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")
        raise

def convert_stir_nifti(input_filename):
    """
    Converts a STIR ProjData file to a NIfTI image and saves it with the same filename but with a .nii extension.
    
    Parameters:
    input_filename (str): The path to the input STIR ProjData file.
    """
    try:
        print(f"Handling:\n  {input_filename}")
        # Load the projection data from the file
        proj_data = stir.ProjData.read_from_file(input_filename)
        
        # Convert the ProjData to a NumPy array
        np_data = stirextra.to_numpy(proj_data)
        print(f"Original shape: {np_data.shape}")
        
        # Reshape the array to (249, 210, 576) from (576, 210, 249)
        reshaped_data = np.transpose(np_data, (2, 1, 0))
        print(f"Reshaped data shape: {reshaped_data.shape}")
        
        # Create an affine matrix for the NIfTI image
        affine = np.eye(4)
        affine[0, :] *= -1  # Multiply srow_x by -1
        affine[1, :] *= -1  # Multiply srow_y by -1
        
        # Create the NIfTI image
        nifti_img = nib.Nifti1Image(reshaped_data, affine=affine)
        
        # Generate output filename with .nii extension
        output_filename = input_filename.rsplit('.', 1)[0] + '.nii'
        
        # Save the NIfTI image
        nib.save(nifti_img, output_filename)
        print(f"NIfTI file saved as: {output_filename}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise ValueError("Failed to convert STIR ProjData to NIfTI format.")

def process_folders(source_base_path, destination_base_path, start_folder, end_folder, exclude_folders):
    """
    Crawls through specified folders, processes subfolders, runs SSRB command, converts to NIfTI, and saves the result.
    
    Parameters:
    source_base_path (str): The base directory containing the source folders.
    destination_base_path (str): The directory where the NIfTI files will be saved.
    start_folder (int): The starting folder number to process.
    end_folder (int): The ending folder number to process.
    exclude_folders (set): Set of folder numbers to exclude from processing.
    """
    # Ensure the destination folder exists
    os.makedirs(destination_base_path, exist_ok=True)

    # Iterate through the specified range of folders
    for folder_number in range(start_folder, end_folder + 1):
        if folder_number in exclude_folders:
            continue
        
        folder_path = os.path.join(source_base_path, str(folder_number))
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            print(f"Folder {folder_number} does not exist. Skipping.")
            continue
        
        # Process each subfolder under the current folder number
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder, "SimSET_Sim_ge_discovery_st", "OSEM3D")
            if not os.path.isdir(subfolder_path):
                continue
            
            hs_file = os.path.join(subfolder_path, "stir_sinogram.hs")
            if os.path.exists(hs_file):
                # Update the .hs file before proceeding
                update_hs_file(hs_file)
                
                # Create a corresponding destination folder
                result_subfolder_path = os.path.join(destination_base_path, str(folder_number), subfolder)
                os.makedirs(result_subfolder_path, exist_ok=True)
                
                # SSRB output file path
                ssrb_output_file = os.path.join(result_subfolder_path, f"{folder_number}_{subfolder}_SSRB_stir_sinogram.hs")
                stir_input_file = hs_file

                # Run SSRB command
                run_ssrb_command(ssrb_output_file, stir_input_file)
                
                # After the SSRB command completes, convert the result to NIfTI
                convert_stir_nifti(ssrb_output_file)
                
                print(f"Converted {ssrb_output_file}")
            else:
                print(f"Source file {hs_file} does not exist in {subfolder}. Skipping.")

    print("File conversion completed.")

def main():
    # Define the source base path and destination path
    source_base_path = r"dataset_gen"
    destination_base_path = r"dataset_processed"

    # Define the range of folders to iterate through
    start_folder = 100
    end_folder = 200
    exclude_folders = {300}

    process_folders(source_base_path, destination_base_path, start_folder, end_folder, exclude_folders)

if __name__ == '__main__':
    main()
