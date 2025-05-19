import os
import shutil
import matlab.engine

# Start MATLAB engine
eng = matlab.engine.start_matlab()

# Define template file for normalization
template_file = r"K:\spm12\spm12\toolbox\OldNorm\PET.nii"

def organize_nifti_files(source_folder, dest_folder):
    """Organizes .nii files into individual folders."""
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for file in os.listdir(source_folder):
        if file.endswith(".nii"):  
            file_name = os.path.splitext(file)[0]
            new_folder_path = os.path.join(dest_folder, file_name)

            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)

            shutil.copy2(os.path.join(source_folder, file), os.path.join(new_folder_path, file))

    print(f"Files from {source_folder} have been organized into {dest_folder}.")

def process_nifti_files(dest_folder):
    """Processes .nii files by setting origin, copying files, and running normalization."""
    for folder in os.listdir(dest_folder):
        folder_path = os.path.join(dest_folder, folder)

        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".nii"):
                    original_file_path = os.path.join(folder_path, file)

                    # Set origin using MATLAB function
                    eng.B_nii_setOriginCOM(original_file_path)
                    print("SET ORIGIN COMPLETED")

                    # Define paths for SPM folder and 's' file
                    spm_folder_path = os.path.join(folder_path, "SPM")
                    spm_file_path = os.path.join(spm_folder_path, "s" + file)

                    if os.path.exists(spm_folder_path):
                        for spm_file in os.listdir(spm_folder_path):
                            if spm_file.startswith("s") and spm_file.endswith(".nii"):
                                spm_file_path = os.path.join(spm_folder_path, spm_file)
                                break

                    # Create "s" file inside SPM folder
                    shutil.copy(os.path.join(spm_folder_path, file), os.path.join(spm_folder_path, "s" + file))

                    # Create "os" file in the destination folder
                    os_file_name = "os" + file
                    os_file_path = os.path.join(folder_path, os_file_name)

                    if os.path.exists(spm_file_path):
                        shutil.copy2(spm_file_path, os_file_path)
                        print(f"Copied {spm_file_path} to {os_file_path}")
                    else:
                        print(f"SPM file not found for {original_file_path}")

                    # Run mni_normalizer on "os" file and produce "wos" file
                    eng.mni_normalizer(os_file_path, template_file)
                    print(f"Normalization complete for {os_file_path}")

def move_final_files(process_folder, final_dest_folder):
    """Moves all files with 'wos' suffix to final MNI directory, removing 'wos' suffix."""
    if not os.path.exists(final_dest_folder):
        os.makedirs(final_dest_folder)

    for folder in os.listdir(process_folder):
        folder_path = os.path.join(process_folder, folder)

        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.startswith("wos") and file.endswith(".nii"):
                    original_file_path = os.path.join(folder_path, file)

                    # Remove "wos" from filename
                    new_file_name = file.replace("wos", "")
                    final_file_path = os.path.join(final_dest_folder, new_file_name)

                    shutil.copy2(original_file_path, final_file_path)
                    print(f"Moved {original_file_path} to {final_file_path}")

# Define source, process, and final MNI folders
source_dest_pairs = [
    (r"L:\Results_Compare\Ground_NIFTI", r"L:\Results_Compare\Ground_NIFTI_Process"),
    (r"L:\Results_Compare\Model_NIFTI", r"L:\Results_Compare\Model_NIFTI_Process")
]

final_dest_pairs = [
    (r"L:\Results_Compare\Ground_NIFTI_Process", r"L:\Results_Compare\Ground_NIFTI_MNI"),
    (r"L:\Results_Compare\Model_NIFTI_Process", r"L:\Results_Compare\Model_NIFTI_MNI")
]

# Organize files into folders
for source, dest in source_dest_pairs:
    organize_nifti_files(source, dest)

print("All files have been organized successfully.")

# Process files (Set Origin, Copy & Normalize)
for _, dest in source_dest_pairs:
    process_nifti_files(dest)

print("All MATLAB operations completed successfully.")

# Move final "wos" files to MNI directories
for process_folder, final_dest_folder in final_dest_pairs:
    move_final_files(process_folder, final_dest_folder)

print("All final files moved successfully.")
