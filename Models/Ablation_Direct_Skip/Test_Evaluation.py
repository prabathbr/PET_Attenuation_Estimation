import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
from concurrent.futures import ThreadPoolExecutor

def generate_file_lists(ground_truth_dir, model_output_dir):
    """
    Generate lists of ground truth and model output file paths by matching filenames without suffixes.
    """
    ground_truth_files = []
    model_output_files = []

    model_output_filenames = os.listdir(model_output_dir)

    for filename in model_output_filenames:
        if filename.endswith("_output.nii"):
            model_output_files.append(os.path.join(model_output_dir, filename))
            ground_truth_file = filename.replace("_output.nii", "_preprocessed.nii")
            ground_truth_files.append(os.path.join(ground_truth_dir, ground_truth_file))

    return ground_truth_files, model_output_files

def calculate_dice(ground_truth_flat, model_output_flat, class_label):
    """
    Calculate Dice score for a specific class.
    """
    intersection = np.sum((ground_truth_flat == class_label) & (model_output_flat == class_label))
    size_gt = np.sum(ground_truth_flat == class_label)
    size_pred = np.sum(model_output_flat == class_label)
    
    # Avoid division by zero
    if size_gt + size_pred == 0:
        return np.nan
    
    return 2 * intersection / (size_gt + size_pred)


def calculate_accuracy(ground_truth_flat, model_output_flat, class_label):
    """
    Calculate accuracy for a specific class using accuracy_score.
    """
    # Mask to only calculate accuracy for the specific class
    mask = (ground_truth_flat == class_label) | (model_output_flat == class_label)
    if np.sum(mask) == 0:
        return np.nan
    return accuracy_score(ground_truth_flat[mask], model_output_flat[mask])

def calculate_precision(ground_truth_flat, model_output_flat, num_classes):
    """
    Calculate precision for all classes.
    """
    return precision_score(ground_truth_flat, model_output_flat, average=None, labels=np.arange(num_classes))

def calculate_recall(ground_truth_flat, model_output_flat, num_classes):
    """
    Calculate recall for all classes.
    """
    return recall_score(ground_truth_flat, model_output_flat, average=None, labels=np.arange(num_classes))

def calculate_f1(ground_truth_flat, model_output_flat, num_classes):
    """
    Calculate F1-score for all classes.
    """
    return f1_score(ground_truth_flat, model_output_flat, average=None, labels=np.arange(num_classes))

def process_single_file(gt_file, model_file, num_classes):
    """
    Process a single pair of ground truth and model output files to compute metrics.
    """
    ground_truth_img = nib.load(gt_file)
    model_output_img = nib.load(model_file)

    # Ensure data is in uint8 format
    ground_truth = ground_truth_img.get_fdata().astype(np.uint8)
    model_output = model_output_img.get_fdata().astype(np.uint8)

    ground_truth_flat = ground_truth.flatten()
    model_output_flat = model_output.flatten()

    accuracies = [calculate_accuracy(ground_truth_flat, model_output_flat, class_label) for class_label in range(num_classes)]
    precision = calculate_precision(ground_truth_flat, model_output_flat, num_classes)
    recall = calculate_recall(ground_truth_flat, model_output_flat, num_classes)
    f1 = calculate_f1(ground_truth_flat, model_output_flat, num_classes)
    dice_scores = [calculate_dice(ground_truth_flat, model_output_flat, class_label) for class_label in range(num_classes)]


    filename = os.path.basename(model_file).replace("_output.nii", "")
    image_metrics = {"Image": filename}

    for class_label in range(num_classes):
        image_metrics[f"Accuracy_Class_{class_label}"] = accuracies[class_label]
        image_metrics[f"Precision_Class_{class_label}"] = precision[class_label]
        image_metrics[f"Recall_Class_{class_label}"] = recall[class_label]
        image_metrics[f"F1_Class_{class_label}"] = f1[class_label]
        image_metrics[f"Dice_Class_{class_label}"] = dice_scores[class_label]

    return image_metrics

def calculate_metrics_for_nifti_images_multithreaded(ground_truth_files, model_output_files, output_csv_path, num_classes=3, max_workers=30):
    """
    Calculate per-class accuracy, precision, recall, and F1-score for each validation image in parallel and save to CSV.
    """
    results = []

    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_file, gt_file, model_file, num_classes)
                   for gt_file, model_file in zip(ground_truth_files, model_output_files)]
        for future in futures:
            results.append(future.result())

    # Convert results to DataFrame and sort by "Image" column
    metrics_table = pd.DataFrame(results)
    metrics_table = metrics_table.sort_values(by="Image").reset_index(drop=True)

    # Save the DataFrame as a CSV file
    metrics_table.to_csv(output_csv_path, index=False)
    print(f"Metrics saved to {output_csv_path}")

########## Run

# Define the directories where your files are located
ground_truth_dir = r"/preprocessed_target"
model_output_dir = r"/results/test"

# Call generate_file_lists to get the list of ground truth and model output file paths
ground_truth_files, model_output_files = generate_file_lists(ground_truth_dir, model_output_dir)

# Specify the output CSV file path
output_csv_path = "validation_image_metrics_test.csv"

# Define the number of worker threads to use
max_workers = 30  # Adjust this value based on your system's capability; default is 30

# Run with multithreading
calculate_metrics_for_nifti_images_multithreaded(ground_truth_files, model_output_files, output_csv_path, num_classes=3, max_workers=max_workers)
print(f"Metrics have been calculated and saved to {output_csv_path}")
