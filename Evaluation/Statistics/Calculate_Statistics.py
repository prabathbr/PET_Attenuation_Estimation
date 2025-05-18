import os
import pandas as pd
from scipy.stats import wilcoxon

# Define the directory containing the CSV files
folder_path = r"csv_results"

# Define the mapping of filenames to LaTeX table columns
file_mapping = {
    "baseline.csv": "Baseline",
    "ourmodel.csv": "Our Model",
    "directskip.csv": "Direct Skip",
    "noskip.csv": "No Skip"
}

# Mapping class numbers to their corresponding names
class_mapping = {
    "Class_0": "Air",
    "Class_1": "Tissue",
    "Class_2": "Bones"
}

# Initialize a dictionary to store mean, std, and p-values
metrics = {metric: {class_name: {} for class_name in class_mapping.values()} for metric in ["Accuracy", "Precision", "Recall", "Dice"]}

# Read and process each file
dataframes = {}

for file_name, column_name in file_mapping.items():
    file_path = os.path.join(folder_path, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        dataframes[column_name] = df  # Store dataframe for Wilcoxon test later

        # Compute mean and standard deviation for each column (excluding 'Image' column)
        means = df.mean()
        stds = df.std()

        # Store results in a structured way
        for metric in metrics.keys():
            for class_id, class_name in class_mapping.items():
                column_name_formatted = f"{metric}_Class_{class_id[-1]}"  # Correct column format

                if column_name_formatted in df.columns:  # Check if column exists
                    mean_value = means[column_name_formatted]
                    std_value = stds[column_name_formatted]

                    # Ensure exactly 3 decimal places
                    mean_str = "{:.3f}".format(mean_value)
                    std_str = "{:.3f}".format(std_value)

                    # Only enclose \pm inside $...$
                    metrics[metric][class_name][column_name] = f"{mean_str} $\\pm$ {std_str}"
                else:
                    metrics[metric][class_name][column_name] = "-"  # Handle missing columns

# Compute Wilcoxon signed-rank test for p-values
for metric in metrics.keys():
    for class_id, class_name in class_mapping.items():
        column_name_formatted = f"{metric}_Class_{class_id[-1]}"  # Correct column format
        
        if "Baseline" in dataframes and "Our Model" in dataframes:
            baseline_data = dataframes["Baseline"].get(column_name_formatted)
            ourmodel_data = dataframes["Our Model"].get(column_name_formatted)

            if baseline_data is not None and ourmodel_data is not None:
                try:
                    stat, p_value = wilcoxon(baseline_data.dropna(), ourmodel_data.dropna())  # Drop NaNs

                    # Determine stat rounding:
                    if stat % 1 == 0:  # If stat is an integer, round to 0 decimal places
                        stat_str = "{:.0f}".format(stat)
                    else:  # Otherwise, round to 1 decimal place
                        stat_str = "{:.1f}".format(stat)

                    # If p-value is too small (< 0.001), report as "$<$ 0.001"
                    if p_value < 0.001:
                        p_value_str = "$<$ 0.001"
                    else:
                        p_value_str = "{:.3f}".format(p_value)

                    metrics[metric][class_name]["p-value"] = f"{stat_str} ({p_value_str})"
                except ValueError:
                    metrics[metric][class_name]["p-value"] = "-"
            else:
                metrics[metric][class_name]["p-value"] = "-"
        else:
            metrics[metric][class_name]["p-value"] = "-"

# Generate LaTeX table
latex_table = """\\begin{table*}[ht]
    \\centering
    \\begin{tabular}{lccc|cc}
        \\toprule
        & \\multicolumn{3}{c|}{(A)} & \\multicolumn{2}{c}{(B)} \\\\
        \\midrule
        Metric       & Baseline & Our Model & p-value & Direct Skip & No Skip \\\\
        \\midrule
"""

for metric, classes in metrics.items():
    latex_table += f"        \\textbf{{{metric}}} &         &         &         &         &         \\\\\n"
    for class_name, values in classes.items():
        row_values = [
            values.get('Baseline', "-"),
            values.get('Our Model', "-"),
            values.get('p-value', "-"),  # Insert computed Wilcoxon stat and p-value
            values.get('Direct Skip', "-"),
            values.get('No Skip', "-")
        ]
        latex_table += f"        \\quad {class_name} & {' & '.join(row_values)} \\\\\n"


latex_table += """        \\bottomrule
    \\end{tabular}
    \\caption{Comparison of evaluation metrics across models for hold-out test dataset.}
    \\label{tab:metrics}
\\end{table*}"""

# Save LaTeX table to file
latex_file_path = os.path.join(folder_path, "summary_table.tex")
with open(latex_file_path, "w") as f:
    f.write(latex_table)

print(f"LaTeX table saved to: {latex_file_path}")
