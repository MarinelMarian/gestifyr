import os
import csv
import pandas as pd
from tools import clear_terminal 
import numpy as np
from videoProcessingTools import points_to_extract, get_angles

clear_terminal()
# Define input and output folders
input_folder = (
    "app/samples3/processed"  # Change to your folder containing CSV files
)
input_folder_raw = "app/samples3/raw"
output_folder = "app/samples3/processed_trimmed"  # Folder to save processed files

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)



# Process each CSV file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):  # Process only CSV files

        input_file_path = os.path.join(input_folder, filename)
        input_file_path_raw = os.path.join(input_folder_raw, filename)
        print(f"Analysing file {input_file_path}")

        output_file_path = os.path.join(output_folder, filename)  # Save with same name
        processedFeatures = pd.read_csv(input_file_path, skiprows=1, header=None)
        rawFeatures = pd.read_csv(input_file_path_raw, skiprows=1, header=None)

        extracted_data = []
        for pf, rf in zip(processedFeatures.values, rawFeatures.values):
            extracted_row = [pf[i] for i in points_to_extract]
            x, y = get_angles(rf)
            extracted_row.append(x)
            extracted_row.append(y)

            extracted_data.append(extracted_row.copy())

        # for row in reader:
        #     extracted_row = []
        #     for point_idx in points_to_extract:
        #         base_index = (point_idx -1)* 3  # Each point has x, y, z coordinates
        #         extracted_row.extend(row[base_index:base_index + 3])
        #     extracted_data.append(extracted_row)

        # Write extracted data to the output file
        with open(output_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            header = [
                "browOuterUpLeft",
                "browOuterUpRight",
                "jawOpen",
                "eyeBlinkLeft",
                "eyeBlinkRight",
                "smileLeft",
                "smileRight",
                "angleUpDown",
                "angleLeftRight",
            ]
            writer.writerow(header)

            # Write extracted data
            writer.writerows(extracted_data)

        print(f"Processed: {filename} → Saved to {output_file_path}")

print("Batch processing complete!")
