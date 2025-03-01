import os
import csv
import pandas as pd
import tools as t
from mediapipe_extract import getDistanteBetweenCornerEyes
import numpy as np
import cv2

t.clearTerminal()
# Define input and output folders
input_folder = "tudor/forTesting/processed"  # Change to your folder containing CSV files
input_folder_raw = "tudor/forTesting/raw"
output_folder = "tudor/forTesting/processed_trimmed"  # Folder to save processed files

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)


points_to_extract = [4,5, 25] # zero position indexs, browOuterUpLeft, browOuterUpRight, jawOpen
points_to_get_angles = [1, 61 , 291, 33, 263, 199]
img_w,img_h = 1980,1080

def getAngles(result_features):
    sublist = []
    for point_idx in points_to_get_angles:
        base_index = (point_idx -1)* 3  # Each point has x, y, z coordinates
        sublist.append(result_features[base_index:base_index + 3])
    face_3d = [ [int(e[0]*img_w), int(e[1]*img_h), e[2] ] for e in sublist] 
    face_3d = np.array(face_3d, dtype=np.float64)
    face_2d = np.array(face_3d[:, 0:2],  dtype=np.float64)
    focal_length = 1 * img_w

    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                            [0, focal_length, img_w / 2],
                            [0, 0, 1]])

    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    
                # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)

    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360
    return x,y
    


# Process each CSV file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".csv"):  # Process only CSV files

        input_file_path = os.path.join(input_folder, filename)
        input_file_path_raw = os.path.join(input_folder_raw, filename)
        print(f'Analysing file {input_file_path}')

        output_file_path = os.path.join(output_folder, filename)  # Save with same name
        processedFeatures = pd.read_csv(input_file_path, skiprows=1, header=None)   
        rawFeatures = pd.read_csv(input_file_path_raw, skiprows=1, header=None)   

        extracted_data = []
        for pf, rf in zip(processedFeatures.values, rawFeatures.values):
            extracted_row = [pf[i] for i in points_to_extract]
            extracted_row.append(getDistanteBetweenCornerEyes(np.array(rf)))
            x,y = getAngles(rf)
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
        with open(output_file_path, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            header = ["browOuterUpLeft", "browOuterUpRight", "jawOpen", "distanceBetweenCornerEyes", "verticalAngle", "horizontalAngle"]
            writer.writerow(header)

            # Write extracted data
            writer.writerows(extracted_data)

        print(f"Processed: {filename} → Saved to {output_file_path}")

print("Batch processing complete!")