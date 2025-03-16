import os
import glob

# Directories
mislabeled_videos_dir = "mislabeled_videos"
samples_dir = "samples"

# Build a dictionary for CSV files based on the part after "_30_"
csv_files = glob.glob(os.path.join(samples_dir, "*.csv"))
csv_by_second_part = {}

for csv_path in csv_files:
    base_csv = os.path.splitext(os.path.basename(csv_path))[0]
    parts = base_csv.split("_30_")
    if len(parts) == 2:
        # Save a tuple with (first part, second part)
        csv_by_second_part[parts[1]] = (parts[0], parts[1])
    else:
        print(f"Skipping csv {csv_path} as it doesn't split into 2 parts.")

# Process mp4 files in the mislabeled_videos directory
mp4_files = glob.glob(os.path.join(mislabeled_videos_dir, "*.mp4"))

for mp4_path in mp4_files:
    base_mp4 = os.path.splitext(os.path.basename(mp4_path))[0]
    parts = base_mp4.split("_30_")
    if len(parts) != 2:
        print(f"Skipping mp4 {mp4_path} as it doesn't split into 2 parts.")
        continue

    second_part = parts[1]
    if second_part in csv_by_second_part:
        first_part_csv, second_part_csv = csv_by_second_part[second_part]
        # Build new filename based on the CSV file's naming convention
        new_filename = f"{first_part_csv}_30_{second_part_csv}.mp4"
        new_path = os.path.join(mislabeled_videos_dir, new_filename)
        print(f"Renaming:\n  {mp4_path}\nto\n  {new_path}\n")
        os.rename(mp4_path, new_path)
    else:
        print(f"No matching CSV found for {mp4_path}.")