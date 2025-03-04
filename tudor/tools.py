from os import system, name
import cv2
import csv
import os


def clear_terminal():

    # for windows
    if name == "nt":
        _ = system("cls")

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system("clear")


def show_points_of_interest(frame, all_features, points_of_interest):
    h, w, _ = frame.shape
    for p in points_of_interest:
        cv2.circle(
            frame,
            (int(all_features[p, 0] * w), int(all_features[p, 1] * h)),
            10,
            (0, 255, 0),
            3,
        )
    cv2.imshow("Webcam Frame", frame)
    # Wait for a key press
    cv2.waitKey(5)


def write_to_csv(file_name, data):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print(f"CSV file {file_name} has been written successfully.")
