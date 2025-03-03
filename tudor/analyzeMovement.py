import csv
import matplotlib.pyplot as plt
import numpy as np
from tools import clear_terminal, write_to_csv
from videoProcessingTools import play_video_and_extract, extract_features_for_gesture
import cv2
import datetime as dt
import os
from dotenv import load_dotenv

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH")

clear_terminal()

# ------ Input data ---------
movement_file = f"{BASE_PATH}tudor/movie/gest1-4.csv"
# -------------------


# ~~~~~~ Prepare data ~~~~~~~~
data = []
header_names = []
with open(movement_file, "r") as file:
    reader = csv.reader(file)
    video_file = next(reader)
    header_names = next(reader)
    data = [list(map(float, row)) for row in reader]
cap = cv2.VideoCapture(video_file[0])
fps = cap.get(cv2.CAP_PROP_FPS)
# ~~~~~~~~~~~~~~~~~~


# --------- open plot as GUI, key events on plot will trigger video and processing ---------
latest_mouse_x_coordinate = 0
latest_start_coordinate = 0
latest_stop_coordinate = 0

fig, ax = plt.subplots()


def on_mouse_move(event):
    global latest_mouse_x_coordinate
    if event.xdata is not None and event.ydata is not None:  # Check if inside the axes
        latest_mouse_x_coordinate = event.xdata


def on_key_press(event):
    global latest_start_coordinate, latest_stop_coordinate, line_start, line_stop
    if event.key == "z":
        line_start.remove()
        latest_start_coordinate = latest_mouse_x_coordinate
        line_start = ax.axvline(
            x=latest_start_coordinate,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Start cut",
        )
        fig.canvas.draw()  # Update the figure
    if event.key == "x":
        line_stop.remove()
        latest_stop_coordinate = latest_mouse_x_coordinate
        line_stop = ax.axvline(
            x=latest_stop_coordinate,
            color="green",
            linestyle="--",
            linewidth=2,
            label="Stop cut",
        )
        fig.canvas.draw()  # Update the figure
    if event.key == "a":
        play_video_and_extract(
            cap, latest_start_coordinate / fps, latest_stop_coordinate / fps
        )
    if event.key in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:  # 6 - long blink
        timestamp = int(dt.datetime.now().timestamp())
        raw_features, raw_features_normalized, processed_features, reduced_features = (
            extract_features_for_gesture(
                cap, latest_start_coordinate / fps, latest_stop_coordinate / fps
            )
        )
        file_name_raw = (
            f"{BASE_PATH}tudor/samples3/raw/gesture_{event.key}__{timestamp}.csv"
        )
        first_row = [
            [dict([("gestureId", event.key), ("fps", cap.get(cv2.CAP_PROP_FPS))])]
        ]
        write_to_csv(file_name_raw, first_row + raw_features)
        file_name_raw_norm = f"{BASE_PATH}tudor/samples3/raw_normalized/gesture_{event.key}__{timestamp}.csv"
        write_to_csv(file_name_raw_norm, first_row + raw_features_normalized)
        file_name_processed = (
            f"{BASE_PATH}tudor/samples3/processed/gesture_{event.key}__{timestamp}.csv"
        )
        write_to_csv(file_name_processed, first_row + processed_features)
        file_name_processed = f"{BASE_PATH}tudor/samples3/processed_trimmed/gesture_{event.key}__{timestamp}.csv"
        write_to_csv(file_name_processed, first_row + reduced_features)

        print("Done saving")
    if event.key == "q":
        plt.close(fig)
        cap.release()
        cv2.destroyAllWindows()
        exit()


fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
fig.canvas.mpl_connect("key_press_event", on_key_press)  # Key press

feature_values = np.array(data)
for i in range(1, len(header_names)):
    ax.plot(feature_values[:, i], label=header_names[i])

line_start = ax.axvline(
    x=0, color="red", linestyle="--", linewidth=2, label="Start cut (press z to change)"
)
line_stop = ax.axvline(
    x=10,
    color="green",
    linestyle="--",
    linewidth=2,
    label="Stop cut(press x to change)",
)

ax.set_xlabel("time (s)")
ax.legend()
ax.set_title("z,x --> set markers ; a --> play selection ; 1..9 --> save gesture")
plt.show()


# ====  clean up =====
cap.release()
cv2.destroyAllWindows()
# ===================

print("Done")
