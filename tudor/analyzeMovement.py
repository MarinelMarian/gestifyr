import csv
from email import header
import matplotlib.pyplot as plt
import numpy as np
from tools import clear_terminal, write_to_csv
from videoProcessingTools import play_video_and_extract, extract_features_for_gesture
import cv2
import datetime as dt

BASE_PATH = "D:/onedrive/source/repos/gestifyr/"

clear_terminal()

# ------ Input data ---------
movementFile = f"{BASE_PATH}tudor/movie/gest1-4.csv"
# -------------------


# ~~~~~~ Prepare data ~~~~~~~~
data = []
header_names = []
with open(movementFile, "r") as file:
    reader = csv.reader(file)
    videoFile = next(reader)
    header_names = next(reader)
    data = [list(map(float, row)) for row in reader]
cap = cv2.VideoCapture(videoFile[0])
fps = cap.get(cv2.CAP_PROP_FPS)
# ~~~~~~~~~~~~~~~~~~


# --------- open plot as GUI, key events on plot will trigger video and processing ---------
latestMouseXcoordinate = 0
latestStartCoordinate = 0
latestStopCoordinate = 0

fig, ax = plt.subplots()


def onMouseMove(event):
    global latestMouseXcoordinate
    if event.xdata is not None and event.ydata is not None:  # Check if inside the axes
        latestMouseXcoordinate = event.xdata


def onKeyPress(event):
    global latestStartCoordinate, latestStopCoordinate, lineStart, lineStop
    if event.key == "z":
        lineStart.remove()
        latestStartCoordinate = latestMouseXcoordinate
        lineStart = ax.axvline(
            x=latestStartCoordinate,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Start cut",
        )
        fig.canvas.draw()  # Update the figure
    if event.key == "x":
        lineStop.remove()
        latestStopCoordinate = latestMouseXcoordinate
        lineStop = ax.axvline(
            x=latestStopCoordinate,
            color="green",
            linestyle="--",
            linewidth=2,
            label="Stop cut",
        )
        fig.canvas.draw()  # Update the figure
    if event.key == "a":
        play_video_and_extract(
            cap, latestStartCoordinate / fps, latestStopCoordinate / fps
        )
    if event.key in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:  # 6 - long blink
        timestamp = int(dt.datetime.now().timestamp())
        rawFeatures, rawFeaturesNormalized, processedFeatures, reducedFeatures = (
            extract_features_for_gesture(
                cap, latestStartCoordinate / fps, latestStopCoordinate / fps
            )
        )
        fileNameraw = (
            f"{BASE_PATH}tudor/samples3/raw/gesture_{event.key}__{timestamp}.csv"
        )
        firstRow = [
            [dict([("gestureId", event.key), ("fps", cap.get(cv2.CAP_PROP_FPS))])]
        ]
        write_to_csv(fileNameraw, firstRow + rawFeatures)
        fileNamerawNorm = f"{BASE_PATH}tudor/samples3/raw_normalized/gesture_{event.key}__{timestamp}.csv"
        write_to_csv(fileNamerawNorm, firstRow + rawFeaturesNormalized)
        fileNameProcessed = (
            f"{BASE_PATH}tudor/samples3/processed/gesture_{event.key}__{timestamp}.csv"
        )
        write_to_csv(fileNameProcessed, firstRow + processedFeatures)
        fileNameProcessed = f"{BASE_PATH}tudor/samples3/processed_trimmed/gesture_{event.key}__{timestamp}.csv"
        write_to_csv(fileNameProcessed, firstRow + reducedFeatures)

        print("Done saving")
    if event.key == "q":
        plt.close(fig)
        cap.release()
        cv2.destroyAllWindows()
        exit()


fig.canvas.mpl_connect("motion_notify_event", onMouseMove)
fig.canvas.mpl_connect("key_press_event", onKeyPress)  # Key press

feature_values = np.array(data)
for i in range(1, len(header_names)):
    ax.plot(feature_values[:, i], label=header_names[i])

lineStart = ax.axvline(
    x=0, color="red", linestyle="--", linewidth=2, label="Start cut (press z to change)"
)
lineStop = ax.axvline(
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
