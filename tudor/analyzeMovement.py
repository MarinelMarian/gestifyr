import csv
import matplotlib.pyplot as plt
import numpy as np
from tools import clearTerminal, write2Csv
from videoProcessingTools import playVideoAndExtract, extractFeatures4gesture
import cv2
import datetime as dt


clearTerminal()

# ------ Inpud data ---------
movementFile = 'movement_output.csv'
# -------------------


# ~~~~~~ Prepare data ~~~~~~~~
data = []
with open(movementFile, "r") as file:
    reader = csv.reader(file)
    videoFile = next(reader)
    data = [list(map(float,row)) for row in reader]
cap = cv2.VideoCapture(videoFile[0])
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
    if event.key =='z':
        lineStart.remove()
        latestStartCoordinate = latestMouseXcoordinate
        lineStart = ax.axvline(x=latestStartCoordinate, color="red", linestyle="--", linewidth=2, label="Start cut")
        fig.canvas.draw()  # Update the figure
    if event.key =='x':
        lineStop.remove()
        latestStopCoordinate = latestMouseXcoordinate
        lineStop = ax.axvline(x=latestStopCoordinate, color="green", linestyle="--", linewidth=2, label="Stop cut")
        fig.canvas.draw()  # Update the figure
    if event.key == 'a':
        playVideoAndExtract(cap, latestStartCoordinate, latestStopCoordinate)
    if event.key in ['1','2','3','4','5','6','7','8','9']:
        rawFeatures, rawFeaturesNormalized, processedFeatures = extractFeatures4gesture(cap, latestStartCoordinate, latestStopCoordinate)
        fileNameraw = f'tudor/raw/gesture_{event.key}__{int(dt.datetime.now().timestamp())}.csv'
        firstRow = [[dict([('gestureId', event.key), ('fps', cap.get(cv2.CAP_PROP_FPS))])]]
        write2Csv(fileNameraw, firstRow + rawFeatures)
        fileNamerawNorm = f'tudor/raw_normalized/gesture_{event.key}__{int(dt.datetime.now().timestamp())}.csv'
        write2Csv(fileNamerawNorm, firstRow + rawFeaturesNormalized)
        fileNameProcessed = f'tudor/processed/gesture_{event.key}__{int(dt.datetime.now().timestamp())}.csv'
        write2Csv(fileNameProcessed, firstRow + processedFeatures)
        print("Done saving")
         
         
fig.canvas.mpl_connect("motion_notify_event", onMouseMove)
fig.canvas.mpl_connect("key_press_event", onKeyPress)  # Key press

xPoints = data[0]
ax.plot( xPoints, data[1], label="Nose movement", color='blue')
ax.plot(xPoints,  data[2], label='Mouth movement', color='red')
ax.plot(xPoints, data[3], label= 'Eyebrows movement', color='green')
lineStart = ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Start cut (press z to change)")
lineStop = ax.axvline(x=xPoints[-1], color="green", linestyle="--", linewidth=2, label="Stop cut(press x to change)")

ax.set_xlabel('time (s)')
ax.legend()
ax.set_title("z,x --> set markers ; a --> play selection ; 1..9 --> save gesture")
plt.show()




# ====  clean up =====
cap.release()
cv2.destroyAllWindows()
# ===================

print('Done')
