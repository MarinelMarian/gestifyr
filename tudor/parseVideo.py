import cv2
from tools import clearTerminal, showPointsOfInterest, write2Csv
from mediapipe_extract import extractFeatures, getMovementFromFeatures, featureNormalization
import numpy as np
import math
import matplotlib.pyplot as plt

# ~~~~~~ Setup params ~~~~~~~~~~~
windowLenghtMs = 500
windowOverlapRatio = 0.5
inputFilePath = './tudor/test_tudor.mov'
clearTerminal()
# ~~~~~~~~~~~~~~~~~~~~~~~~~

# ----- Read video , show details ----
cap = cv2.VideoCapture(inputFilePath)
videoHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
videoWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# frame_count = 200
duration = frame_count/fps
print(f'----Video details:----\nHeight: {videoHeight}\nWidth: {videoWidth}\nFPS: {fps}\nDuration[s]: {duration}\nTotal frames:{frame_count}\n---------')
# ----------------------

#~~~~~~~~~~~~ parse the file, output a matrix with all features ~~~~~~~~~~~
print("\n\n-->Reading video frame\n")
allFeatures = np.zeros((frame_count, 478, 3))
pointsOfInterestNose = [1]
pointsOfInterestMouth = [61, 11, 291, 16]
pointsOfInterestEyeBrows = [53,52,65,295,282,276]
pointsOfInterest = pointsOfInterestNose + pointsOfInterestMouth + pointsOfInterestEyeBrows

for frameNr in range(0, frame_count-1):
    ret, frame = cap.read()
    print("Reading frame nr {} from total of {}".format(frameNr, frame_count-1), end='\r')
    if ret:
        allFeatures[frameNr, :, :] = extractFeatures(frame)
        # showPointsOfInterest(frame, allFeatures[frameNr, :, :], pointsOfInterest)

print('\nDone')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#===== group frame features in overlapping windows and check movement degree per window ========
windowLengthInFrames = math.floor(windowLenghtMs * fps/1000)
step = math.floor((1-windowOverlapRatio) * fps)

movementDegreeNose = np.zeros(math.floor((frame_count - 1 - windowLengthInFrames)/step) +1 ) # how many windows of windowLength can fit with step advancement in total frame count
movementDegreeMouth = np.zeros(math.floor((frame_count - 1 - windowLengthInFrames)/step) +1 )
movementDegreeEyebrows = np.zeros(math.floor((frame_count - 1 - windowLengthInFrames)/step) +1 )
for i in range(0, frame_count - 1 - windowLengthInFrames, step):
    movementDegreeNose[int(i/step)] = getMovementFromFeatures(allFeatures[i:i+step-1,:,:], pointsOfInterestNose)
    movementDegreeMouth[int(i/step)] = getMovementFromFeatures(allFeatures[i:i+step-1,:,:], pointsOfInterestMouth)
    movementDegreeEyebrows[int(i/step)] = getMovementFromFeatures(allFeatures[i:i+step-1,:,:], pointsOfInterestEyeBrows)

#=================

# --------- print results ---------
xPoints = np.arange(len(movementDegreeNose))*step / fps
plt.plot( xPoints, movementDegreeNose, label="Nose movement", color='blue')
plt.plot(xPoints,  movementDegreeMouth, label='Mouth movement', color='red')
plt.plot(xPoints, movementDegreeEyebrows, label= 'Eyebrows movement', color='green')
plt.xlabel('time (s)')
plt.legend()
plt.show()
write2Csv('movement_output.csv',[[inputFilePath], xPoints.tolist(), movementDegreeNose.tolist(), movementDegreeMouth.tolist(), movementDegreeEyebrows.tolist()])

# ------------------------


# ====  clean up =====
cap.release()
cv2.destroyAllWindows()
# ===================



