import os
import cv2
from tools import clear_terminal, show_points_of_interest, write_to_csv
from mediapipe_extract import (
    extract_features,
    extract_features_v2,
    get_distance_between_corner_eyes,
    get_movement_from_features,
    feature_normalization,
)
import numpy as np
import math
import matplotlib.pyplot as plt

from videoProcessingTools import get_angles

from dotenv import load_dotenv

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH")


# ~~~~~~ Setup params ~~~~~~~~~~~
windowLenghtMs = 500
windowOverlapRatio = 0.5
inputFilePath = f"{BASE_PATH}tudor/movie/WIN_20250303_21_20_32_Pro.mp4"
csvOutputFile = f"{BASE_PATH}tudor/movie/gest1-4.csv"
clear_terminal()
# ~~~~~~~~~~~~~~~~~~~~~~~~~

# ----- Read video , show details ----
cap = cv2.VideoCapture(inputFilePath)
videoHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
videoWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
print(
    f"----Video details:----\nHeight: {videoHeight}\nWidth: {videoWidth}\nFPS: {fps}\nDuration[s]: {duration}\nTotal frames:{frame_count}\n---------"
)
# ----------------------

# ~~~~~~~~~~~~ parse the file, output a matrix with all features ~~~~~~~~~~~
print("\n\n-->Reading video frame\n")
allFeatures = np.zeros((frame_count, 10))
pointsOfInterestNose = [1]
pointsOfInterestMouth = [61, 11, 291, 16]
pointsOfInterestEyeBrows = [53, 52, 65, 295, 282, 276]
pointsOfInterest = (
    pointsOfInterestNose + pointsOfInterestMouth + pointsOfInterestEyeBrows
)
points_to_extract = [
    4,
    5,
    25,
    9,
    10,
    44,
    45,
]  # zero position index, browOuterUpLeft, browOuterUpRight, jawOpen, eyeBlinkLeft, eyeBlinkRight, smileLeft, smileRight


for frameNr in range(0, frame_count - 1):
    ret, frame = cap.read()
    img_h, img_w, _ = frame.shape

    print(
        "Reading frame nr {} from total of {}".format(frameNr, frame_count - 1),
        end="\r",
    )
    if ret:
        result_features = extract_features_v2(frame)

        processedFeaturesRow = [c.score for c in result_features.face_blendshapes[0]]
        reducedFeaturesRow = [processedFeaturesRow[i] for i in points_to_extract]

        rawArray = np.array(
            [[el.x, el.y, el.z] for el in result_features.face_landmarks[0]]
        )
        rawArrayRow = rawArray.reshape(-1)

        reducedFeaturesRow.append(get_distance_between_corner_eyes(rawArrayRow))
        x, y = get_angles(rawArrayRow, img_w, img_h)
        reducedFeaturesRow.append(x)
        reducedFeaturesRow.append(y)
        allFeatures[frameNr, :] = reducedFeaturesRow


print("\nDone")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ===== group frame features in overlapping windows and check movement degree per window ========
windowLengthInFrames = math.floor(windowLenghtMs * fps / 1000)
step = math.floor((1 - windowOverlapRatio) * fps)


# =================

# --------- print results ---------
write_to_csv(
    csvOutputFile,
    [
        [inputFilePath],
        [
            "browOuterUpLeft",
            "browOuterUpRight",
            "jawOpen",
            "eyeBlinkLeft",
            "eyeBlinkRight",
            "smileLeft",
            "smileRight",
            "distanceBetweenCornerEyes",
            "angleUpDown",
            "angleLeftRight",
        ],
        *allFeatures,
    ],
)

# ------------------------


# ====  clean up =====
cap.release()
cv2.destroyAllWindows()
# ===================
