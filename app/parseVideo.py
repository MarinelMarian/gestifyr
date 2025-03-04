import os
import cv2
from tools import clear_terminal, show_points_of_interest, write_to_csv
from mediapipe_extract import (
    extract_features,
    extract_features_v2,
    get_movement_from_features,
)
import numpy as np
import math
import matplotlib.pyplot as plt

from videoProcessingTools import get_angles

from dotenv import load_dotenv

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH")
APP_REL_PATH = os.getenv("APP_REL_PATH")


# ~~~~~~ Setup params ~~~~~~~~~~~
window_length_ms = 500
window_overlap_ratio = 0.5
input_file_path = f"{APP_REL_PATH}/movies/WIN_20250303_21_20_32_Pro.mp4"
csv_output_file = f"{APP_REL_PATH}/movies/gest1-4.csv"
clear_terminal()
# ~~~~~~~~~~~~~~~~~~~~~~~~~

# ----- Read video , show details ----
cap = cv2.VideoCapture(f"{BASE_PATH}{input_file_path}")
video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
print(
    f"----Video details:----\nHeight: {video_height}\nWidth: {video_width}\nFPS: {fps}\nDuration[s]: {duration}\nTotal frames:{frame_count}\n---------"
)
# ----------------------

# ~~~~~~~~~~~~ parse the file, output a matrix with all features ~~~~~~~~~~~
print("\n\n-->Reading video frame\n")
points_of_interest_nose = [1]
points_of_interest_mouth = [61, 11, 291, 16]
points_of_interest_eyebrows = [53, 52, 65, 295, 282, 276]
points_of_interest = (
    points_of_interest_nose + points_of_interest_mouth + points_of_interest_eyebrows
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
all_features = np.zeros((frame_count, len(points_to_extract) + 2))


for frame_nr in range(0, frame_count - 1):
    ret, frame = cap.read()
    img_h, img_w, _ = frame.shape

    print(
        "Reading frame nr {} from total of {}".format(frame_nr, frame_count - 1),
        end="\r",
    )
    if ret:
        result_features = extract_features_v2(frame)

        processed_features_row = [c.score for c in result_features.face_blendshapes[0]]
        reduced_features_row = [processed_features_row[i] for i in points_to_extract]

        raw_array = np.array(
            [[el.x, el.y, el.z] for el in result_features.face_landmarks[0]]
        )
        raw_array_row = raw_array.reshape(-1)

        x, y = get_angles(raw_array_row, img_w, img_h)
        reduced_features_row.append(x)
        reduced_features_row.append(y)
        all_features[frame_nr, :] = reduced_features_row


print("\nDone")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ===== group frame features in overlapping windows and check movement degree per window ========
window_length_in_frames = math.floor(window_length_ms * fps / 1000)
step = math.floor((1 - window_overlap_ratio) * fps)


# =================

# --------- print results ---------
write_to_csv(
    f"{BASE_PATH}{csv_output_file}",
    [
        [input_file_path],
        [
            "browOuterUpLeft",
            "browOuterUpRight",
            "jawOpen",
            "eyeBlinkLeft",
            "eyeBlinkRight",
            "smileLeft",
            "smileRight",
            "angleUpDown",
            "angleLeftRight",
        ],
        *all_features,
    ],
)

# ------------------------


# ====  clean up =====
cap.release()
cv2.destroyAllWindows()
# ===================
