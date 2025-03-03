import cv2
from mediapipe_extract import (
    extract_features,
    feature_normalization,
    extract_features_v2,
    get_distance_between_corner_eyes,
)
import numpy as np


def play_video_and_extract(cap, startSec, stopSec):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        print("Error: Cannot open video file")
        exit()

    start_frame = int(startSec * fps)  # Define start frame
    end_frame = int(stopSec * fps)  # Define end frame

    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while True:
        current_frame = int(
            cap.get(cv2.CAP_PROP_POS_FRAMES)
        )  # Get current frame number
        if current_frame > end_frame:  # Stop when reaching the end frame
            break
        ret, frame = cap.read()
        if not ret:
            break  # Exit if video ends or there's an error
        cv2.imshow("Video Playback", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):  # Press 'q' to exit
            break


points_to_extract = [
    4,
    5,
    25,
    9,
    10,
    44,
    45,
]  # zero position index, browOuterUpLeft, browOuterUpRight, jawOpen, eyeBlinkLeft, eyeBlinkRight, smileLeft, smileRight
points_to_get_angles = [1, 61, 291, 33, 263, 199]


def extract_features_for_gesture(cap, startSec, stopSec):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        print("Error: Cannot open video file")
        exit()

    start_frame = int(startSec * fps)  # Define start frame
    end_frame = int(stopSec * fps)  # Define end frame
    rawFeatures = []
    rawFeaturesNormalized = []
    processedFeatures = []
    reducedFeatures = []
    # Set the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while True:
        current_frame = int(
            cap.get(cv2.CAP_PROP_POS_FRAMES)
        )  # Get current frame number
        if current_frame > end_frame:  # Stop when reaching the end frame
            break
        ret, frame = cap.read()
        img_h, img_w, _ = frame.shape
        if not ret:
            break  # Exit if video ends or there's an error
        result_features = extract_features_v2(frame)
        rawArray = np.array(
            [[el.x, el.y, el.z] for el in result_features.face_landmarks[0]]
        )
        rawArrayNormalized = feature_normalization(rawArray.copy())
        rawArrayRow = rawArray.reshape(-1)
        rawFeatures.append(list(rawArrayRow))
        rawFeaturesNormalized.append(list(rawArrayNormalized.reshape(-1)))
        processedFeaturesRow = [c.score for c in result_features.face_blendshapes[0]]
        processedFeatures.append(processedFeaturesRow)
        reducedFeaturesRow = [processedFeaturesRow[i] for i in points_to_extract]
        reducedFeaturesRow.append(get_distance_between_corner_eyes(rawArrayRow))
        x, y = get_angles(rawArrayRow, img_w, img_h)
        reducedFeaturesRow.append(x)
        reducedFeaturesRow.append(y)
        reducedFeatures.append(reducedFeaturesRow)

    return (rawFeatures, rawFeaturesNormalized, processedFeatures, reducedFeatures)


def get_angles(result_features, img_w=1980, img_h=1080):
    sublist = []
    for point_idx in points_to_get_angles:
        base_index = (point_idx - 1) * 3  # Each point has x, y, z coordinates
        sublist.append(result_features[base_index : base_index + 3])
    face_3d = [[int(e[0] * img_w), int(e[1] * img_h), e[2]] for e in sublist]
    face_3d = np.array(face_3d, dtype=np.float64)
    face_2d = np.array(face_3d[:, 0:2], dtype=np.float64)
    focal_length = 1 * img_w

    cam_matrix = np.array(
        [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
    )

    # The distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, dist_matrix
    )

    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)

    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    # Get the y rotation degree
    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360
    return x, y
