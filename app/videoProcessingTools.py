import cv2
from mediapipe_extract import (
    extract_features,
    feature_normalization,
    extract_features_v2,
)
import numpy as np


def play_video_and_extract(cap, start_sec, stop_sec):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        print("Error: Cannot open video file")
        exit()

    start_frame = int(start_sec * fps)  # Define start frame
    end_frame = int(stop_sec * fps)  # Define end frame

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


def extract_features_for_gesture(cap, start_sec, stop_sec):
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened():
        print("Error: Cannot open video file")
        exit()

    start_frame = int(start_sec * fps)  # Define start frame
    end_frame = int(stop_sec * fps)  # Define end frame
    raw_features = []
    raw_features_normalized = []
    processed_features = []
    reduced_features = []
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
        raw_array = np.array(
            [[el.x, el.y, el.z] for el in result_features.face_landmarks[0]]
        )
        raw_array_normalized = feature_normalization(raw_array.copy())
        raw_array_row = raw_array.reshape(-1)
        raw_features.append(list(raw_array_row))
        raw_features_normalized.append(list(raw_array_normalized.reshape(-1)))
        processed_features_row = [c.score for c in result_features.face_blendshapes[0]]
        processed_features.append(processed_features_row)
        reduced_features_row = [processed_features_row[i] for i in points_to_extract]
        x, y = get_angles(raw_array_row, img_w, img_h)
        reduced_features_row.append(x)
        reduced_features_row.append(y)
        reduced_features.append(reduced_features_row)

    return (raw_features, raw_features_normalized, processed_features, reduced_features)


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


def overlayBar( frame, **kwargs):
    position_idx = kwargs['position_idx']
    value = kwargs['value']
    icon_image = kwargs['icon_image']
    thresh = kwargs['threshold']
    # Resize the icon if needed
    icon_size = 60  # Adjust this for desired size
    bar_width = 200
    bar_height = 60
    icon_image = cv2.resize(icon_image, (icon_size, icon_size))
    # Get frame dimensions
    height, width, _ = frame.shape

    # Define positions
    
    bar_x = width - bar_width - 60  # Move left to fit icon
    bar_y = 100 + (bar_height +30) * position_idx 

    # Icon position
    icon_x = bar_x - icon_size - 10  # Shift left of progress bar
    icon_y = bar_y  # Align with progress bar

    # Background box for overlay (Larger size)
    bg_x1, bg_y1 = icon_x - 10, bar_y - 15  # Top-left corner
    bg_x2, bg_y2 = bar_x + bar_width + 20, bar_y + bar_height + 15  # Bottom-right corner
    overlay = frame.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (50, 50, 50), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)  # Blend with transparency
    
    # Overlay PNG icon (handling transparency)
    icon_h, icon_w, icon_c = icon_image.shape
    if icon_c == 4:  # Check if image has an alpha channel
        for c in range(0, 3):  # Loop over BGR channels
            frame[icon_y:icon_y + icon_h, icon_x:icon_x + icon_w, c] = (
                frame[icon_y:icon_y + icon_h, icon_x:icon_x + icon_w, c] * (1 - icon_image[:, :, 3] / 255.0) +
                icon_image[:, :, c] * (icon_image[:, :, 3] / 255.0)
            )
    # Draw progress bar background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (80, 80, 80), -1)

    # Draw progress bar foreground
    fill_width = int(value * bar_width)
    fill_color = (0, 255, 0) if value>thresh else (33, 222, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), fill_color, -1)
    return frame