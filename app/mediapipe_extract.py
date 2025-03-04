import cv2
import mediapipe as mp
import numpy as np
import math
from mediapipe.tasks.python import vision, BaseOptions
import os
from dotenv import load_dotenv

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH")
APP_REL_PATH = os.getenv("APP_REL_PATH")


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, refine_landmarks=True)

# base_options = python.BaseOptions(model_asset_path=f'{APP_REL_PATH}/face_landmarker_v2_with_blendshapes.task')
base_options = BaseOptions(model_asset_buffer=open(f"{BASE_PATH}{APP_REL_PATH}/face_landmarker_v2_with_blendshapes.task", "rb").read())
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


def get_distance_between_eyes(points: np.ndarray):
    left_eye_middle = (points[33] + points[133]) / 2
    right_eye_middle = (points[362] + points[263]) / 2
    return np.sqrt(np.sum(np.square(left_eye_middle - right_eye_middle)))


# def get_distance_between_corner_eyes(points:np.ndarray):
#     return np.sqrt( np.sum( np.square(points[33] - points[263]) ) )

def extract_features(frame: cv2.typing.MatLike):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]  # get first face recognized
        features = np.array([[lmk.x, lmk.y, lmk.z] for lmk in face_landmarks.landmark])
        return features
    return np.array([])


def extract_features_v2(frame: cv2.typing.MatLike):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    return detection_result


def feature_normalization(features):
    return features / get_distance_between_eyes(features)


def get_movement_from_features(features_window, points_of_interest):
    delta_x = features_window[:, points_of_interest, 0].max(axis=0) - features_window[:, points_of_interest, 0].min(axis=0)
    delta_y = features_window[:, points_of_interest, 1].max(axis=0) - features_window[:, points_of_interest, 1].min(axis=0)
    delta_z = features_window[:, points_of_interest, 2].max(axis=0) - features_window[:, points_of_interest, 2].min(axis=0)
    return math.sqrt((delta_x ** 2).sum() + (delta_y ** 2).sum() + (delta_z ** 2).sum())
