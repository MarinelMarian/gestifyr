import cv2
import mediapipe as mp
import numpy as np
import math

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5, refine_landmarks=True)

def getDistanteBetweenEyes(points:np.ndarray):
    leftEyeMiddle = ( points[33] + points[133] ) / 2
    rightEyeMiddle = ( points[362] + points[263] ) / 2
    return np.sqrt( np.sum( np.square(leftEyeMiddle - rightEyeMiddle) ) )

def extractFeatures(frame:cv2.typing.MatLike):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0] # get first face recognized
        features = np.array([[lmk.x, lmk.y, lmk.z] for lmk in face_landmarks.landmark])
        return features
    return np.array([])

def featureNormalization(features):
    return features / getDistanteBetweenEyes(features)

def getMovementFromFeatures(featuresWindow, pointsOfInterest):
    deltaX = featuresWindow[:, pointsOfInterest, 0].max(axis=0) - featuresWindow[:,pointsOfInterest,0].min(axis=0)
    deltaY = featuresWindow[:, pointsOfInterest, 1].max(axis=0) - featuresWindow[:,pointsOfInterest,1].min(axis=0)
    deltaZ = featuresWindow[:, pointsOfInterest, 2].max(axis=0) - featuresWindow[:,pointsOfInterest,2].min(axis=0)
    return math.sqrt( (deltaX**2).sum() + (deltaY**2).sum() + (deltaZ**2).sum())

