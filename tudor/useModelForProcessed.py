import cv2
from collections import deque
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python





# ----- user params --------
queueFrameSizeSec = 1.5 #window length in seconds. This window will be the input to prediction
analyseVideoStepSec = 0.3 #in seconds, how much time should pass until next window is analyzed
#------------------------


# ~~~~~~ init ~~~~~
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
# Get the frames per second (FPS) of the video
fps = cap.get(cv2.CAP_PROP_FPS)
queueFrameSize = int(fps * queueFrameSizeSec)
analyseVideoStep = int(fps * analyseVideoStepSec)
frameQueue = deque(maxlen=queueFrameSize)
# ~~~~~~~~~~~~~~~~~~

# ====== mediapipe init =========
base_options = python.BaseOptions(model_asset_path='tudor/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)
def extractFeaturesv2(frame:cv2.typing.MatLike):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)
    return detection_result
# ============================

idx = 0
while True:    
    ret, frame = cap.read()
    idx += 1
    if not ret:
        print("End of video ? can't read new frame.")
        break
    
    result_features = extractFeaturesv2(frame) # extract frame features
    if len(result_features.face_blendshapes) == 0:
        continue
    frameQueue.append([c.score for c in result_features.face_blendshapes[0]])

    # Optional: Display the frame
    cv2.imshow('Capturing Frames', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if idx % analyseVideoStep == 0: #stop to analyze when step time has completed
        predictedLabel, confidence  = predictLabel(frameQueue) # <-------- add your model prediciton method
        print(f"Predicted Gesture    -->    {predictedLabel},    Confidence: {confidence}\n")

