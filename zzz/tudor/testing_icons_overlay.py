import cv2
from videoProcessingTools import overlayBar

# Load the PNG icon with transparency
icon_path_raise_eyebrow = "/Users/tudormarcel.culda/Work/python/inovationLab2024/github/gestifyr/app/images/raise_eyebrow.png"  # Change this to your PNG file
icon_path_open_mouth = "/Users/tudormarcel.culda/Work/python/inovationLab2024/github/gestifyr/app/images/mouth_open.png"  # Change this to your PNG file
icon_path_nod = "/Users/tudormarcel.culda/Work/python/inovationLab2024/github/gestifyr/app/images/nod.png"  # Change this to your PNG file
icon_path_shake = "/Users/tudormarcel.culda/Work/python/inovationLab2024/github/gestifyr/app/images/shake.png"  # Change this to your PNG file
icon_path_smile = "/Users/tudormarcel.culda/Work/python/inovationLab2024/github/gestifyr/app/images/smile.png"  # Change this to your PNG file
icon_path_eyes_shut = "/Users/tudormarcel.culda/Work/python/inovationLab2024/github/gestifyr/app/images/eyes_shut.png"  # Change this to your PNG file
icon_path_none = "/Users/tudormarcel.culda/Work/python/inovationLab2024/github/gestifyr/app/images/none.png"  # Change this to your PNG file

icon_raise_eyebrows = cv2.imread(icon_path_raise_eyebrow, cv2.IMREAD_UNCHANGED)
icon_open_mouth = cv2.imread(icon_path_open_mouth, cv2.IMREAD_UNCHANGED)
icon_nod = cv2.imread(icon_path_nod, cv2.IMREAD_UNCHANGED)
icon_shake = cv2.imread(icon_path_shake, cv2.IMREAD_UNCHANGED)
icon_smile = cv2.imread(icon_path_smile, cv2.IMREAD_UNCHANGED)
icon_eyes_shut = cv2.imread(icon_path_eyes_shut, cv2.IMREAD_UNCHANGED)   
icon_none = cv2.imread(icon_path_none, cv2.IMREAD_UNCHANGED)   

# Initialize webcam
cap = cv2.VideoCapture(0)

# Progress bar settings
progress = 0
max_progress = 100
step = 0.2  # Change rate per frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = overlayBar( frame, position_idx = 1, value = min(progress/2/max_progress,1), icon_image = icon_nod , threshold = 0.3)
    frame = overlayBar( frame, position_idx = 2, value = min(progress*1.2/max_progress,1), icon_image = icon_shake , threshold = 0.5)
    frame = overlayBar( frame, position_idx = 3, value = min(progress*1.1/max_progress,1), icon_image = icon_open_mouth , threshold = 0.3)
    frame = overlayBar( frame, position_idx = 4, value = min(progress/max_progress,1), icon_image = icon_raise_eyebrows , threshold = 0.2)
    frame = overlayBar( frame, position_idx = 5, value = min(progress/0.7/max_progress,1), icon_image = icon_smile , threshold = 0.3)
    frame = overlayBar( frame, position_idx = 6, value = min(progress*1.3/max_progress,1), icon_image = icon_eyes_shut , threshold = 0.4)  
    frame = overlayBar( frame, position_idx = 7, value = min(progress*0.9/max_progress,1), icon_image = icon_none, threshold = 0.8 )

    

    # Update progress
    progress += step
    if progress > max_progress:
        progress = 0

    # Show video with progress bar and PNG icon
    cv2.imshow("Webcam", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()