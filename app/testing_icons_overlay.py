import cv2
from videoProcessingTools import overlayBar

# Load the PNG icon with transparency
icon_path_raise_eyebrow = "/Users/tudormarcel.culda/Work/python/inovationLab2024/github/gestifyr/app/images/raise_eyebrow.png"  # Change this to your PNG file
icon_path_open_mouth = "/Users/tudormarcel.culda/Work/python/inovationLab2024/github/gestifyr/app/images/mouth_open.png"  # Change this to your PNG file

icon_raise_eyebrows = cv2.imread(icon_path_raise_eyebrow, cv2.IMREAD_UNCHANGED)
icon_open_mouth = cv2.imread(icon_path_open_mouth, cv2.IMREAD_UNCHANGED)
    

# Initialize webcam
cap = cv2.VideoCapture(0)

# Progress bar settings
progress = 0
max_progress = 100
step = 2  # Change rate per frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = overlayBar( frame, position_idx = 2, value = progress/max_progress, icon_image = icon_raise_eyebrows , threshold = 0.3)
    
    frame = overlayBar( frame, position_idx = 3, value = progress/max_progress, icon_image = icon_open_mouth, threshold = 0.8 )

    

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