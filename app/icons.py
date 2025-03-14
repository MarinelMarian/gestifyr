import cv2
from videoProcessingTools import overlayBar, overlayRoundedSquare
import numpy as np

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