import os
import cv2
from videoProcessingTools import overlayBar, overlayRoundedSquare
import numpy as np
from dotenv import load_dotenv

load_dotenv()
BASE_PATH = os.getenv("BASE_PATH")
APP_REL_PATH = os.getenv("APP_REL_PATH")

# Load the PNG icon with transparency
icon_path_raise_eyebrow = f"{BASE_PATH}{APP_REL_PATH}images/raise_eyebrow.png"  # Change this to your PNG file
icon_path_open_mouth = f"{BASE_PATH}{APP_REL_PATH}images/mouth_open.png"  # Change this to your PNG file
icon_path_nod = f"{BASE_PATH}{APP_REL_PATH}images/nod.png"  # Change this to your PNG file
icon_path_shake = f"{BASE_PATH}{APP_REL_PATH}images/shake.png"  # Change this to your PNG file
icon_path_smile = f"{BASE_PATH}{APP_REL_PATH}images/smile.png"  # Change this to your PNG file
icon_path_eyes_shut = f"{BASE_PATH}{APP_REL_PATH}images/eyes_shut.png"  # Change this to your PNG file
icon_path_none = f"{BASE_PATH}{APP_REL_PATH}images/none.png"  # Change this to your PNG file

icon_raise_eyebrows = cv2.imread(icon_path_raise_eyebrow, cv2.IMREAD_UNCHANGED)
icon_open_mouth = cv2.imread(icon_path_open_mouth, cv2.IMREAD_UNCHANGED)
icon_nod = cv2.imread(icon_path_nod, cv2.IMREAD_UNCHANGED)
icon_shake = cv2.imread(icon_path_shake, cv2.IMREAD_UNCHANGED)
icon_smile = cv2.imread(icon_path_smile, cv2.IMREAD_UNCHANGED)
icon_eyes_shut = cv2.imread(icon_path_eyes_shut, cv2.IMREAD_UNCHANGED)   
icon_none = cv2.imread(icon_path_none, cv2.IMREAD_UNCHANGED)   