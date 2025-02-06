from os import system, name
import cv2
import csv

def clearTerminal():

    # for windows
    if name == 'nt':
        _ = system('cls')

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

def showPointsOfInterest(frame, allFeatures, pointsOfInterest):
    h, w, _ = frame.shape
    for p in pointsOfInterest:
        cv2.circle(frame, 
                (int(allFeatures[p, 0]*w), int(allFeatures[p, 1]*h)), 
                10, (0, 255, 0), 3)
    cv2.imshow("Webcam Frame", frame) 
    # Wait for a key press
    cv2.waitKey(5)

def write2Csv(fileName, data):
    with open(fileName, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    print("CSV file has been written successfully.")
