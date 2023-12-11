import cv2 as cv
import pandas as pd
import numpy as np

#folder path constant
folder_path = "train/game4/"

#fill in with information
video_name = "game_4.mp4"

#destination folder path
dest_path = "train/game4/images/"
#base for the names of images
name_base = "frame"
#file format
file_extension = ".jpg"


# Source the video feed
cap = cv.VideoCapture(folder_path + "input/"+  video_name)
i = 0
data = pd.read_csv(folder_path + "data/formatted_data.csv")

arr = pd.read_csv(folder_path + "data/valid_idx.csv")
valid_idx = arr.to_numpy()

# While there is video, run the model per frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Resize optional
    # frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    if(i in valid_idx): 
        cv.imwrite(dest_path + name_base + str(i) + file_extension, frame)
        print("valid " + str(i))
    else:
        print("invalid " + str(i))
    i += 1

print("final i:" + str(i))
# Release the video
cap.release()
