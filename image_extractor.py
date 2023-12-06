import cv2 as cv



#fill in with information
video_path = "train/game4/input/game_4.mp4"

#destination folder path
dest_path = "train/game4/images/"
#base for the names of images
name_base = "frame"
#file format
file_extension = ".jpg"


# Source the video feed
cap = cv.VideoCapture(video_path)
i = 0

# While there is video, run the model per frame of the video
while i < 10:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize optional
    # frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    cv.imwrite(dest_path + name_base + str(i) + file_extension, frame)
    i += 1

# Release the video
cap.release()
