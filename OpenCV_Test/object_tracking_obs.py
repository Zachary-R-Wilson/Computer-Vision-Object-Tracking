import cv2 as cv
import numpy as np
import time

cap = cv.VideoCapture(0)
count = 0
while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    cv.imshow("Computer Vision", frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()