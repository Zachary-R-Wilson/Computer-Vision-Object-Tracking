import cv2 as cv
import numpy as np
from mss import mss
import pyautogui
import time

w, h = pyautogui.size()
print("Screen Resolution: " + str(w) + 'x' + str(h))

img = None
t0=time.time()
n_frames = 1
monitor = {"top": 0, "left": 0, "width": int(w/2), "height": h}

with mss() as sct:
    while True:
        img = sct.grab(monitor)
        img = np.array(img)

        small = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
        cv.imshow("Computer Vision", small)
        key = cv.waitKey(1)
        if key == ord('q'):
            break

        elapsed_time = time.time() - t0
        avg_fps = (n_frames / elapsed_time)
        print("Average FPS: " + str(avg_fps))
        n_frames += 1

cv.destroyAllWindows()