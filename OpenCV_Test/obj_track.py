import cv2
import sys

print(cv2.__version__)
 


# tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
# tracker_type = tracker_types[7]

# if int(minor_ver) < 3:
#     tracker = cv2.Tracker_create(tracker_type)
# else:
#     if tracker_type == 'BOOSTING':
#         tracker = cv2.TrackerBoosting_create()
#     elif tracker_type == 'MIL':
#         tracker = cv2.TrackerMIL_create()
#     elif tracker_type == 'KCF':
#         tracker = cv2.TrackerKCF_create()
#     elif tracker_type == 'TLD':
#         tracker = cv2.TrackerTLD_create()
#     elif tracker_type == 'MEDIANFLOW':
#         tracker = cv2.TrackerMedianFlow_create()
#     elif tracker_type == 'GOTURN':
#             tracker = cv2.TrackerGOTURN_create()
#     elif tracker_type == 'MOSSE':
#         tracker = cv2.TrackerMOSSE_create()
#     elif tracker_type == "CSRT":
#         tracker = cv2.TrackerCSRT_create()

# video = cv2.VideoCapture('resources/soccer_juggle.mp4')
# # load video
# if not video.isOpened():
#     print('[ERROR] video file not loaded')
#     sys.exit()
# # capture first frame
# ok, frame = video.read()
# if not ok:
#     print('[ERROR] no frame captured')
#     sys.exit()
# print('[INFO] video loaded and frame capture started')

# bbox = cv2.selectROI(frame)
# print('[INFO] select ROI and press ENTER or SPACE')
# print('[INFO] cancel selection by pressing C')
# print(bbox)