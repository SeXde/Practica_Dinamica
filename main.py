import cv2
import os

import src.utils as utils
from src.person_detector import PersonDetector
from src.constants import VIDEOS_PATH

cap = cv2.VideoCapture(os.path.join(VIDEOS_PATH, "Walking.54138969.mp4"))
person_detector = PersonDetector()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        output_frame = frame.copy()
        bbox = person_detector.detect(frame)

        if bbox is not None:
            utils.draw_bbox(output_frame, bbox)
            cx, cy = utils.centroid_from_bbox(bbox)
            cv2.circle(output_frame, (cx, cy), 5, (0, 255, 0), -1)

        cv2.imshow('Original video', output_frame)
        cv2.imshow('Foreground Mask', person_detector.bs_filtered)

    if cv2.waitKey(1) & 0xFF == 27:  # esc
        print('Exiting...')
        break

cv2.destroyAllWindows()
cap.release()
