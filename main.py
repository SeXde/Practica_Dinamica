import os

import cv2

import src.utils as utils
from src.constants import VIDEOS_PATH
from src.pipeline import Pipeline
from src.step import BackgroundSubtractionStep, BoundingBoxStep

example_pipeline = Pipeline('Example pipeline',
                            [
                                BackgroundSubtractionStep('BS Step test',
                                                          cv2.createBackgroundSubtractorMOG2(detectShadows=True)),
                                BoundingBoxStep('BBOX step test')
                            ])


cap = cv2.VideoCapture(os.path.join(VIDEOS_PATH, "Walking.54138969.mp4"))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        output_frame = frame.copy()
        bbox = example_pipeline.run(frame)

        if bbox is not None:
            utils.draw_bbox(output_frame, bbox)
            cx, cy = utils.centroid_from_bbox(bbox)
            cv2.circle(output_frame, (cx, cy), 5, (0, 255, 0), -1)

        cv2.imshow('Original video', output_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # esc
        print('Exiting...')
        break

cv2.destroyAllWindows()
cap.release()

