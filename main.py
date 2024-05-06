import os

import cv2

import src.utils as utils
from src.constants import VIDEOS_PATH
from src.pipeline import Pipeline
from src.step import BackgroundSubtractionStep, BoundingBoxStep, CentroidStep, KalmanFilterStep

example_pipeline = Pipeline('Example pipeline',
                            [
                                BackgroundSubtractionStep('BS Step test',
                                                          cv2.createBackgroundSubtractorMOG2(detectShadows=True)),
                                BoundingBoxStep('BBOX step test'),
                                CentroidStep('Centroid step test'),
                                KalmanFilterStep('Kalman filter step test')
                            ])


cap = cv2.VideoCapture(os.path.join(VIDEOS_PATH, "Walking.54138969.mp4"))
estimations = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    output_frame = frame.copy()
    run_result = example_pipeline.run(frame)
    if run_result is None:
        continue
    estimation, correction = run_result
    estimations.append(estimation)

cv2.destroyAllWindows()
cap.release()
print(estimations)
