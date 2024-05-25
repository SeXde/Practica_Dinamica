import os

import cv2
import numpy as np

from src.constants import VIDEOS_PATH, BS_PATH
from src.evaluator.pipeline_evaluator import PipelineEvaluator
from src.pipeline import Pipeline
from src.step import BackgroundSubtractionStep, BoundingBoxStep, CentroidStep, KalmanFilterStep, BSCentroidStep, PFStep

kalman_pipeline = Pipeline('Kalman pipeline',
                           [
                               BackgroundSubtractionStep('BS Step',
                                                         cv2.createBackgroundSubtractorMOG2(detectShadows=True)),
                               BoundingBoxStep('BBOX step'),
                               CentroidStep('Centroid step'),
                               KalmanFilterStep('Kalman filter step')
                           ])

pf_pipeline = Pipeline('Particle filter pipeline',
                       [
                           BackgroundSubtractionStep('BS Step',
                                                     cv2.createBackgroundSubtractorMOG2(detectShadows=True)),
                           PFStep(100, (300, 300), (1002, 1000), 'Particle filter step'),
                           CentroidStep('Centroid step')
                       ])

gt_pipeline = Pipeline('GT pipeline',
                       [
                           BSCentroidStep('BS Centroid step')
                       ])

video_name = 'Walking.54138969.mp4'
x_cap = cv2.VideoCapture(os.path.join(VIDEOS_PATH, video_name))
y_cap = cv2.VideoCapture(os.path.join(BS_PATH, video_name))

kalman_estimations = []
pf_estimations = []
gt_estimations = []

while x_cap.isOpened() and y_cap.isOpened():
    ret_x, frame_x = x_cap.read()
    ret_y, frame_y = y_cap.read()
    if not ret_x or not ret_y:
        break

    # GT Results

    frame_y = cv2.cvtColor(frame_y, cv2.COLOR_BGR2GRAY)
    _, frame_y = cv2.threshold(frame_y, 1, 255, cv2.THRESH_BINARY)
    centroid, _ = gt_pipeline.run(frame_y)
    gt_estimations.append(centroid)

    # Kalman results

    run_result_kalman, successful_kalman = kalman_pipeline.run(frame_x.copy())
    if not successful_kalman:
        kalman_estimations.append(run_result_kalman)
    else:
        prediction_kalman, correction_kalman = run_result_kalman
        kalman_estimations.append(correction_kalman.flatten()[:2])

    # PF results

    run_result_pf, successful_pf = pf_pipeline.run(frame_x.copy())
    if not successful_pf:
        pf_estimations.append(run_result_pf)
    else:
        centroid_pf = run_result_pf
        pf_estimations.append(centroid_pf)

cv2.destroyAllWindows()
x_cap.release()
y_cap.release()

x_kalman = np.array(kalman_estimations)
x_pf = np.array(pf_estimations)
x = np.stack((x_kalman, x_pf))
y = np.array(gt_estimations)

pipeline_names = ['Kalman', 'PF']
evaluator = PipelineEvaluator(x, y, pipeline_names)
evaluator.plot_evaluation(save=False, sample_rate=60)
