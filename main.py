import os
import cv2
import numpy as np

from src.constants import VIDEOS_PATH, BS_PATH, VIDEO_NAME, INIT_FRAME_NAME
from src.evaluator.pipeline_evaluator import PipelineEvaluator
from src.pipeline import Pipeline
from src.step import (
    BackgroundSubtractionStep, BoundingBoxStep, CentroidStep,
    KalmanFilterStep, BSCentroidStep, PFStep, MeanShiftStep, CamShiftStep
)

# Initialize tracking pipelines
kalman_pipeline = Pipeline(
    'Kalman pipeline',
    [
        BackgroundSubtractionStep('BS Step', cv2.createBackgroundSubtractorMOG2()),
        BoundingBoxStep('BBOX step'),
        CentroidStep('Centroid step'),
        KalmanFilterStep('Kalman filter step')
    ]
)

pf_pipeline = Pipeline(
    'Particle filter pipeline',
    [
        BackgroundSubtractionStep('BS Step', cv2.createBackgroundSubtractorMOG2()),
        PFStep(100, (300, 300), (1002, 1000), 'Particle filter step'),
        CentroidStep('Centroid step')
    ]
)

ms_pipeline = Pipeline(
    'Mean Shift pipeline',
    [
        MeanShiftStep(
            init_frame=cv2.imread(os.path.join(VIDEOS_PATH, INIT_FRAME_NAME)),
            init_window=(430, 420, 100, 70), step_name='Mean Shift step'
        ),
        CentroidStep('Centroid step')
    ]
)

cs_pipeline = Pipeline(
    'Cam Shift pipeline',
    [
        CamShiftStep(
            init_frame=cv2.imread(os.path.join(VIDEOS_PATH, INIT_FRAME_NAME)),
            init_window=(430, 420, 100, 70), step_name='Cam Shift step'
        ),
        CentroidStep('Centroid step')
    ]
)

gt_pipeline = Pipeline(
    'GT pipeline',
    [
        BSCentroidStep('BS Centroid step')
    ]
)

# Open video captures for input video and ground truth video
x_cap = cv2.VideoCapture(os.path.join(VIDEOS_PATH, VIDEO_NAME))
y_cap = cv2.VideoCapture(os.path.join(BS_PATH, VIDEO_NAME))

# Lists to store estimations from different pipelines
kalman_estimations = []
pf_estimations = []
gt_estimations = []
ms_estimations = []
cs_estimations = []

# Process video frames
while x_cap.isOpened() and y_cap.isOpened():
    ret_x, frame_x = x_cap.read()
    ret_y, frame_y = y_cap.read()
    if not ret_x or not ret_y:
        break

    # Ground truth results
    frame_y = cv2.cvtColor(frame_y, cv2.COLOR_BGR2GRAY)
    _, frame_y = cv2.threshold(frame_y, 1, 255, cv2.THRESH_BINARY)
    centroid, _ = gt_pipeline.run(frame_y)
    gt_estimations.append(centroid)

    # Kalman filter results
    run_result_kalman, successful_kalman = kalman_pipeline.run(frame_x.copy())
    if successful_kalman:
        prediction_kalman, correction_kalman = run_result_kalman
        kalman_estimations.append(correction_kalman.flatten()[:2])
    else:
        kalman_estimations.append(run_result_kalman)

    # Particle filter results
    run_result_pf, successful_pf = pf_pipeline.run(frame_x.copy())
    pf_estimations.append(run_result_pf if successful_pf else run_result_pf)

    # Mean Shift results
    run_result_ms, successful_ms = ms_pipeline.run(frame_x.copy())
    ms_estimations.append(run_result_ms if successful_ms else run_result_ms)

    # Cam Shift results
    run_result_cs, successful_cs = cs_pipeline.run(frame_x.copy())
    cs_estimations.append(run_result_cs if successful_cs else run_result_cs)

# Release video captures and close any open windows
cv2.destroyAllWindows()
x_cap.release()
y_cap.release()

# Convert estimations to numpy arrays
x_kalman = np.array(kalman_estimations)
x_pf = np.array(pf_estimations)
x_ms = np.array(ms_estimations)
x_cs = np.array(cs_estimations)
x = np.stack((x_kalman, x_pf, x_ms, x_cs))
y = np.array(gt_estimations)

# Evaluate and plot results
pipeline_names = ['Kalman', 'PF', 'MShift', 'CShift']
evaluator = PipelineEvaluator(x, y, pipeline_names)
evaluator.plot_evaluation(save=False, sample_rate=60)
