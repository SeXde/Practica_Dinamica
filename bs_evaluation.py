import os
import cv2
import numpy as np
from tqdm import tqdm

from src.constants import VIDEOS_PATH, BS_PATH, VIDEO_NAME
from src.evaluator.bs_evaluator import BSEvaluator

# Initialize video captures for input video and ground truth background subtraction
x_cap = cv2.VideoCapture(os.path.join(VIDEOS_PATH, VIDEO_NAME))
y_cap = cv2.VideoCapture(os.path.join(BS_PATH, VIDEO_NAME))

# Initialize background subtraction algorithms
bs_algorithms = [
    cv2.createBackgroundSubtractorKNN(),
    cv2.createBackgroundSubtractorMOG2(),
    cv2.bgsegm.createBackgroundSubtractorCNT(),
    cv2.bgsegm.createBackgroundSubtractorGSOC(),
]

# Parameters
num_algorithms = len(bs_algorithms)
max_frames = 1000
height, width = 501, 500

# Initialize arrays to store background subtraction results and ground truth
bs_results = np.zeros((num_algorithms, max_frames, height, width), np.uint8)
bs_ground_truth = np.zeros((max_frames, height, width), np.uint8)

# Process frames from the video
for frame_count in tqdm(range(max_frames), desc="Evaluating frames"):
    if not (x_cap.isOpened() and y_cap.isOpened()):
        break

    ret_x, frame_x = x_cap.read()
    ret_y, frame_y = y_cap.read()
    if not ret_x or not ret_y:
        break

    # Process ground truth frame
    frame_y = cv2.cvtColor(frame_y, cv2.COLOR_BGR2GRAY)
    frame_y = cv2.resize(frame_y, (width, height))
    frame_y[frame_y != 255] = 0  # Convert shadows to 0
    bs_ground_truth[frame_count, :, :] = np.copy(frame_y)

    # Process input frame for each background subtraction algorithm
    frame_x = cv2.resize(frame_x, (width, height))
    for i, bs in enumerate(bs_algorithms):
        fg = bs.apply(frame_x.copy())
        fg[fg != 255] = 0  # Convert shadows to 0
        bs_results[i, frame_count, :, :] = np.copy(fg)

# Release video captures
x_cap.release()
y_cap.release()

# Background subtraction algorithm names
bs_names = ['KNN', 'MOG2', 'CNT', 'GSOC']

# Evaluate and plot results
evaluator = BSEvaluator(bs_results, bs_ground_truth, bs_names)
evaluator.plot_evaluation(save=False)
