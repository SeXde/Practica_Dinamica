import numpy as np
import cv2


def centroid_from_bbox(bbox: (int, int, int, int)) -> (int, int):
    x, y, w, h = bbox
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy


def draw_bbox(frame: np.array, bbox: (int, int, int, int)):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
