import numpy as np
import cv2


def centroid_from_bbox(bbox: (int, int, int, int)) -> (int, int):
    """
    Calculates the centroid of a bounding box.

    Parameters:
    -----------
    bbox : tuple
        Bounding box coordinates (x, y, w, h).

    Returns:
    --------
    tuple
        Centroid coordinates (cx, cy).
    """
    x, y, w, h = bbox
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy


def draw_bbox(frame: np.array, bbox: (int, int, int, int)):
    """
    Draws a bounding box on a frame.

    Parameters:
    -----------
    frame : np.array
        The frame on which to draw the bounding box.
    bbox : tuple
        Bounding box coordinates (x, y, w, h).
    """
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
