import cv2
import numpy as np


class PersonDetector:
    def __init__(self):
        self.bs = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.bs_raw = None
        self.bs_filtered = None

    def detect(self, frame: np.array) -> (int, int, int, int):
        self.bs_raw = self.bs.apply(frame)
        self.bs_filtered = self.__filter_bs(self.bs_raw)
        contours, _ = cv2.findContours(self.bs_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            all_contour_points = np.concatenate(contours)
            hull = cv2.convexHull(all_contour_points)
            # cv2.drawContours(frame, [hull], -1, (0, 255, 0), 3)
            return cv2.boundingRect(hull)
        return None

    @staticmethod
    def __filter_bs(bs_image: np.array) -> np.array:
        _, threshold = cv2.threshold(bs_image, 240, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)
        bs_filtered = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        return bs_filtered

