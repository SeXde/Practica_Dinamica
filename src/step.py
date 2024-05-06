from abc import ABC, abstractmethod
import cv2
import numpy as np


class Step(ABC):
    def __init__(self, step_name: str, debug: bool = False):
        self.step_name = step_name
        self.debug = debug

    @abstractmethod
    def run(self, inputs):
        pass

    def debug_step(self, image):
        cv2.imshow(self.step_name, image)
        cv2.waitKey(0)


class BackgroundSubtractionStep(Step):

    def __init__(self, step_name: str, bs_alg, debug: bool = False):
        super().__init__(step_name, debug)
        self.bs_alg = bs_alg

    def run(self, inputs):
        frame = inputs
        background_image = self.bs_alg.apply(frame)
        if self.debug:
            self.debug_step(background_image)
        return background_image


class BoundingBoxStep(Step):

    @staticmethod
    def __filter_bs(bs_image: np.array) -> np.array:
        _, threshold = cv2.threshold(bs_image, 240, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)
        bs_filtered = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        return bs_filtered

    def __init__(self, step_name: str, debug: bool = False):
        super().__init__(step_name, debug)

    def run(self, inputs):
        bs_raw = inputs
        bs_filtered = self.__filter_bs(bs_raw)
        contours, _ = cv2.findContours(bs_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            all_contour_points = np.concatenate(contours)
            hull = cv2.convexHull(all_contour_points)
            if self.debug:
                debug_image = bs_raw.copy()
                cv2.drawContours(debug_image, [hull], -1, (0, 255, 0), 3)
                self.debug_step(debug_image)
            return cv2.boundingRect(hull)
        return None
