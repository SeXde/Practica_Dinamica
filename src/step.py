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


class KalmanFilterStep(Step):
    def run(self, inputs):
        z = inputs
        z = np.array([z]).T
        prediction = self.__predict()
        estimation = self.__update(z)
        return prediction, estimation

    def __init__(self,
                 step_name: str,
                 x: np.array = np.array([[0, 0, 0, 0]]).T,
                 P: np.array = np.eye(4),
                 A: np.array = np.array([[1, 0, 1, 0],
                                         [0, 1, 0, 1],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]]),
                 H: np.array = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]]),
                 R: np.array = np.eye(2) * 0.01,
                 Q: np.array = np.eye(4) * 0.1,
                 debug: bool = False):
        """
        Initializes the KalmanFilter.

        Parameters:
            - x (np.array): the state of the system.
            - P (np.array): the covariance matrix of the estimation error.
            - A (np.array): the state transition matrix.
            - H (np.array): the measurement matrix.
            - R (np.array): the covariance matrix of the measurement noise.
            - Q (np.array): the covariance matrix of the process noise.
        """
        super().__init__(step_name, debug)
        self.x = x
        self.P = P
        self.A = A
        self.H = H
        self.R = R
        self.Q = Q

    def __predict(self) -> np.array:
        """
        The prediction step

        Returns:
            - x (np.array): the predicted state.
        """
        self.x = self.A @ self.x
        self.P = self.A @ (self.P @ self.A.T) + self.Q
        return self.x

    def __update(self, z: np.array) -> np.array:
        """
        The update or correction step

        Parameters:
            - z (np.array): the measurement.

        Returns:
            - x (np.array): the final estimate of the state.
        """
        K = self.P @ self.H.T @ np.linalg.inv((self.H @ self.P @ self.H.T) + self.R)
        self.x = self.x + K @ (z - self.H @ self.x)
        I = np.eye(self.H.shape[1])
        self.P = (I - K @ self.H) @ self.P
        return self.x


class CentroidStep(Step):

    def __init__(self, step_name: str, debug: bool = False):
        super().__init__(step_name, debug)

    def run(self, inputs):
        x, y, w, h = inputs
        centroid_x = x + w / 2
        centroid_y = y + h / 2
        return centroid_x, centroid_y


class BSCentroidStep(Step):

    def __init__(self, step_name: str, debug: bool = False):
        super().__init__(step_name, debug)

    def run(self, inputs, epsilon=1e-20):
        subtraction_image = inputs
        contours, _ = cv2.findContours(subtraction_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return 0, 0
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)
        centroid_x = int(M['m10'] / M['m00'] + epsilon)
        centroid_y = int(M['m01'] / M['m00'] + epsilon)

        return centroid_x, centroid_y
