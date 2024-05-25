from abc import ABC, abstractmethod
import cv2
import numpy as np
from src.particle_filter.particle_filter import ParticleFilter


class Step(ABC):
    """
    Abstract base class for pipeline steps.

    Attributes:
    -----------
    step_name : str
        Name of the step.
    debug : bool
        If True, enables debug mode.
    """

    def __init__(self, step_name: str, debug: bool = False):
        self.step_name = step_name
        self.debug = debug

    @abstractmethod
    def run(self, inputs):
        """
        Abstract method to run the step.

        Parameters:
        -----------
        inputs : any
            Inputs to the step.

        Returns:
        --------
        any
            Outputs from the step.
        """
        pass

    def debug_step(self, image):
        """
        Displays the image for debugging purposes.

        Parameters:
        -----------
        image : np.array
            The image to display.
        """
        cv2.imshow(self.step_name, image)
        cv2.waitKey(0)


class BackgroundSubtractionStep(Step):
    """
    Step for performing background subtraction.

    Attributes:
    -----------
    bs_alg : object
        Background subtraction algorithm.
    """

    def __init__(self, step_name: str, bs_alg, debug: bool = False):
        super().__init__(step_name, debug)
        self.bs_alg = bs_alg

    def run(self, inputs):
        """
        Applies background subtraction to the input frame.

        Parameters:
        -----------
        inputs : np.array
            Input frame.

        Returns:
        --------
        np.array
            Background subtracted image.
        """
        frame = inputs
        background_image = self.bs_alg.apply(frame)
        if self.debug:
            self.debug_step(background_image)
        return background_image


class BoundingBoxStep(Step):
    """
    Step for detecting bounding boxes.

    Methods:
    --------
    __filter_bs(bs_image):
        Filters the background subtracted image.
    """

    @staticmethod
    def __filter_bs(bs_image: np.array) -> np.array:
        """
        Filters the background subtracted image.

        Parameters:
        -----------
        bs_image : np.array
            Background subtracted image.

        Returns:
        --------
        np.array
            Filtered image.
        """
        _, threshold = cv2.threshold(bs_image, 240, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)
        bs_filtered = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        return bs_filtered

    def __init__(self, step_name: str, debug: bool = False):
        super().__init__(step_name, debug)

    def run(self, inputs):
        """
        Detects bounding boxes in the filtered image.

        Parameters:
        -----------
        inputs : np.array
            Filtered image.

        Returns:
        --------
        tuple or None
            Bounding box coordinates or None if no contours found.
        """
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
    """
    Step for applying a Kalman filter.

    Attributes:
    -----------
    x : np.array
        State vector.
    P : np.array
        Covariance matrix.
    A : np.array
        State transition matrix.
    H : np.array
        Measurement matrix.
    R : np.array
        Measurement noise covariance matrix.
    Q : np.array
        Process noise covariance matrix.
    """

    def __init__(self, step_name: str,
                 x: np.array = np.array([[0, 0, 0, 0]]).T,
                 P: np.array = np.eye(4),
                 A: np.array = np.array([[1, 0, 1, 0],
                                         [0, 1, 0, 1],
                                         [0, 0, 1, 0],
                                         [0, 0, 0, 1]]),
                 H: np.array = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]]),
                 R: np.array = np.eye(2) * 15,
                 Q: np.array = np.eye(4) * 0.001,
                 debug: bool = False):
        """
        Initializes the KalmanFilterStep.

        Parameters:
        -----------
        step_name : str
            Name of the step.
        x : np.array, optional
            Initial state vector. Defaults to [[0, 0, 0, 0]].T.
        P : np.array, optional
            Initial covariance matrix. Defaults to np.eye(4).
        A : np.array, optional
            State transition matrix. Defaults to a predefined matrix.
        H : np.array, optional
            Measurement matrix. Defaults to a predefined matrix.
        R : np.array, optional
            Measurement noise covariance matrix. Defaults to np.eye(2) * 15.
        Q : np.array, optional
            Process noise covariance matrix. Defaults to np.eye(4) * 0.001.
        debug : bool, optional
            If True, enables debug mode. Defaults to False.
        """
        super().__init__(step_name, debug)
        self.x = x
        self.P = P
        self.A = A
        self.H = H
        self.R = R
        self.Q = Q

    def run(self, inputs):
        """
        Applies the Kalman filter to the inputs.

        Parameters:
        -----------
        inputs : np.array
            Measurements.

        Returns:
        --------
        tuple
            Prediction and estimation.
        """
        z = inputs
        z = np.array([z]).T
        prediction = self.__predict()
        estimation = self.__update(z)
        return prediction, estimation

    def __predict(self) -> np.array:
        """
        The prediction step.

        Returns:
        --------
        np.array
            Predicted state.
        """
        self.x = self.A @ self.x
        self.P = self.A @ (self.P @ self.A.T) + self.Q
        return self.x

    def __update(self, z: np.array) -> np.array:
        """
        The update or correction step.

        Parameters:
        -----------
        z : np.array
            Measurement vector.

        Returns:
        --------
        np.array
            Updated state estimate.
        """
        K = self.P @ self.H.T @ np.linalg.inv((self.H @ self.P @ self.H.T) + self.R)
        self.x = self.x + K @ (z - self.H @ self.x)
        I = np.eye(self.H.shape[1])
        self.P = (I - K @ self.H) @ self.P
        return self.x


class PFStep(Step):
    """
    Step for applying a Particle Filter.

    Attributes:
    -----------
    pf : ParticleFilter
        Particle filter object.
    """

    def __init__(self, num_particles: int, particle_shape: (int, int), image_shape: (int, int),
                 step_name: str, debug: bool = False):
        """
        Initializes the PFStep.

        Parameters:
        -----------
        num_particles : int
            Number of particles.
        particle_shape : tuple
            Shape of each particle.
        image_shape : tuple
            Shape of the input image.
        step_name : str
            Name of the step.
        debug : bool, optional
            If True, enables debug mode. Defaults to False.
        """
        super().__init__(step_name, debug)
        self.pf = ParticleFilter(num_particles, particle_shape, image_shape)

    def run(self, inputs):
        """
        Applies the particle filter to the inputs.

        Parameters:
        -----------
        inputs : np.array
            Foreground image.

        Returns:
        --------
        tuple
            Bounding box of the tracked object.
        """
        bs_foreground = inputs
        return self.pf.track(bs_foreground)


class CentroidStep(Step):
    """
    Step for calculating the centroid of a bounding box.
    """

    def __init__(self, step_name: str, debug: bool = False):
        super().__init__(step_name, debug)

    def run(self, inputs):
        """
        Calculates the centroid of a bounding box.

        Parameters:
        -----------
        inputs : tuple
            Bounding box coordinates (x, y, w, h).

        Returns:
        --------
        tuple
            Centroid coordinates (x, y).
        """
        x, y, w, h = inputs
        centroid_x = x + w / 2
        centroid_y = y + h / 2
        return centroid_x, centroid_y


class BSCentroidStep(Step):
    """
    Step for calculating the centroid of the largest contour in a binary image.
    """

    def __init__(self, step_name: str, debug: bool = False):
        super().__init__(step_name, debug)

    def run(self, inputs, epsilon=1e-20):
        """
        Calculates the centroid of the largest contour in a binary image.

        Parameters:
        -----------
        inputs : np.array
            Binary image.
        epsilon : float, optional
            Small value to avoid division by zero. Defaults to 1e-20.

        Returns:
        --------
        tuple
            Centroid coordinates (x, y).
        """
        subtraction_image = inputs
        contours, _ = cv2.findContours(subtraction_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return 0, 0
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)
        centroid_x = int(M['m10'] / M['m00'] + epsilon)
        centroid_y = int(M['m01'] / M['m00'] + epsilon)

        return centroid_x, centroid_y


class MeanShiftStep(Step):
    """
    Step for tracking objects using the MeanShift algorithm.

    Attributes:
    -----------
    track_window : tuple
        Initial tracking window.
    roi_hist : np.array
        Histogram of the region of interest.
    term_crit : tuple
        Termination criteria.
    """

    def __init__(self, init_frame: np.array, init_window: (int, int, int, int),
                 step_name: str, debug: bool = False):
        """
        Initializes the MeanShiftStep.

        Parameters:
        -----------
        init_frame : np.array
            Initial frame.
        init_window : tuple
            Initial tracking window.
        step_name : str
            Name of the step.
        debug : bool, optional
            If True, enables debug mode. Defaults to False.
        """
        super().__init__(step_name, debug)
        self.track_window = init_window
        x, y, w, h = init_window
        self.roi = init_frame[y:y + h, x:x + w]
        self.hsv_roi = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(self.hsv_roi, np.array((124, 41, 0)), np.array((162, 166, 135)))
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.roi_hist = cv2.calcHist([self.hsv_roi], [0], self.mask, [180], [0, 180])

    def run(self, inputs):
        """
        Applies the MeanShift algorithm to track the object.

        Parameters:
        -----------
        inputs : np.array
            Current frame.

        Returns:
        --------
        tuple
            Updated tracking window.
        """
        frame = inputs
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        _, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)
        return self.track_window


class CamShiftStep(Step):
    """
    Step for tracking objects using the CamShift algorithm.

    Attributes:
    -----------
    track_window : tuple
        Initial tracking window.
    roi_hist : np.array
        Histogram of the region of interest.
    term_crit : tuple
        Termination criteria.
    """

    def __init__(self, init_frame: np.array, init_window: (int, int, int, int),
                 step_name: str, debug: bool = False):
        """
        Initializes the CamShiftStep.

        Parameters:
        -----------
        init_frame : np.array
            Initial frame.
        init_window : tuple
            Initial tracking window.
        step_name : str
            Name of the step.
        debug : bool, optional
            If True, enables debug mode. Defaults to False.
        """
        super().__init__(step_name, debug)
        self.track_window = init_window
        x, y, w, h = init_window
        self.roi = init_frame[y:y + h, x:x + w]
        self.hsv_roi = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV)
        self.mask = cv2.inRange(self.hsv_roi, np.array((124, 41, 0)), np.array((162, 166, 135)))
        self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        self.roi_hist = cv2.calcHist([self.hsv_roi], [0], self.mask, [180], [0, 180])

    def run(self, inputs):
        """
        Applies the CamShift algorithm to track the object.

        Parameters:
        -----------
        inputs : np.array
            Current frame.

        Returns:
        --------
        tuple
            Updated tracking window.
        """
        frame = inputs
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        _, self.track_window = cv2.CamShift(dst, self.track_window, self.term_crit)
        return self.track_window
