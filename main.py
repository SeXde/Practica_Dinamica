import cv2
import numpy as np


def filter_bs(bs_image: np.array) -> np.array:
    _, threshold = cv2.threshold(bs_image, 240, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    return closing


bs = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
video_file = "video/Walking.54138969.mp4"

cap = cv2.VideoCapture(video_file)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        bs_output = bs.apply(frame)
        mask = filter_bs(bs_output)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            all_contour_points = np.concatenate(contours)
            hull = cv2.convexHull(all_contour_points)
            cv2.drawContours(frame, [hull], -1, (0, 255, 0), 3)

            # Bounding box
            # x, y, w, h = cv2.boundingRect(hull)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Original video', frame)
        cv2.imshow('Foreground Mask', mask)

    if cv2.waitKey(1) & 0xFF == 27:  # esc
        print('Exiting...')
        break

cv2.destroyAllWindows()
cap.release()
