import cv2
import pybgs as bgs

algorithm = bgs.VuMeter()
video_file = "video/Walking.54138969.mp4"

cap = cv2.VideoCapture(video_file)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        output = algorithm.apply(frame)
        cv2.imshow('Original video', frame)
        cv2.imshow('Foreground Mask', output)

    if cv2.waitKey(1) & 0xFF == 27:  # esc
        print('Exiting...')
        break

cv2.destroyAllWindows()
cap.release()
