import cv2


def fill_holes(mask):
    h, w = mask.shape[:2]
    for row in range(h):
        if mask[row, 0] == 255:
            cv2.floodFill(mask, None, (0, row), 0)
        if mask[row, w - 1] == 255:
            cv2.floodFill(mask, None, (w - 1, row), 0)
    for col in range(w):
        if mask[0, col] == 255:
            cv2.floodFill(mask, None, (col, 0), 0)
        if mask[h - 1, col] == 255:
            cv2.floodFill(mask, None, (col, h - 1), 0)

    # flood fill background to find inner holes
    holes = mask.copy()
    cv2.floodFill(holes, None, (0, 0), 255)
    holes = cv2.bitwise_not(holes)
    mask = cv2.bitwise_or(mask, holes)
    return mask


bs = cv2.createBackgroundSubtractorMOG2()
video_file = "video/Walking.54138969.mp4"

cap = cv2.VideoCapture(video_file)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        mask = bs.apply(frame)
        holes = fill_holes(mask.copy())
        cv2.imshow('Original video', frame)
        cv2.imshow('Foreground Mask', mask)
        cv2.imshow('Foreground Holes', holes)

    if cv2.waitKey(1) & 0xFF == 27:  # esc
        print('Exiting...')
        break

cv2.destroyAllWindows()
cap.release()
