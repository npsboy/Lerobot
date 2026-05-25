import cv2

for i in range(4):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    ok, frame = cap.read()

    print(i, ok, frame.shape if ok else None)

    cap.release()