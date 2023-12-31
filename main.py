import cv2

from hand_tracking_module import HandTracker

hand_tracker = HandTracker()

WCam, HCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, WCam)
cap.set(4, HCam)
hand_tracker.cap = cap


while True:
    frame = hand_tracker.process_frame()
    coords = hand_tracker.find_coordinate(frame, landmark_id=4)
    print(coords)
    if frame is not None:
        cv2.imshow('frame', frame)
        cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


