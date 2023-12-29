import cv2

from hand_tracking_module import HandTracker

hand_tracker = HandTracker()

while True:
    frame = hand_tracker.process_frame()
    if frame is not None:
        cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hand_tracker.release()
cv2.destroyAllWindows()
