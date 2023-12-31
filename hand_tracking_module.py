import cv2
import mediapipe as mp


class HandTracker:
    def __init__(self):
        self.results = None
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, landmark_ids=None):
        ret, frame = self.cap.read()
        if not ret:
            return None

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                # Draw all landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                if landmark_ids is not None:
                    for id_, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if id_ in landmark_ids:
                            cv2.circle(frame, (cx, cy), 25, (255, 0, 255), cv2.FILLED)

        return frame

    def find_coordinate(self, frame, handNo=0, draw=True, landmark_id=None):
        landmarks_list = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[handNo]
            for id_, lm in enumerate(hand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks_list.append((cx, cy))
                if draw:
                    cv2.circle(frame, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
        return landmark_id, landmarks_list[landmark_id] if len(landmarks_list) > 0 else []

    def release(self):
        self.cap.release()
