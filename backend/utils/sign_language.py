import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

def fingers_up(landmarks):
    fingers = []

    # Thumb: check x (since horizontal for thumb)
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers: check y (vertical)
    for tip_id in [8, 12, 16, 20]:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def recognize_sign(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    sign = "Unknown"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm = handLms.landmark
            fingers = fingers_up(lm)

            if fingers == [0, 1, 0, 0, 0]:
                sign = "1"
            elif fingers == [0, 1, 1, 0, 0]:
                sign = "V"
            elif fingers == [1, 1, 1, 1, 1]:
                sign = "Open Palm"
            elif fingers == [0, 0, 0, 0, 0]:
                sign = "Fist"
            else:
                sign = "Other"

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    return sign
