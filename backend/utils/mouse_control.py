import cv2
import mediapipe as mp
import pyautogui
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
screen_w, screen_h = pyautogui.size()

def control_mouse(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            index_tip = handLms.landmark[8]
            thumb_tip = handLms.landmark[4]

            # Cursor movement
            screen_x = screen_w * index_tip.x
            screen_y = screen_h * index_tip.y
            pyautogui.moveTo(screen_x, screen_y, duration=0.05)

            # Calculate distance
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
            dist = math.hypot(tx - ix, ty - iy)

            if dist < 40:
                pyautogui.click()
                cv2.putText(frame, "Click!", (ix, iy - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame
