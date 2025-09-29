import cv2
import mediapipe as mp
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

moving = False
last_hand_pos = None

# ------------------------
# Global Drawing Settings
# ------------------------
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red
current_color = colors[0]
brush_thickness = 5


def fingers_up(landmarks):
    fingers = []
    # Thumb
    if landmarks[4].x < landmarks[3].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other fingers
    for tip_id in [8, 12, 16, 20]:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers


def air_draw(frame, draw_points):
    global moving, last_hand_pos, current_color, brush_thickness

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    h, w, _ = frame.shape

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lm = handLms.landmark
            fingers = fingers_up(lm)

            index_tip = lm[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            # ✋ Clear board
            if fingers == [1, 1, 1, 1, 1]:
                draw_points.clear()
                cv2.putText(frame, "Erased!", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                moving = False
                last_hand_pos = None

            # ✌️ Move mode
            elif fingers == [0, 1, 1, 0, 0]:
                cx, cy = int(lm[8].x * w), int(lm[8].y * h)
                if not moving:
                    moving = True
                    last_hand_pos = (cx, cy)
                else:
                    dx = cx - last_hand_pos[0]
                    dy = cy - last_hand_pos[1]
                    # ✅ Shift all points correctly
                    draw_points[:] = [
                        (px + dx, py + dy, color, thickness)
                        for (px, py, color, thickness) in draw_points
                    ]
                    last_hand_pos = (cx, cy)
                cv2.putText(frame, "Move Mode", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                moving = False
                last_hand_pos = None

            # 🖌️ Color switching
            if fingers == [0, 1, 1, 0, 0]:
                current_color = colors[0]
            elif fingers == [0, 1, 1, 1, 0]:
                current_color = colors[1]
            elif fingers == [0, 1, 1, 1, 1]:
                current_color = colors[2]

            # 🎚️ Brush thickness from thumb-index distance
            ix, iy = int(lm[4].x * w), int(lm[4].y * h)
            dist = math.hypot(x - ix, y - iy)
            brush_thickness = max(2, min(20, int(dist / 5)))

            # ✍️ Draw with index finger
            if fingers == [0, 1, 0, 0, 0]:
                draw_points.append((x, y, current_color, brush_thickness))
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Draw strokes
    for i in range(1, len(draw_points)):
        cv2.line(frame,
                 draw_points[i - 1][:2],  # (x, y)
                 draw_points[i][:2],      # (x, y)
                 draw_points[i][2],       # color
                 draw_points[i][3])       # thickness

    return frame, draw_points
