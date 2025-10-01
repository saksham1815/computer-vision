from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2, base64, numpy as np
from ultralytics import YOLO
import face_recognition

# YOLO for object detection
yolo_model = YOLO("yolov8n.pt")

# Haar Cascade for fallback face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Utils safe imports
try:
    from utils.hand_utils import detect_hands
    from utils.pose_utils import detect_pose
    from utils.sign_language import recognize_sign
    from utils.draw_utils import air_draw
    from utils.mouse_control import control_mouse
except ImportError:
    def detect_hands(frame): return frame
    def detect_pose(frame): return frame
    def recognize_sign(frame): return "Unknown"
    def air_draw(frame, points): return frame, points
    def control_mouse(frame): return None

app = Flask(__name__)
CORS(app)

draw_points = []
mode = "gesture"
reference_encoding = None  # will hold uploaded face encoding

def decode_image(img_base64):
    if "," in img_base64:
        img_base64 = img_base64.split(",")[1]
    img_data = base64.b64decode(img_base64)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

@app.route("/upload_face", methods=["POST"])
def upload_face():
    """Upload reference image for face recognition"""
    global reference_encoding
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    encodings = face_recognition.face_encodings(img)
    if len(encodings) > 0:
        reference_encoding = encodings[0]
        return jsonify({"status": "Reference face stored"})
    else:
        return jsonify({"error": "No face found in uploaded image"}), 400

@app.route("/process", methods=["POST"])
def process():
    global mode, draw_points, reference_encoding
    try:
        data = request.json
        frame = decode_image(data["image"])
        mode = data.get("mode", mode)

        detected_labels = []

        if mode == "gesture":
            frame = detect_hands(frame)

        elif mode == "sign":
            sign = recognize_sign(frame)
            cv2.putText(frame, f"Sign: {sign}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        elif mode == "draw":
            frame, draw_points = air_draw(frame, draw_points)

        elif mode == "mouse":
            control_mouse(frame)

        elif mode == "pose":
            frame = detect_pose(frame)

        elif mode == "object":
            results = yolo_model(frame)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = yolo_model.names[cls_id]
                    detected_labels.append(label)

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        elif mode == "face":
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                label = "Unknown"
                if reference_encoding is not None:
                    matches = face_recognition.compare_faces([reference_encoding], face_encoding)
                    if matches[0]:
                        label = "Matched"

                detected_labels.append(label)
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(frame, label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return jsonify({
            "image": encode_image(frame),
            "detected": detected_labels
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Flask server is running! Use POST /process"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
