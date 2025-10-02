from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2, base64, numpy as np, os, time
from ultralytics import YOLO
import face_recognition

# ---------------- Models ----------------
yolo_model = YOLO("yolov8n.pt")  # YOLO for object detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------------- Utils (safe imports) ----------------
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

# ---------------- Flask Setup ----------------
app = Flask(__name__)
CORS(app)

draw_points = []
mode = "gesture"
reference_encoding = None  # uploaded face encoding
last_snapshot_time = 0
os.makedirs("snapshots", exist_ok=True)

# ---------------- Helper Functions ----------------
def decode_image(img_base64):
    if "," in img_base64:
        img_base64 = img_base64.split(",")[1]
    img_data = base64.b64decode(img_base64)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

def make_sketch(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 150)
    sketch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return sketch

# ---------------- Routes ----------------
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
    global mode, draw_points, reference_encoding, last_snapshot_time
    try:
        data = request.json
        frame = decode_image(data["image"])
        mode = data.get("mode", mode)

        detected_labels = []
        snapshot_b64, sketch_b64 = None, None

        # ---------------- Gesture ----------------
        if mode == "gesture":
            frame = detect_hands(frame)

        # ---------------- Sign ----------------
        elif mode == "sign":
            sign = recognize_sign(frame)
            cv2.putText(frame, f"Sign: {sign}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # ---------------- Draw ----------------
        elif mode == "draw":
            frame, draw_points = air_draw(frame, draw_points)

        # ---------------- Mouse ----------------
        elif mode == "mouse":
            control_mouse(frame)

        # ---------------- Pose ----------------
        elif mode == "pose":
            frame = detect_pose(frame)

        # ---------------- Object Detection ----------------
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

        # ---------------- Face Detection ----------------
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

                        # Snapshot + sketch every 2 seconds
                        if time.time() - last_snapshot_time > 2:
                            last_snapshot_time = time.time()
                            face_crop = frame[top:bottom, left:right]
                            if face_crop.size > 0:
                                filename = f"snapshots/match_{int(time.time())}.jpg"
                                cv2.imwrite(filename, face_crop)
                                snapshot_b64 = encode_image(face_crop)
                                sketch_b64 = encode_image(make_sketch(face_crop))

                detected_labels.append(label)
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                cv2.putText(frame, label, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        return jsonify({
            "image": encode_image(frame),
            "detected": detected_labels,
            "snapshot": snapshot_b64,
            "sketch": sketch_b64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Flask server is running! Use POST /process"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
