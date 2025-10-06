from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2, base64, numpy as np, os, time
from ultralytics import YOLO
import face_recognition
from deepface import DeepFace
import easyocr

reader = easyocr.Reader(['en'])
yolo_model = YOLO("yolov8n.pt")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

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
reference_encoding = None
target_object = None
last_detection_time = 0
last_detected_label = None
SAVE_DIR = "snapshots"
os.makedirs(SAVE_DIR, exist_ok=True)
last_face_results = {"locations": [], "labels": [], "snapshot": None, "sketch": None}

def decode_image(img_base64):
    if "," in img_base64:
        img_base64 = img_base64.split(",")[1]
    img_data = base64.b64decode(img_base64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img if img is not None else np.zeros((480, 640, 3), dtype=np.uint8)

def encode_image(img):
    _, buffer = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

def sketch_face(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

@app.route("/upload_face", methods=["POST"])
def upload_face():
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
    return jsonify({"error": "No face found in uploaded image"}), 400

@app.route("/process", methods=["POST"])
def process():
    global mode, draw_points, reference_encoding, last_detection_time, last_detected_label, target_object, last_face_results

    try:
        data = request.json
        frame = decode_image(data["image"])
        latitude = data.get("latitude")
        longitude = data.get("longitude")
        geo_info = {"latitude": latitude, "longitude": longitude} if latitude and longitude else None

        if "mode" in data:
            mode = data["mode"]

        detected_labels = []
        snapshot_b64, sketch_b64, object_snapshot_b64 = None, None, None

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
            target_object = data.get("object_name", "").strip().lower() or target_object
            results = yolo_model(frame)
            now = time.time()

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = yolo_model.names[cls_id].lower()
                    detected_labels.append(label)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    color = (0, 255, 0) if (target_object and label == target_object) else (255, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        elif mode == "emotion":
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                cv2.putText(frame, f"Emotion: {emotion}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                detected_labels.append(emotion)
            except Exception as e:
                detected_labels.append("Error")

        elif mode == "ocr":
            try:
                results = reader.readtext(frame)
                extracted_texts = [t for (_, t, _) in results]
                for (bbox, text, conf) in results:
                    (tl, tr, br, bl) = bbox
                    tl, br = tuple(map(int, tl)), tuple(map(int, br))
                    cv2.rectangle(frame, tl, br, (0, 255, 0), 2)
                    cv2.putText(frame, text, (tl[0], tl[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                detected_labels.extend(extracted_texts or ["No text found"])
            except Exception as e:
                detected_labels.append("OCR Error")

        elif mode == "face":
            now = time.time()
            if now - last_detection_time > 0.5:
                small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb)
                encs = face_recognition.face_encodings(rgb, locs)
                new_labels = []
                for (top, right, bottom, left), enc in zip(locs, encs):
                    top, right, bottom, left = [v * 4 for v in (top, right, bottom, left)]
                    label = "Unknown"
                    if reference_encoding is not None:
                        match = face_recognition.compare_faces([reference_encoding], enc)
                        if match[0]:
                            label = "Matched"
                            face_crop = frame[top:bottom, left:right]
                            snapshot_b64 = encode_image(face_crop)
                            sketch_b64 = encode_image(sketch_face(face_crop))
                    new_labels.append(label)
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                    cv2.putText(frame, label, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                last_face_results = {"locations": locs, "labels": new_labels,
                                     "snapshot": snapshot_b64, "sketch": sketch_b64}
                last_detection_time = now
            else:
                for (top, right, bottom, left), label in zip(
                    last_face_results["locations"], last_face_results["labels"]
                ):
                    top, right, bottom, left = [v * 4 for v in (top, right, bottom, left)]
                    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
                    cv2.putText(frame, label, (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                snapshot_b64 = last_face_results.get("snapshot")
                sketch_b64 = last_face_results.get("sketch")
            detected_labels.extend(last_face_results["labels"])

        return jsonify({
            "image": encode_image(frame),
            "detected": detected_labels,
            "snapshot": snapshot_b64,
            "sketch": sketch_b64,
            "geo_info": geo_info
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
