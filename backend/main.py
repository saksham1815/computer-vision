from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2, base64, numpy as np

# Import your utils safely
try:
    from utils.hand_utils import detect_hands
    from utils.pose_utils import detect_pose
    from utils.sign_language import recognize_sign
    from utils.draw_utils import air_draw
    from utils.mouse_control import control_mouse
except ImportError:
    # fallback if utils not available
    def detect_hands(frame): return frame
    def detect_pose(frame): return frame
    def recognize_sign(frame): return "Unknown"
    def air_draw(frame, points): return frame, points
    def control_mouse(frame): return None

app = Flask(__name__)
CORS(app)  # allow frontend requests

draw_points = []
mode = "gesture"  # default

def decode_image(img_base64):
    """Decode base64 string to OpenCV image"""
    if "," in img_base64:
        img_base64 = img_base64.split(",")[1]
    img_data = base64.b64decode(img_base64)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def encode_image(img):
    """Encode OpenCV image to base64 string"""
    _, buffer = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

@app.route("/process", methods=["POST"])
def process():
    global mode, draw_points
    try:
        data = request.json
        frame = decode_image(data["image"])
        mode = data.get("mode", mode)

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

        return jsonify({"image": encode_image(frame)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def home():
    return "Flask server is running! Use POST /process"

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
