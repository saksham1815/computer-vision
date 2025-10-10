# AetherEye

> Full-stack local web app for real-time webcam-based computer vision: gesture recognition, sign language, air drawing, mouse control, pose estimation, object detection (YOLOv8), face recognition & sketching, emotion detection (DeepFace), and OCR (EasyOCR).

---

## Overview

This repository contains a frontend (HTML/JavaScript) and a Flask-based Python backend that work together to process webcam frames in real time. The frontend captures the webcam frames and posts them to the backend `/process` endpoint. The backend runs different CV pipelines depending on the selected _mode_ and returns a processed image plus metadata (detected labels, snapshots, sketches).

The project supports:

- **Gesture Recognition** (hand detection/tracking)
- **Sign Language Recognition** (placeholder to plug a model)
- **Air Drawing** (finger-tracking to draw on canvas)
- **Mouse Control** (use hand gestures to control mouse pointer)
- **Pose Estimation**
- **Object Detection** using **YOLOv8** (Ultralytics)
- **Face Detection & Recognition** (face_recognition + uploaded reference face)
- **Face Sketching** (pencil-sketch effect saved for matched faces)
- **Emotion Detection** using **DeepFace**
- **Text Extraction (OCR)** using **EasyOCR**
- **Geolocation tagging** (browser geolocation included with each frame)
- **Snapshot saving** (snapshots/sketches saved to `snapshots/`)

## Recommended Python setup (step-by-step)

1. Clone the repo (or place files in a folder):

```bash
git clone <https://github.com/saksham1815/AetherEye.git>
cd project-root
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# or cmd
venv\Scripts\activate.bat
```

3. Install PyTorch first (important):

> **Why first?** Many packages (Ultralytics, DeepFace, EasyOCR) rely on the `torch` package. Choose the appropriate PyTorch build for your environment (GPU/CPU). If you are unsure, install the CPU-only wheel.

**CPU-only example** (safe fallback):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**GPU (CUDA)**: Visit the official PyTorch site for the correct command for your CUDA version (recommended if you have an NVIDIA GPU).

4. Install the rest of the Python dependencies

Create a `requirements.txt` file or use the example below, then install:

```bash
# Example requirements (save to requirements.txt)
flask
flask-cors
numpy
opencv-python
ultralytics
face_recognition
deepface
easyocr
Pillow
requests

# Then install
pip install -r requirements.txt
```

> If `face_recognition` or `dlib` fails to install, ensure you installed system-level build tools shown earlier (cmake, build-essential). On many systems prebuilt wheels exist, but if not dlib will compile from source and needs CMake & a C++ compiler.

---

## Model files and data

- **YOLOv8 weights**: `yolov8n.pt` should be placed in the backend working directory (where `main.py` is). The code uses:

```python
from ultralytics import YOLO
yolo_model = YOLO("yolov8n.pt")
```

The `ultralytics` library will try to load the local `yolov8n.pt` file; if not present, it may attempt to download it automatically. You can also manually download `yolov8n.pt` from the Ultralytics release page and put it in the project root.

- **DeepFace & EasyOCR models**: DeepFace and EasyOCR will download model files on first run into their cache directories. Allow the first run to complete (may take a while).

---

## Configuration

You can edit `main.py` to change:

- `SAVE_DIR` (where snapshots are saved)
- Host/port used by Flask (`app.run(host=..., port=...)`)
- Detection tolerances (face matching tolerance)
- Which utils are enabled (utils/...) if you add/replace modules

To make the Flask server reachable from other machines, set `host='0.0.0.0'` in `app.run`. **Be careful** exposing it to public networks without proper security.

---

## Running the app (backend + frontend)

1. Start the backend Flask server (from the backend folder):

```bash
source venv/bin/activate   # or Windows activate
python main.py
```

By default, the Flask server runs on `http://127.0.0.1:5000`.

2. Serve the frontend (recommended to avoid `file://` restrictions). From the `frontend/` folder run:

```bash
# Python 3
python -m http.server 5500
```

Open your browser at `http://127.0.0.1:5500/frontend/index.html` (or the appropriate path). The frontend will request webcam permission and start sending frames to `http://127.0.0.1:5000/process`.

> Note: If you host backend on a different host/port, update the `fetch()` URLs inside `index.html` accordingly.

---

## API (Backend Endpoints)

### `POST /upload_face`

- **Description**: Upload a reference face image (multipart `image` field). The server will compute and store an encoding to use for later face matching.
- **Response**:

  - `200` `{ "status": "Reference face stored" }` on success
  - `400` `{ "error": "No face found in uploaded image" }` if no face detected

Example (curl):

```bash
curl -X POST -F "image=@/path/to/face.jpg" http://127.0.0.1:5000/upload_face
```

### `POST /process`

- **Description**: Send a frame (base64 JPEG data) and mode. The server processes and returns an annotated image and metadata.
- **Request JSON** (example):

```json
{
  "image": "data:image/jpeg;base64,...",
  "mode": "face",
  "object_name": "person",
  "latitude": 12.34,
  "longitude": 56.78
}
```

- **Response JSON** (example):

```json
{
  "image": "data:image/jpeg;base64,...",
  "detected": ["Matched"],
  "snapshot": "data:image/jpeg;base64,...",
  "sketch": "data:image/jpeg;base64,...",
  "object_snapshot": null,
  "geo_info": { "latitude": 12.34, "longitude": 56.78 }
}
```

---

## How to Use (UI hints)

- Select a processing mode from the dropdown (Gesture, Sign, Draw, Mouse, Pose, Object, Face, Emotion, OCR).
- For **Object** mode: type an object name in the search input and it will be highlighted and saved when detected.
- For **Face** mode: upload a reference face image (left panel) first to enable matching. When a match occurs the server saves a snapshot and a sketch.
- Press **Capture Snapshot** to save the current frame client-side (or extend backend to save on server-side).
- Geolocation (browser) will be shown on the UI and included with each request if the user allows it.

---

## Troubleshooting

- **Camera not working / permission denied**: Ensure you allow camera access in the browser. Don't open the page via file:// — serve it using `python -m http.server` or similar.

- **CORS errors**: The backend uses `flask_cors.CORS(app)`. If you still see issues, confirm the frontend is requesting the correct host/port and that the browser console shows CORS headers.

- **face_recognition / dlib install fails**: Install system build tools (CMake, build-essential/Visual Studio Build Tools) and try again. On Ubuntu:

```bash
sudo apt-get install build-essential cmake
pip install dlib
```

- **Slow first run**: DeepFace and EasyOCR may download models on first invocation. This can take time.

- **YOLO not detecting objects**: Confirm `yolov8n.pt` is available in the working directory or let Ultralytics auto-download it. Ensure the frame passed to the model is a proper `numpy` image.

- **GPU/CUDA problems**: Install matching CUDA drivers and a CUDA-enabled PyTorch build. Verify via Python:

```python
import torch
print(torch.cuda.is_available())
```

---

## Optional improvements & TODOs

- Add authentication + HTTPS if exposing the server across the network.
- Replace/extend placeholder `utils/*` modules with real models for sign language, gesture classification, and robust pose detection (e.g., MediaPipe, BlazePose, OpenPose, or a trained model).
- Add WebSocket or WebRTC streaming for lower-latency frame streaming.
- Add a React frontend with live overlays and controls.

---

## Where snapshots are stored

By default snapshots and sketches are written to the `snapshots/` directory. You can change `SAVE_DIR` in `main.py`.

---

## License & Credits

- This project integrates the following libraries: Ultralytics YOLOv8, OpenCV, face_recognition (dlib), DeepFace, EasyOCR, Flask.
- Please check each library's license for redistribution terms.

---

## Contact / Further help

If you want, I can:

- Generate a `requirements.txt` with pinned versions.
- Create a `docker-compose` file to simplify deployment.
- Convert the simple frontend to a React app.

---

🤝 Contributing

We welcome all contributions!
If you have ideas for new features, optimizations, or want to fix bugs or improve performance, feel free to contribute.

Here’s how you can help:

Fork this repository.

Create a new branch for your feature or bug fix.

Make your changes and ensure everything runs smoothly.

Submit a pull request (PR) describing your improvements or fixes.

💡 You can also share suggestions or open issues if you notice anything that can be enhanced — from performance tweaks to new vision-based features.
Your contributions are highly appreciated and help this project grow!
