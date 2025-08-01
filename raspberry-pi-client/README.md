````markdown
# Raspberry Pi Face Recognition Client

This repository contains a Python client designed to run on a Raspberry Pi (or any Linux machine with a camera) that:

- Captures frames from a connected webcam.
- Runs local face detection (InsightFace) every _N_ frames.
- Extracts face crops, encodes them in base64, and sends them (alongside bounding boxes) to a remote FastAPI backend for recognition.
- Displays a live preview with FPS and detected person count.

---

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Installation](#installation)  
3. [Configuration](#configuration)  
4. [Running the Client](#running-the-client)  
5. [Code Overview](#code-overview)  
6. [Project Structure](#project-structure)  
7. [Tips & Troubleshooting](#tips--troubleshooting)  
8. [License](#license)  

---

## Prerequisites

- **Raspberry Pi** (any model with a camera input) or a Linux machine with a connected webcam.  
- **Python 3.8+** installed on your Raspberry Pi.  
- A working camera module (e.g., Raspberry Pi Camera or USB webcam) recognized by OpenCV.  
- A running Face Recognition backend (FastAPI) accessible via `SERVER_URL` (by default `http://192.168.146.198:8000/recognize`).  

---

## Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/yourusername/pi-face-recognition-client.git
   cd pi-face-recognition-client
````

2. **Create (and activate) a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate       # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   This project includes a `requirements.txt` file listing all required Python packages. Run:

   ```bash
   pip install -r requirements.txt
   ```

   Example contents of `requirements.txt`:

   ```
   opencv-python
   numpy
   requests
   insightface
   ```

   Ensure you have a compatible version of OpenCV that supports your camera device.

---

## Configuration

At the top of the `client.py` script, you’ll find several configurable constants:

```python
# ——— CONFIG ———
SERVER_URL         = "http://192.168.146.198:8000/recognize"
MODEL_NAME         = "buffalo_sc"
DET_SIZE           = (320, 320)
DOWNSCALE_FACTOR   = 1.5
PROCESS_EVERY_N    = 5
USE_CENTRAL_REGION = False
```

* `SERVER_URL`
  The full URL of the backend “/recognize” endpoint. Update this to match your FastAPI server’s IP/hostname and port (e.g., `http://<server-ip>:8000/recognize`).

* `MODEL_NAME`
  InsightFace model to load (e.g., `"buffalo_sc"`). Ensure the same model is available on both client and server if you want consistent detection and embedding size.

* `DET_SIZE`
  The face detector’s input size (height, width). Larger values can yield more accurate detections but reduce FPS. Commonly `(320, 320)` or `(640, 640)`.

* `DOWNSCALE_FACTOR`
  Downscale the camera frame by this factor before running detection. E.g., `1.5` means the detection runs on a frame that is ⅔ of the original resolution. Increase to speed up detection at the cost of small faces potentially being missed.

* `PROCESS_EVERY_N`
  Only run face detection once every *N* frames. For example, if set to `5`, the code will run the detector on frames 5, 10, 15, etc., and reuse the previous bounding boxes in between. Adjust to balance CPU load vs. detection latency.

* `USE_CENTRAL_REGION`
  If `True`, crop out the central 60% of the downscaled frame before passing it to the detector. This can speed up detection if you know faces always appear near the center.

---

## Running the Client

Once dependencies are installed and `client.py` is configured:

1. **Activate your virtual environment** (if not already active):

   ```bash
   source .venv/bin/activate
   ```

2. **Launch the client**:

   ```bash
   python client.py
   ```

3. **Behavior**

   * The client will open the default camera (device 0).
   * Every `PROCESS_EVERY_N` frames, it runs InsightFace to detect faces on a (possibly downscaled) copy of the frame.
   * For each detected face, it computes a bounding box in the original resolution, crops that face, and encodes it as a base64 JPEG.
   * The client builds a JSON payload:

     ```json
     {
       "fps":       <current_FPS>,
       "people_count": <number_of_faces_detected>,
       "bboxes": [
         {
           "bbox": [x1, y1, x2, y2],
           "crop": "<base64-jpg-string>"
         },
         ...
       ]
     }
     ```
   * It sends this payload via HTTP POST to `SERVER_URL` (timeout 1 second).
   * On the live preview window, it overlays “FPS: <fps>” and “Count: \<people\_count>” in green text at the top-left corner.
   * Press `q` in the OpenCV window to quit.

---

## Code Overview

Below is a high-level walkthrough of the key sections in `client.py`:

1. **Imports & Configuration**

   ```python
   import cv2
   import time
   import json
   import base64
   import numpy as np
   import requests
   from insightface.app import FaceAnalysis
   ```

   * OpenCV (`cv2`): for grabbing frames, displaying the preview, and image encoding.
   * `insightface.app.FaceAnalysis`: runs face detection and produces bounding boxes + embeddings (though in this client, embedding is not used—only detection is).

2. **Detector Initialization**

   ```python
   def load_detector(name: str) -> FaceAnalysis:
       app = FaceAnalysis(name=name)
       app.prepare(ctx_id=0, det_size=DET_SIZE)
       return app

   face_app = load_detector(MODEL_NAME)
   ```

   * Loads and prepares the InsightFace detector on GPU (if `ctx_id=0`) or CPU (set to `-1` if no GPU).

3. **Video Capture & Main Loop**

   ```python
   cap = cv2.VideoCapture(0)
   prev_time = None
   frame_count = 0
   fps_history = []

   while True:
       ret, frame = cap.read()
       if not ret:
           break

       frame_count += 1
       h, w = frame.shape[:2]

       # Calculate FPS using timestamps of recent frames
       now = time.time()
       if prev_time is not None:
           fps_history.append(1.0 / (now - prev_time))
           fps_history = fps_history[-5:]
       prev_time = now
       fps = float(np.mean(fps_history)) if fps_history else 0.0

       bboxes = []
       people_count = 0
   ```

   * Continuously reads frames from the camera.
   * Maintains a moving average of the last 5 frame‐interval FPS measurements.

4. **Face Detection (Every N Frames)**

   ```python
   if frame_count % PROCESS_EVERY_N == 0:
       proc = frame.copy()

       # Optional downscale
       if DOWNSCALE_FACTOR > 1.0:
           proc = cv2.resize(proc, (int(w / DOWNSCALE_FACTOR), int(h / DOWNSCALE_FACTOR)))

       # Optional central crop
       if USE_CENTRAL_REGION:
           ph, pw = proc.shape[:2]
           mh, mw = int(ph * 0.2), int(pw * 0.2)
           proc = proc[mh:ph-mh, mw:pw-mw]

       faces = face_app.get(proc)
       people_count = len(faces)

       scale_x = w / proc.shape[1]
       scale_y = h / proc.shape[0]

       for face in faces:
           x1, y1, x2, y2 = face.bbox.astype(int)
           x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
           y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

           crop = frame[y1:y2, x1:x2]
           success, buf = cv2.imencode('.jpg', crop)
           if not success:
               continue
           crop_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

           bboxes.append({
               "bbox": [x1, y1, x2, y2],
               "crop": crop_b64
           })
   ```

   * Runs InsightFace’s `face_app.get(...)` on the possibly downscaled/cropped frame.
   * Converts each detected bounding box back to the original frame’s coordinate space.
   * Crops the original frame to extract each face, JPEG‐encodes it, and base64‐encodes that JPEG for JSON transmission.
   * Accumulates a list of `{ "bbox": [...], "crop": "..." }` dictionaries.

5. **Payload Construction & HTTP POST**

   ```python
   payload = {
       "fps": round(fps, 2),
       "people_count": people_count,
       "bboxes": bboxes
   }

   try:
       resp = requests.post(SERVER_URL, json=payload, timeout=1)
       # Optionally check resp.status_code or resp.json()
   except requests.RequestException as e:
       print("Upload failed:", e)
   ```

   * Builds a JSON payload containing the current FPS, detected face count, and the list of bounding boxes + base64 crops.
   * Sends it to the backend in a single POST. If the request times out or fails, it prints an error and continues.

6. **Preview Overlay & Display**

   ```python
   cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
   cv2.putText(frame, f"Count: {people_count}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

   cv2.imshow("Face Detection Only", frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break
   ```

   * Draws the FPS and people count on the top-left of the camera frame.
   * Opens an OpenCV window named “Face Detection Only.”
   * Press `q` to quit cleanly.

---

## Project Structure

```
pi-face-recognition-client/
├── README.md
├── requirements.txt
├── client.py         # Main Python script containing the code above
└── .venv/            # (Optional) Virtual environment directory
```

* **`client.py`**
  Contains the entire logic for capturing frames, detecting faces locally, encoding crops, and sending them as JSON to the backend.

* **`requirements.txt`**
  Python dependencies required to run `client.py`. Example:

  ```
  opencv-python
  numpy
  requests
  insightface
  ```

---

## Tips & Troubleshooting

1. **Camera Permission / Device Index**

   * The script uses `cv2.VideoCapture(0)`, which grabs the default camera. If you have multiple cameras, try changing the index to `1` or `2`.
   * Ensure your user has permission to access the `/dev/video0` device on the Pi (e.g., belong to the `video` group or run with `sudo`).

2. **InsightFace on Raspberry Pi**

   * InsightFace models are relatively large. On a Pi 4 or Pi 3 B+, GPU acceleration is not available unless you have a specialized inference accelerator (e.g., Coral USB).
   * For pure CPU mode, modify `load_detector` to use `ctx_id=-1`:

     ```python
     face_app.prepare(ctx_id=-1, det_size=DET_SIZE)
     ```
   * You may need to install `onnxruntime` for CPU inference:

     ```bash
     pip install onnxruntime
     ```

3. **Reducing CPU Load / Increasing FPS**

   * Increase `PROCESS_EVERY_N` to run detection less frequently.
   * Increase `DOWNSCALE_FACTOR` to run detection on a smaller resolution (e.g., `2.0`).
   * Disable `USE_CENTRAL_REGION` unless faces reliably appear near the center.

4. **Backend Server Connectivity**

   * Verify that `SERVER_URL` is correct and that the FastAPI server is running/accessible from your Pi’s network.
   * If running your backend on a different subnet, adjust firewall rules or port forwarding.
   * Test connectivity manually:

     ```bash
     curl -X POST http://<server-ip>:8000/recognize \
          -H "Content-Type: application/json" \
          -d '{"bboxes": []}'
     ```

5. **Handling Timeouts**

   * The `timeout=1` parameter in `requests.post` will abort if the server doesn’t respond within 1 second.
   * If your network is slower or your server processing takes more time, increase the timeout (e.g., `timeout=2` or `timeout=5`).

6. **Running Headless (No Preview Window)**

   * If you plan to run this client headlessly (no GUI), comment out or remove the `cv2.imshow(...)` and `cv2.waitKey(...)` calls.
   * You can still capture, detect, and send payloads in a loop without opening any window.

---

## License

This project is released under the [MIT License](LICENSE). Feel free to use, modify, and distribute.

```
```
