import cv2
import time
import json
import base64
import numpy as np
import requests
from insightface.app import FaceAnalysis

# ——— CONFIG ———
SERVER_URL      = "http://<your-end-point>.com/recognize"
MODEL_NAME      = "buffalo_sc"
DET_SIZE        = (320, 320)
DOWNSCALE_FACTOR= 1.5
PROCESS_EVERY_N = 5
USE_CENTRAL_REGION = False

def load_detector(name: str) -> FaceAnalysis:
    app = FaceAnalysis(name=name)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    return app

face_app = load_detector(MODEL_NAME)

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

    # ---- calculate FPS ----
    now = time.time()
    if prev_time is not None:
        fps_history.append(1.0 / (now - prev_time))
        fps_history = fps_history[-5:]
    prev_time = now
    fps = float(np.mean(fps_history)) if fps_history else 0.0

    bboxes = []
    people_count = 0

    # ---- run detection every N frames ----
    if frame_count % PROCESS_EVERY_N == 0:
        proc = frame.copy()

        # optional downscale
        if DOWNSCALE_FACTOR > 1.0:
            proc = cv2.resize(proc,
                              (int(w / DOWNSCALE_FACTOR), int(h / DOWNSCALE_FACTOR)))

        # optional central crop
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
            # map back to original frame coords
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

            # crop full-res face
            crop = frame[y1:y2, x1:x2]

            # encode + base64
            success, buf = cv2.imencode('.jpg', crop)
            if not success:
                continue
            crop_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

            bboxes.append({
                "bbox": [x1, y1, x2, y2],
                "crop": crop_b64
            })

    # ---- prepare payload ----
    payload = {
        "fps":       round(fps, 2),
        "people_count": people_count,
        "bboxes":    bboxes
    }

    # ---- send as JSON ----
    try:
        resp = requests.post(
            SERVER_URL,
            json=payload,
            timeout=1
        )
        # optional: check resp.status_code or resp.json()
    except requests.RequestException as e:
        print("Upload failed:", e)

    # ---- preview ----
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    cv2.putText(frame, f"Count: {people_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

    cv2.imshow("Face Detection Only", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
