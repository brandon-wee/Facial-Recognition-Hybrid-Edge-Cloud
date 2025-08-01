# yolo5_face_demo.py
import torch
import cv2
import numpy as np
from pathlib import Path

# ----------------------------------------------------------------------
# 1.  Download weights (first run only)  ───────────────────────────────
#     deepcam-cn/yolov5-face provides yolov5n-face.pt (~6 MB)
# ----------------------------------------------------------------------

weights_path = "yolov5n-face.pt"          # ← existing file in your cwd
repo_path    = "/ultralytics/yolov5"                 # cloned Ultralytics repo (or pip-installed)

model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=True)

model.conf = 0.25     # confidence threshold
model.iou  = 0.45     # NMS IoU threshold


# ----------------------------------------------------------------------
# 3.  Inference on an image  ───────────────────────────────────────────
img_path = "/Users/yousufsarfraz_1/Documents/GitHub/mds-25/downsizing_experimental/calibration_images_v1/000044.jpg"                 # change to your file
assert Path(img_path).exists(), "Image not found."

results = model(img_path, size=640)    # run once, size can be 320-1280
detections = results.xyxy[0].cpu().numpy()  # (N,6): x1,y1,x2,y2,conf,cls

# ----------------------------------------------------------------------
# 4.  Visualise & save  ────────────────────────────────────────────────
image = cv2.imread(img_path)
for x1, y1, x2, y2, conf, cls in detections:
    cv2.rectangle(
        image,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0), 2,
    )
    cv2.putText(
        image,
        f"{conf:.2f}",
        (int(x1), int(y1) - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

cv2.imwrite("output.jpg", image)
print(f"Done. {len(detections)} faces → output.jpg")
