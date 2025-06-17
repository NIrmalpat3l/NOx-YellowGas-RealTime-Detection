import cv2
import numpy as np
from ultralytics import YOLO

# Tweak these thresholds
CONF_THRESH = 0.10
NMS_IOU     = 0.55

_model = None
def annotate_frame(frame, model_path):
    global _model
    if _model is None:
        _model = YOLO(model_path)

    results = _model(frame, conf=CONF_THRESH, iou=NMS_IOU)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int).tolist()
    confidences = results.boxes.conf.cpu().numpy().tolist()

    if not boxes:
        return [], [], []

    H, W = frame.shape[:2]
    rois, yellow_flags = [], []

    for (x1, y1, x2, y2), conf in zip(boxes, confidences):
        w = x2 - x1
        new_w = int(w * 1.5)
        cx = (x1 + x2) // 2
        sx1 = max(0, cx - new_w // 2)
        sx2 = min(W, cx + new_w // 2)

        new_h = int(w * 2.0)
        sy2 = y1 + int(new_h * 0.2)
        sy1 = max(0, sy2 - new_h)

        rois.append([sx1, sy1, sx2, sy2])

        roi = frame[sy1:sy2, sx1:sx2]
        yellow = False
        if roi.size:
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([10,100,100]), np.array([40,255,255]))
            yp = int(cv2.countNonZero(mask))
            total = roi.shape[0] * roi.shape[1]
            yellow = total > 0 and yp > total * 0.01
        yellow_flags.append(yellow)

    return boxes, rois, yellow_flags
