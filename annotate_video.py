# annotate_video.py

import cv2
import argparse
import sys
import os
import time
from ultralytics import YOLO
from annotation import annotate_frame
from yellow_event_logger import YellowGasEventLogger
from matplotlib import pyplot as plt
from camera_motion_detector import CameraMotionDetector


def box_iou(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b
    xi1, yi1 = max(xa1, xb1), max(ya1, yb1)
    xi2, yi2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = iw * ih
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_lost=5):
        self.next_id = 1
        self.tracks = {}  # id: {"box": [...], "lost": int}
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost

    def update(self, detections):
        updated = {}
        assigned = set()

        # 1) match each detection to existing track
        for det in detections:
            best_iou, best_id = 0, None
            for tid, tr in self.tracks.items():
                iou = box_iou(det, tr["box"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou, best_id = iou, tid
            if best_id is not None:
                updated[best_id] = {"box": det, "lost": 0}
                assigned.add(best_id)
            else:
                # new track
                updated[self.next_id] = {"box": det, "lost": 0}
                self.next_id += 1

        # 2) carry over unmatched (optional, but we won't draw them)
        for tid, tr in self.tracks.items():
            if tid not in assigned:
                lost = tr["lost"] + 1
                if lost <= self.max_lost:
                    updated[tid] = {"box": tr["box"], "lost": lost}

        self.tracks = updated
        return self.tracks

def show_with_matplotlib(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.clf()
    plt.imshow(rgb)
    plt.axis('off')
    plt.pause(0.001)

def main(input_path, model_path, output_path=None):
    # Load
    model = YOLO(model_path)
    tracker = SimpleTracker(iou_threshold=0.3, max_lost=5)
    logger  = YellowGasEventLogger()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: cannot open {input_path!r}")
        sys.exit(1)

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # Matplotlib window sized to video resolution, but capped to max display size
    dpi = 100
    # video size in inches at this DPI
    vid_w_in, vid_h_in = W / dpi, H / dpi
    # maximum figure size in inches
    max_w_in, max_h_in = 12, 8
    # compute scale so we never exceed max dims
    scale = min(max_w_in / vid_w_in, max_h_in / vid_h_in, 1.0)
    fig_w, fig_h = vid_w_in * scale, vid_h_in * scale

    plt.ion()
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    last_ms = -1000

    motion_detector = CameraMotionDetector(
        max_trans_thresh=20.0,  # tweak to your scenario
        max_rot_thresh=10.0,
        min_inliers=30
    )

    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if motion_detector.is_camera_moved(frame):
                print("⚠️  Camera moved! Re-aligning reference.")


            now_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if now_ms - last_ms < 1000:
                continue
            last_ms = now_ms

            # 1) Detect chimneys + ROIs + yellow flags
            boxes, rois, yellow_flags = annotate_frame(frame, model)
            # returns lists of equal length

            # 2) Track
            tracks = tracker.update(boxes)
            # build det→tid map ONLY for detections in this frame
            det_to_tid = {
                tuple(tr["box"]): tid
                for tid, tr in tracks.items()
                if tuple(tr["box"]) in {tuple(b) for b in boxes}
            }


            yellow_map = {}
            for idx, flag in enumerate(yellow_flags, start=1):
                tid = det_to_tid.get(tuple(boxes[idx-1]), idx)
                yellow_map[tid] = flag

            # 3) log any starts/ends
            ts = time.time() - start_time
            logger.update(yellow_map, timestamp=ts)

            # 4) Draw each detection once
            canvas = frame.copy()
            for idx, b in enumerate(boxes):
                x1, y1, x2, y2 = b
                sx1, sy1, sx2, sy2 = rois[idx]
                is_yellow = yellow_flags[idx]
                tid = det_to_tid.get(tuple(b), None)

                # choose colors/thickness
                box_col = (255,0,0) if is_yellow else (0,255,255)
                box_th  = 2
                roi_col = (0,0,255) if is_yellow else (0,255,0)
                roi_th  = 2

                # detection box + label
                cv2.rectangle(canvas, (x1,y1),(x2,y2), box_col, box_th)
                label = f"Chimney {tid or (idx+1)}"
                cv2.putText(canvas, label, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_col, box_th)

                # smoke ROI + label
                cv2.rectangle(canvas, (sx1,sy1),(sx2,sy2), roi_col, roi_th)
                cv2.putText(canvas, f"SmokeROI {tid or (idx+1)}",
                            (sx1,sy1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_col, roi_th)

            # 4) display & save
            show_with_matplotlib(canvas)
            if writer:
                writer.write(canvas)

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        logger.close_all(timestamp=time.time()-start_time)
        cap.release()
        if writer:
            writer.release()
        plt.ioff()
        plt.close(fig)
        print("✅ Done.")

if __name__=="__main__":
    p = argparse.ArgumentParser(description="Detect→track→draw 1 FPS")
    p.add_argument("input", help="video file path")
    p.add_argument("-m","--model", default=r"D:\gasNOx_Detection\SavedModel\lastYolo12CCTV.pt")
    p.add_argument("-o","--output", help="optional output mp4")
    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"Input not found: {args.input!r}"); sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"Model not found: {args.model!r}"); sys.exit(1)

    main(args.input, args.model, args.output)
