"""
YOLOv3 Object Detection Deployment Script
=========================================
Loads a Darknet YOLOv3 model (cfg + weights) via OpenCV DNN and runs inference.

Usage:
  python deploy_yolo.py --image path/to/image.jpg           # single image
  python deploy_yolo.py --image path/to/image.jpg --save     # save output image
  python deploy_yolo.py --webcam                             # live webcam
  python deploy_yolo.py --video path/to/video.mp4            # video file
"""

import argparse
import atexit
import os
import shutil
import sys
import tempfile
import time

import cv2
import numpy as np


# ── paths ────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(MODEL_DIR, "yolov3.cfg")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "yolov3.weights")
NAMES_PATH = os.path.join(MODEL_DIR, "coco.names")

# ── settings ─────────────────────────────────────────────────────────────
CONF_THRESHOLD = 0.5   # minimum confidence to keep a box
NMS_THRESHOLD = 0.4    # IoU threshold for non-maximum suppression

# Temp directory for clean ASCII copies (OpenCV DNN can't handle CJK paths)
_temp_dir = None


def _is_ascii(s):
    return all(ord(c) < 128 for c in s)


def _ensure_ascii_path(src_path):
    """If path contains non-ASCII chars, copy the file to a temp dir and return ASCII path."""
    global _temp_dir
    if _is_ascii(src_path):
        return src_path
    if _temp_dir is None:
        _temp_dir = tempfile.mkdtemp(prefix="yolo_")
        atexit.register(lambda: shutil.rmtree(_temp_dir, ignore_errors=True))
    dst = os.path.join(_temp_dir, os.path.basename(src_path))
    if not os.path.exists(dst):
        print(f"  [copying to temp: {dst}]")
        shutil.copy2(src_path, dst)
    return dst


def load_model():
    """Load YOLOv3 via OpenCV DNN and return (net, output_layers, classes)."""
    if not os.path.exists(CFG_PATH):
        sys.exit(f"cfg not found: {CFG_PATH}")
    if not os.path.exists(WEIGHTS_PATH):
        sys.exit(f"weights not found: {WEIGHTS_PATH}")
    if not os.path.exists(NAMES_PATH):
        sys.exit(f"coco.names not found: {NAMES_PATH}")

    # Work around OpenCV DNN path handling (fails on CJK characters)
    cfg_ascii = _ensure_ascii_path(CFG_PATH)
    weights_ascii = _ensure_ascii_path(WEIGHTS_PATH)

    # Read class names (plain Python open handles Unicode fine)
    with open(NAMES_PATH, "r") as f:
        classes = [line.strip() for line in f if line.strip()]

    print(f"Loading YOLOv3 from {cfg_ascii} ...")
    net = cv2.dnn.readNetFromDarknet(cfg_ascii, weights_ascii)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # change to DNN_TARGET_CUDA if GPU available

    print(f"Model loaded. {len(classes)} classes, input: 608x608")
    return net, output_layers, classes


def detect(net, output_layers, frame):
    """Run forward pass and return raw detections."""
    blob = cv2.dnn.blobFromImage(
        frame, scalefactor=1/255.0, size=(608, 608),
        mean=(0, 0, 0), swapRB=True, crop=False,
    )
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs


def parse_detections(outputs, frame_h, frame_w, classes):
    """Parse raw YOLO outputs into (boxes, confidences, class_ids)."""
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence < CONF_THRESHOLD:
                continue

            cx, cy, w, h = detection[0:4]
            cx, cy, w, h = cx * frame_w, cy * frame_h, w * frame_w, h * frame_h

            left = int(cx - w / 2)
            top = int(cy - h / 2)

            boxes.append([left, top, int(w), int(h)])
            confidences.append(confidence)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    final = []
    if len(indices) > 0:
        for i in indices.flatten():
            final.append((boxes[i], confidences[i], class_ids[i]))

    return final


def draw_boxes(frame, detections, classes):
    """Draw bounding boxes and labels on frame. Return annotated frame."""
    colours = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)

    for (box, confidence, class_id) in detections:
        x, y, w, h = box
        colour = [int(c) for c in colours[class_id]]
        label = f"{classes[class_id]}: {confidence:.2f}"

        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - th - 4), (x + tw, y), colour, -1)
        cv2.putText(frame, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame


def process_image(image_path, save=False):
    """Run detection on a single image."""
    net, output_layers, classes = load_model()

    frame = cv2.imread(image_path)
    if frame is None:
        sys.exit(f"Cannot read image: {image_path}")

    h, w = frame.shape[:2]
    t0 = time.time()
    outputs = detect(net, output_layers, frame)
    detections = parse_detections(outputs, h, w, classes)
    elapsed = time.time() - t0

    print(f"Found {len(detections)} objects in {elapsed:.2f}s")
    for box, conf, cid in detections:
        print(f"  {classes[cid]:>15s}  {conf:.3f}  @ ({box[0]}, {box[1]}) {box[2]}x{box[3]}")

    annotated = draw_boxes(frame, detections, classes)

    if save:
        out_path = os.path.splitext(image_path)[0] + "_detected.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"Saved: {out_path}")

    cv2.imshow("YOLOv3 Detection", annotated)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_webcam():
    """Run live detection on webcam feed."""
    net, output_layers, classes = load_model()

    # Try multiple backends + indices
    cap = None
    backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
    for b in backends:
        for idx in range(3):
            c = cv2.VideoCapture(idx, b)
            if c.isOpened():
                cap = c
                print(f"Camera found: index {idx}")
                break
        if cap is not None:
            break

    if cap is None:
        sys.exit(
            "No camera available. Please:\n"
            "  - Connect a USB/webcam\n"
            "  - Check Windows Settings > Privacy > Camera\n"
            "  - Or use: python deploy_yolo.py --video your_video.mp4"
        )

    print("Running webcam detection. Press 'q' to quit.")
    frame_count = 0
    fps_start = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        outputs = detect(net, output_layers, frame)
        detections = parse_detections(outputs, h, w, classes)

        annotated = draw_boxes(frame, detections, classes)

        frame_count += 1
        if frame_count % 10 == 0:
            fps = frame_count / (time.time() - fps_start)
            frame_count = 0
            fps_start = time.time()

        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLOv3 Webcam", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_video(video_path):
    """Run detection on a video file."""
    net, output_layers, classes = load_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    out_path = os.path.splitext(video_path)[0] + "_detected.mp4"
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    print(f"Processing video -> {out_path}  (press 'q' to stop early)")
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        outputs = detect(net, output_layers, frame)
        detections = parse_detections(outputs, h, w, classes)

        annotated = draw_boxes(frame, detections, classes)
        writer.write(annotated)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"  frame {frame_count}")

        cv2.imshow("YOLOv3 Video", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done. {frame_count} frames -> {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv3 Deployment")
    parser.add_argument("--image", help="Path to an image file")
    parser.add_argument("--video", help="Path to a video file")
    parser.add_argument("--webcam", action="store_true", help="Use live webcam")
    parser.add_argument("--save", action="store_true", help="Save annotated output")
    args = parser.parse_args()

    if args.image:
        process_image(args.image, save=args.save)
    elif args.video:
        process_video(args.video)
    elif args.webcam:
        process_webcam()
    else:
        parser.print_help()
