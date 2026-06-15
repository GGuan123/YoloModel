"""
YOLOv3 ONNX Deployment Script
=============================
Loads yolov3.onnx via ONNX Runtime and runs inference.
Supports image, video, and webcam modes.

Usage:
  python deploy_onnx.py --image path/to/image.jpg
  python deploy_onnx.py --image path/to/image.jpg --save
  python deploy_onnx.py --webcam
  python deploy_onnx.py --video path/to/video.mp4
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(MODEL_DIR, "yolov3.onnx")
NAMES_PATH = os.path.join(MODEL_DIR, "coco.names")

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
INPUT_SIZE = 608


def load_classes():
    with open(NAMES_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_model():
    print(f"Loading ONNX model from {ONNX_PATH}...")
    session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    print(f"  Input: {session.get_inputs()[0].name} {session.get_inputs()[0].shape}")
    outputs = session.get_outputs()
    for o in outputs:
        print(f"  Output: {o.name} {o.shape}")
    return session


def preprocess(frame):
    """Resize and normalize frame to 608x608, return blob (1,3,608,608)."""
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_SIZE, INPUT_SIZE),
                                  (0, 0, 0), swapRB=True, crop=False)
    return blob


def postprocess(outputs, frame_h, frame_w, classes):
    """Parse ONNX YOLO outputs into (boxes, confidences, class_ids)."""
    boxes = []
    confidences = []
    class_ids = []

    # YOLOv3 anchors (px relative to 608x608), masks = yolo-layer-to-anchor mapping
    all_anchors = [(10,13),(16,30),(33,23),(30,61),(62,45),(59,119),(116,90),(156,198),(373,326)]
    masks = [[6,7,8],[3,4,5],[0,1,2]]

    for oi, output in enumerate(outputs):
        # output shape: (1, boxes_per_scale, grid_h, grid_w)
        # Each box = [tx, ty, tw, th, obj, class_0, class_1, ...]
        out = output[0]  # batch=0
        num_boxes, gh, gw = out.shape[0], out.shape[1], out.shape[2]
        out = out.reshape(num_boxes, gh, gw)

        num_classes = num_boxes - 5
        # YOLOv3 layout: 255 channels = 3 anchors * (4 bbox + 1 obj + 80 class)
        # Each anchor block: 85 channels (tx, ty, tw, th, obj, class_0..79)
        num_anchors = 3
        num_classes = (num_boxes // num_anchors) - 5  # 80
        grid_x, grid_y = np.meshgrid(np.arange(gw), np.arange(gh))

        for b in range(num_anchors):
            idx = b * (5 + num_classes)
            tx = out[idx + 0]; ty = out[idx + 1]; tw = out[idx + 2]; th = out[idx + 3]
            # Apply sigmoid to tx, ty, obj (YOLOv3 raw outputs need activation)
            tx = 1.0 / (1.0 + np.exp(-tx))
            ty = 1.0 / (1.0 + np.exp(-ty))
            obj = out[idx + 4]
            obj = 1.0 / (1.0 + np.exp(-obj))
            cls_scores = out[idx + 5 : idx + 5 + num_classes, :, :]
            cls_scores = 1.0 / (1.0 + np.exp(-cls_scores))

            anchor_w, anchor_h = all_anchors[masks[oi][b]]

            mask = obj > CONF_THRESHOLD
            if not np.any(mask):
                continue

            ys, xs = np.where(mask)
            for y, x in zip(ys, xs):
                obj_conf = float(obj[y, x])
                if obj_conf < CONF_THRESHOLD:
                    continue

                cx = (float(tx[y, x]) + grid_x[y, x]) / gw
                cy = (float(ty[y, x]) + grid_y[y, x]) / gh
                bw = anchor_w * np.exp(float(tw[y, x])) / INPUT_SIZE
                bh = anchor_h * np.exp(float(th[y, x])) / INPUT_SIZE

                scores = cls_scores[:, y, x]
                class_id = int(np.argmax(scores))
                conf = obj_conf * float(scores[class_id])

                if conf < CONF_THRESHOLD:
                    continue

                cx, cy = cx * frame_w, cy * frame_h
                bw, bh = bw * frame_w, bh * frame_h
                left = int(cx - bw / 2); top = int(cy - bh / 2)

                boxes.append([left, top, int(bw), int(bh)])
                confidences.append(conf)
                class_ids.append(class_id)

    if not boxes:
        return []

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    result = []
    if len(indices) > 0:
        for i in indices.flatten():
            result.append((boxes[i], confidences[i], class_ids[i]))
    return result


def draw_boxes(frame, detections, classes):
    colours = np.random.randint(0, 255, size=(len(classes), 3), dtype=np.uint8)
    for box, confidence, class_id in detections:
        x, y, w, h = box
        colour = [int(c) for c in colours[class_id]]
        label = f"{classes[class_id]}: {confidence:.2f}"
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x, y - th - 4), (x + tw, y), colour, -1)
        cv2.putText(frame, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def process_image(image_path, save=False):
    session = load_model()
    classes = load_classes()
    frame = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        sys.exit(f"Cannot read: {image_path}")
    h, w = frame.shape[:2]

    t0 = time.time()
    blob = preprocess(frame)
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    detections = postprocess(outputs, h, w, classes)
    elapsed = time.time() - t0

    print(f"Found {len(detections)} objects in {elapsed:.2f}s")
    for box, conf, cid in detections:
        print(f"  {classes[cid]:>15s}  {conf:.3f}  @ ({box[0]}, {box[1]}) {box[2]}x{box[3]}")

    annotated = draw_boxes(frame, detections, classes)
    if save:
        out_path = os.path.splitext(image_path)[0] + "_onnx_detected.jpg"
        _, buf = cv2.imencode(".jpg", annotated)
        buf.tofile(out_path)
        print(f"Saved: {out_path}")

    cv2.imshow("YOLOv3 ONNX", annotated)


def process_webcam():
    session = load_model()
    classes = load_classes()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit("No camera available")

    print("Webcam mode. Press 'q' to quit.")
    frame_count = 0
    t0 = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]

        blob = preprocess(frame)
        outputs = session.run(None, {session.get_inputs()[0].name: blob})
        detections = postprocess(outputs, h, w, classes)

        annotated = draw_boxes(frame, detections, classes)

        frame_count += 1
        if frame_count % 10 == 0:
            fps = frame_count / (time.time() - t0)
            frame_count = 0
            t0 = time.time()

        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("YOLOv3 ONNX Webcam", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_video(video_path):
    session = load_model()
    classes = load_classes()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {video_path}")

    out_path = os.path.splitext(video_path)[0] + "_onnx_detected.mp4"
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    print(f"Processing -> {out_path}  (press 'q' to stop)")
    fc = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        blob = preprocess(frame)
        outputs = session.run(None, {session.get_inputs()[0].name: blob})
        detections = postprocess(outputs, h, w, classes)
        annotated = draw_boxes(frame, detections, classes)
        writer.write(annotated)
        fc += 1
        if fc % 30 == 0:
            print(f"  frame {fc}")
        cv2.imshow("YOLOv3 ONNX Video", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Done: {fc} frames -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv3 ONNX Deployment")
    parser.add_argument("--image", help="Path to image file")
    parser.add_argument("--video", help="Path to video file")
    parser.add_argument("--webcam", action="store_true", help="Use webcam")
    parser.add_argument("--save", action="store_true", help="Save output")
    args = parser.parse_args()

    if args.image:
        process_image(args.image, save=args.save)
    elif args.video:
        process_video(args.video)
    elif args.webcam:
        process_webcam()
    else:
        parser.print_help()
