"""
YOLOv11 Object Detection on Video

This script performs object detection on a video file using the YOLOv11 model.
It detects selected object classes and displays bounding boxes with labels and counts.

Author: Your Name
Date: 2025-06-29
"""

from ultralytics import YOLO
import torch
import cv2
import numpy as np

# Select device: use GPU if available, otherwise fallback to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv11 model and move it to the selected device
model = YOLO("yolo11m.pt")
model.to(device)

# Define target classes to focus on (based on COCO dataset)
focused_classes = ["person", "bicycle", "car", "motorbike", "bus", "truck"]

# Specify the input video source
video_path = "free_video_george_morina.mp4"
cap = cv2.VideoCapture(video_path)  # for video file
# cap = cv2.VideoCapture(0)         # for webcam

# Get video properties for saving output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_path = "output_video.mp4"
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    for result in results:
        boxes = result.boxes
        names = model.names
        label_counts = {label: 0 for label in focused_classes}

        for box in boxes:
            x1, y1, x2, y2 = np.asarray(box.xyxy[0]).astype(int)
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = names[cls_id]

            if confidence > 0.3 and label in focused_classes:
                label_counts[label] += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {confidence*100:.0f}%",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.0,
                    (0, 255, 0),
                    2,
                )

        summary = ", ".join([f"{v} {k}" for k, v in label_counts.items() if v > 0])
        box_padding = 12
        (text_width, text_height), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (5, 5), (5 + text_width + box_padding, 5 + text_height + box_padding), (0, 255, 0), -1)
        cv2.putText(frame, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    out.write(frame)
    cv2.imshow("YOLOv11 Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
