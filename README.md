# YOLOv11 Object Detection on Video

This project demonstrates how to use the YOLOv11 model to perform object detection on video files. It detects and highlights specific object classes with bounding boxes and shows a summary overlay of detected object counts.

## Features

- Real-time object detection using YOLOv11
- Supports video files and webcam input
- Draws bounding boxes and class labels
- Displays object counts in the corner
- Saves annotated video to a new file

## Getting Started

1. Clone this repository:

```bash
git clone https://github.com/sainsdataid/yolov11-object-detection.git
cd yolov11-object-detection
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Download the YOLOv11 model file `yolo11m.pt` and place it in the project folder.

4. Run the script:

```bash
python yolo_object_detection.py
```

## License

MIT License
