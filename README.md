# YOLOv11 Object Detection on Video

This project demonstrates how to use the YOLOv11 model to perform object detection on video files. It detects and highlights specific object classes with bounding boxes and shows a summary overlay of detected object counts.

![Detection preview](Output.png)

## Features

- Real-time object detection using YOLOv11
- Supports video files and webcam input
- Draws bounding boxes and class labels
- Displays object counts in the corner
- Saves annotated video to a new file

## Getting Started

1. Clone this repository:

```bash
git clone https://github.com/sainsdataid/object-detection-yolov11.git
cd yolov11-object-detection
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Download the YOLOv11 model file `yolo11m.pt` (or other version i.e `yolo11n.pt`, `yolo11s.pt`, etc. See [here](https://docs.ultralytics.com/models/yolo11/#performance-metrics)) and place it in the project folder.

4. Run the script:

```bash
python yolo_object_detection.py
```

## License

MIT License

## Tutorial (Bahasa Indonesia)

A full tutorial of how to use this project in Indonesian is available at:

🔗 [Deteksi Objek pada Video dengan YOLOv11](https://sainsdata.id/machine-learning/12534/deteksi-objek-pada-video-dengan-yolov11/)

