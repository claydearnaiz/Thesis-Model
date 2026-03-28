# Implementation Plan

## Thesis: Canteen Monitoring Using ROI-based Person Detection using YOLOv5n

## 1. Objective

Develop a real-time computer vision system that detects whether a person is present within specific Regions of Interest (ROI) inside a canteen environment. The system will run on Raspberry Pi devices connected to cameras and use the YOLOv5n object detection model to identify people.

---

## 2. System Overview

The system captures live video from cameras installed in the canteen. A pretrained YOLOv5n object detection model identifies people in each frame. Detected bounding boxes are evaluated against predefined Regions of Interest (ROI) to determine whether a person is present in important areas such as the counter, queue area, or entrance.

**Processing Pipeline**

Camera → Frame Capture → YOLOv5n Person Detection → ROI Filtering → Presence Detection → Data Output

---

## 3. Hardware Requirements

* Raspberry Pi 4 (8GB)
* USB Webcam or Raspberry Pi Camera Module
* MicroSD Card (32GB or higher)
* Power Supply
* Network Connection (WiFi or Ethernet)

Deployment setup:

* One Raspberry Pi per monitored canteen area
* Cameras mounted facing the counter or queue region

---

## 4. Software Stack

Programming Language

* Python

Libraries and Frameworks

* OpenCV
* PyTorch
* YOLOv5 (Ultralytics implementation)
* NumPy
* Flask or FastAPI (optional dashboard backend)

Development Environment

* Python 3.9+
* pip
* Virtual environment

---

## 5. Model Selection

The system uses **YOLOv5 Nano (YOLOv5n)** for object detection.

Reasons for selecting YOLOv5n:

* Lightweight architecture
* Optimized for embedded systems
* Fast inference on CPU devices
* Suitable for Raspberry Pi deployment
* Pretrained on the COCO dataset

The model already contains the **"person" class**, therefore additional training is not required unless custom classes are added.

---

## 6. ROI Definition

Regions of Interest (ROI) represent important areas in the canteen that must be monitored.

Examples:

* Food counter
* Queue area
* Entrance

ROIs are defined using rectangular or polygon coordinates in the video frame.

Example rectangular ROI:

```
roi = (200,150,500,400)
```

These coordinates are calibrated depending on the camera placement and field of view.

---

## 7. Detection Process

The system processes each video frame using the following steps:

1. Capture frame from camera
2. Run YOLOv5n model inference
3. Extract bounding boxes for detected "person" class
4. Compute center point of each bounding box
5. Check whether the center point lies inside the ROI
6. If the center lies within the ROI, a person is considered present in that monitored area

---

## 8. Core Algorithm

Pseudo-code:

```
while camera is running:

    capture frame

    detections = YOLOv5n(frame)

    for each detection:

        if class == person:

            compute bounding box center

            if center inside ROI:
                mark person present

    output detection result
```

---

## 9. System Output

The system produces the following outputs:

* "Person detected at counter"
* "Area empty"
* Number of people detected within ROI

Optional outputs:

* Send results to web dashboard
* Log timestamps for analytics

---

## 10. Multi-Camera Setup

Each Raspberry Pi processes its camera feed locally.

Architecture:

Camera → Raspberry Pi Processing → Send Result → Central Dashboard

Communication methods:

* REST API
* WebSocket
* MQTT

---

## 11. Testing Plan

The system will be evaluated using several scenarios:

1. No person present in ROI
2. Single person entering ROI
3. Multiple people inside ROI
4. Person partially entering ROI
5. Different lighting conditions

Testing metrics:

* Detection accuracy
* False positives
* Processing speed

---

## 12. Evaluation Metrics

The performance of the system will be evaluated using the following metrics:

* Precision
* Recall
* Detection Accuracy
* Processing Latency
* Frames per Second (FPS)

---

## 13. Expected Results

The system is expected to:

* Detect people in real-time using YOLOv5n
* Correctly identify whether individuals are inside predefined ROIs
* Operate at near real-time performance on Raspberry Pi devices
* Provide reliable monitoring of canteen activity

---

## 14. Future Improvements

Possible future enhancements include:

* Queue length estimation
* Crowd density monitoring
* Integration with web dashboard visualization
* Historical data analytics
* Multiple ROI monitoring zones
* Edge optimization using TensorRT or ONNX
