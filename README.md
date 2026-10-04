# Real-Time Visual Quality Inspection & Bottle Tracking System

An end-to-end Computer Vision system for detecting, tracking, inspecting, and counting bottles in video streams.

The project simulates an automated visual quality-control pipeline in which bottles moving through an inspection area are individually detected, assigned persistent identities, inspected for visual anomalies, and counted while real-time production statistics are generated.

> **Project Status:** 🚧 Currently under active development.

---

## 🎯 Project Objective

Manual visual inspection can become repetitive, inconsistent, and difficult to scale in production environments.

This project explores how Computer Vision can be used to build an automated inspection pipeline capable of answering four fundamental questions:

- **Where is each bottle?** — Object Detection
- **Which bottle is which across video frames?** — Multi-Object Tracking
- **Does the bottle contain a visual anomaly?** — Anomaly Detection
- **How many normal and defective bottles passed through the system?** — Counting & Analytics

The goal is not only to train an AI model, but to integrate multiple Computer Vision components into a complete video-processing system.

---

## 🧠 System Architecture

The planned pipeline is:

```text
Video / Camera Stream
        │
        ▼
┌─────────────────────┐
│  Bottle Detection   │
│        YOLO         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Object Tracking     │
│     ByteTrack       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Inspection Zone    │
│  Bottle Extraction  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Anomaly Detection   │
│     PatchCore       │
└──────────┬──────────┘
           │
           ├──── Normal
           │
           └──── Defective
                    │
                    ▼
             Defect Localization
                    │
                    ▼
┌─────────────────────┐
│   Line Crossing     │
│      Counting       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Real-Time Analytics │
│ Total / Good / Bad  │
│ Defect Rate / FPS   │
└─────────────────────┘
```

Each component has a specific responsibility rather than relying on a single model to solve the entire problem.

---

## 🔍 Core Components

### 1. Bottle Detection

A YOLO-based object detector identifies bottles in each video frame and returns their bounding boxes and detection confidence scores.

The detector is responsible only for locating bottles. It does **not** determine whether a bottle is normal or defective.

### 2. Multi-Object Tracking

ByteTrack associates detections between consecutive frames and assigns a persistent ID to each bottle.

This allows the system to understand that the same bottle appearing across multiple frames is still the same physical object.

### 3. Visual Anomaly Detection

Detected bottles entering the inspection region are extracted and analyzed using PatchCore.

Instead of requiring every possible defect type to be explicitly defined beforehand, the anomaly-detection model learns the visual appearance of normal bottles and identifies deviations from normal appearance.

The inspection stage is designed to produce:

- Anomaly score
- Normal / Defective decision
- Anomaly heatmap
- Defect localization

### 4. Bottle Counting

A directional virtual line is used to count each tracked bottle once.

Because counting is associated with persistent Track IDs, the system avoids repeatedly counting the same bottle across multiple video frames.

### 5. Production Analytics

The final system is intended to display real-time statistics such as:

- Total bottles processed
- Normal bottles
- Defective bottles
- Defect rate
- Processing FPS

---

## 🗂️ Project Structure

```text
visual-quality-inspection/
│
├── data/              # Local datasets and input media
├── models/            # Model weights and exported models
├── outputs/           # Generated videos, images, and reports
├── src/               # Main source code
├── tests/             # Tests
│
├── .gitignore
├── README.md
└── requirements.txt
```

The project structure will evolve as individual components are implemented.

---

## 🛠️ Technology Stack

The project is planned around:

- **Python**
- **PyTorch**
- **Ultralytics YOLO**
- **OpenCV**
- **ByteTrack**
- **PatchCore**
- **MVTec AD**
- **NumPy**

Additional dependencies may be introduced as the system evolves.

---

## 📊 Development Roadmap

| Phase | Component | Status |
|------:|-----------|:------:|
| 0 | Project setup & environment | 🟢 |
| 1 | Bottle detection | ⏳ |
| 2 | Video inference | ⏳ |
| 3 | Multi-object tracking | ⏳ |
| 4 | Line-crossing & counting | ⏳ |
| 5 | Bottle anomaly detection | ⏳ |
| 6 | Defect localization | ⏳ |
| 7 | Full pipeline integration | ⏳ |
| 8 | Performance evaluation | ⏳ |
| 9 | System optimization | ⏳ |
| 10 | Final demo & documentation | ⏳ |

---

## 📦 Dataset Strategy

Different stages of the system require different types of visual information.

For object detection, the system requires images containing bottles with object-level localization information.

For anomaly detection, the project will use industrial visual inspection data, with **MVTec AD Bottle** serving as the primary anomaly-detection benchmark.

Dataset preparation and preprocessing procedures will be documented as the corresponding stages are implemented.

---

## 🎬 Expected Final Output

The final application is intended to process a video or camera stream and produce an annotated output similar to:

```text
┌─────────────────────────────────────────────┐
│                                             │
│    ID: 07                    ID: 08          │
│   ┌────────┐                ┌────────┐       │
│   │ BOTTLE │                │ BOTTLE │       │
│   │  GOOD  │                │ DEFECT │       │
│   └────────┘                └────────┘       │
│                                             │
│  ─────────── Inspection / Count Line ─────  │
│                                             │
│  Total: 42                                  │
│  Good: 37                                   │
│  Defective: 5                               │
│  Defect Rate: 11.9%                         │
│  FPS: 28.4                                  │
│                                             │
└─────────────────────────────────────────────┘
```

The final output will combine AI inference with tracking, temporal logic, visualization, and production statistics.

---

## 📈 Evaluation

The completed system will be evaluated at multiple levels rather than relying on a single accuracy value.

Planned evaluation includes:

- Object detection performance
- Anomaly detection performance
- Defect localization quality
- Tracking stability
- Counting accuracy
- End-to-end processing speed

Detailed metrics will be added as each component is implemented and experimentally evaluated.

---

## 🚧 Current Development Status

The repository is currently in the initial development stage.

The project environment and repository structure have been prepared, and development is beginning with the **Bottle Detection** component.

Results, model evaluations, demo media, and implementation details will be added progressively as the system is built.

---

## 📌 Project Scope

This project is designed as a practical Computer Vision engineering project and portfolio demonstration.

It focuses on integrating modern detection, tracking, anomaly-detection, and video-processing techniques into a coherent end-to-end system.

The system should be considered a **proof-of-concept visual inspection platform**, not a production-certified industrial quality-control system.