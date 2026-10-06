# Real-Time Visual Quality Inspection & Bottle Tracking System

A computer vision project that combines bottle detection, object tracking, quality inspection, and production counting in a single video-processing pipeline.

The project is developed as a practical proof of concept, with a modular design intended to support future adaptation to other products and inspection environments.

## Project Objective

Build a system that detects bottles in video, tracks each bottle through the scene, inspects its visual quality, and counts it once when it crosses a defined line.

The final output will include an annotated video and production statistics showing the number of inspected, good, and defective bottles.

## System Overview

Video or Camera → Bottle Detection → Object Tracking → Inspection Zone → Quality Inspection → Directional Counting → Production Statistics

Each inspection result will be associated with a track ID so that the system can connect a bottle’s quality assessment with its movement and counting event.

## Current Capabilities

- Frame-by-frame video processing using OpenCV.
- Detection using a pretrained YOLO11 Nano model.
- Static-image and video detection.
- On-screen display of detection results.
- Export of annotated video.
- CPU inference for local development.

The current detection configuration accepts the COCO **bottle** and **vase** classes to accommodate the containers used in the demonstration.

Tracking, quality inspection, and counting are planned and have not yet been implemented.

## Planned Capabilities

- Multi-object tracking with ByteTrack.
- A configurable inspection zone.
- Visual anomaly detection with PatchCore.
- Anomaly scores and defect localization.
- Inspection results linked to individual track IDs.
- Directional line-crossing detection and counting.
- Production statistics and a complete demonstration.

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Image and video processing | OpenCV |
| Object detection | YOLO11 Nano / Ultralytics |
| Object tracking — planned | ByteTrack |
| Anomaly detection — planned | PatchCore |
| Deep learning framework | PyTorch |
| Version control | Git |

## Project Structure

| Directory | Purpose |
|---|---|
| `src/` | Application source code |
| `data/` | Local images, videos, and datasets |
| `models/` | Model weights and related artifacts |
| `outputs/` | Generated videos and inspection results |
| `tests/` | Automated tests |

Local datasets, model weights, generated outputs, and the virtual environment are excluded from version control.

## Development Roadmap

| Phase | Status |
|---|---|
| 0 — Project setup and environment | Completed |
| 1 — Static-image detection baseline | Completed |
| 2 — Video detection and annotated output | Completed |
| 3 — Multi-object tracking | Next |
| 4 — Directional line crossing and counting | Planned |
| 5 — Bottle anomaly detection | Planned |
| 6 — Defect localization | Planned |
| 7 — Full pipeline integration | Planned |
| 8 — Performance evaluation | Planned |
| 9 — System optimization | Planned |
| 10 — Final demonstration and documentation | Planned |

## Model and Data Strategy

The project uses pretrained YOLO11 Nano weights for detection. Detector fine-tuning is outside the current scope.

Development runs on a CPU laptop. A separate GPU machine will be used for computationally intensive work when required.

MVTec AD’s bottle category is planned as the initial benchmark for anomaly detection. Applying quality inspection to a different product or environment will require representative normal samples and validation in that setting.

The pipeline is intended to support configurable models and settings. Automated model training from user-uploaded datasets is a future extension.

## Evaluation

The completed system will be evaluated on detection reliability, tracking consistency, counting accuracy, anomaly detection, defect localization, and processing speed.

## Scope and Limitations

This project is an educational and engineering proof of concept. It is not a production-certified industrial inspection system.

Performance depends on the selected models, product appearance, lighting, camera position, and visibility. Real-time performance will be measured during evaluation.