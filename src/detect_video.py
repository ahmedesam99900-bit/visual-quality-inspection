from pathlib import Path

import cv2
from ultralytics import YOLO


VIDEO_PATH = "data/videos/bottles_test.mp4"
OUTPUT_PATH = "outputs/bottles_detected.mp4"
MODEL_PATH = "yolo11n.pt"
TARGET_CLASS_IDS = [39, 75]  # bottle, vase

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
writer = None
processed_frames = 0

try:
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError("Could not read a valid video FPS")

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if writer is None:
            height, width = frame.shape[:2]

            writer = cv2.VideoWriter(
                OUTPUT_PATH,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"Could not create output video: {OUTPUT_PATH}"
                )

        results = model(
            frame,
            classes=TARGET_CLASS_IDS,
            device="cpu",
            verbose=False,
        )

        annotated_frame = results[0].plot()

        writer.write(annotated_frame)
        processed_frames += 1

        cv2.imshow("Bottle Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if processed_frames == 0:
        raise RuntimeError("No video frames were read")

finally:
    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()

print(f"Processed frames: {processed_frames}")
print(f"Saved video: {OUTPUT_PATH}")