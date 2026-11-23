r"""Create a visual orientation audit from normal YOLO training images only.

Run from the project root with:
    .\.venv\Scripts\python.exe .\src\audit_patchcore_orientation.py

Uses one image at a time at 320px to keep CPU memory use low.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = ROOT / "splits" / "pcb1_v1_draft.csv"
VISA_ROOT = ROOT / "data" / "visa"
MODEL_PATH = ROOT / "outputs" / "train_runs" / "pcb1_yolo_cpu" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "outputs" / "patchcore_orientation_audit"

EXPECTED_NORMAL_TRAIN = 804
CONFIDENCE = 0.25
# Keep this preprocessing margin identical during final inference/deployment.
CROP_MARGIN = 0.01
SHEET_COLUMNS = 8
SHEET_ROWS = 10
THUMBNAIL_SIZE = (180, 150)
LABEL_HEIGHT = 24


def read_training_images() -> list[Path]:
    if not SPLIT_CSV.is_file():
        raise FileNotFoundError(f"Split manifest not found: {SPLIT_CSV}")
    with SPLIT_CSV.open(encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        paths = [VISA_ROOT / row["image"] for row in rows
                 if row["split"] == "train" and row["label"] == "normal"]
    if len(paths) != EXPECTED_NORMAL_TRAIN:
        raise ValueError(f"Expected {EXPECTED_NORMAL_TRAIN} normal training images, found {len(paths)}.")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing training image, for example: {missing[0]}")
    return sorted(paths)


def expand_box(box: tuple[float, float, float, float], width: int, height: int):
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * CROP_MARGIN, (y2 - y1) * CROP_MARGIN
    return (max(0, int(math.floor(x1 - mx))), max(0, int(math.floor(y1 - my))),
            min(width, int(math.ceil(x2 + mx))), min(height, int(math.ceil(y2 + my))))


def save_contact_sheets(rows: list[dict[str, str]]) -> None:
    sheet_size = SHEET_COLUMNS * SHEET_ROWS
    cell_w, cell_h = THUMBNAIL_SIZE[0], THUMBNAIL_SIZE[1] + LABEL_HEIGHT
    for start in range(0, len(rows), sheet_size):
        page = rows[start:start + sheet_size]
        sheet = Image.new("RGB", (SHEET_COLUMNS * cell_w,
                                   math.ceil(len(page) / SHEET_COLUMNS) * cell_h), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(page):
            with Image.open(OUTPUT_DIR / "crops" / row["crop_file"]) as source:
                image = source.convert("RGB")
                image.thumbnail(THUMBNAIL_SIZE)
            x, y = (index % SHEET_COLUMNS) * cell_w, (index // SHEET_COLUMNS) * cell_h
            sheet.paste(image, (x, y))
            status = "NO DETECTION" if row["status"] != "ok" else row["orientation_review"]
            draw.text((x + 3, y + THUMBNAIL_SIZE[1] + 3), f"{row['image']} | {status}", fill="black")
        sheet.save(OUTPUT_DIR / f"contact_sheet_{start // sheet_size + 1:02d}.jpg", quality=92)


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Fine-tuned detector not found: {MODEL_PATH}")
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Audit output already exists; rename it before a new run: {OUTPUT_DIR}")
    images = read_training_images()
    (OUTPUT_DIR / "crops").mkdir(parents=True)
    model = YOLO(str(MODEL_PATH))
    records: list[dict[str, str]] = []
    for result in model.predict(source=[str(path) for path in images], imgsz=320, batch=1,
                                device="cpu", conf=CONFIDENCE, max_det=10,
                                stream=True, verbose=False):
        image_path = Path(result.path)
        record = {"image": image_path.name,
                  "crop_file": f"{len(records) + 1:04d}_{image_path.stem}.jpg",
                  "confidence": "", "status": "no_detection",
                  "orientation_review": "review direction"}
        if result.boxes is not None and len(result.boxes) > 0:
            scores = result.boxes.conf.cpu().tolist()
            best = max(range(len(scores)), key=scores.__getitem__)
            box = tuple(float(v) for v in result.boxes.xyxy[best].cpu().tolist())
            with Image.open(image_path) as source:
                source = source.convert("RGB")
                crop = source.crop(expand_box(box, *source.size))
            crop.save(OUTPUT_DIR / "crops" / record["crop_file"], quality=95)
            record.update(confidence=f"{scores[best]:.5f}", status="ok")
        else:
            with Image.open(image_path) as source:
                source.convert("RGB").save(OUTPUT_DIR / "crops" / record["crop_file"], quality=90)
        records.append(record)
        if len(records) % 20 == 0:
            print(f"Processed {len(records)}/{len(images)} images", flush=True)
    with (OUTPUT_DIR / "audit.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    (OUTPUT_DIR / "preprocessing.txt").write_text(
        "PCB1 crop preprocessing\n"
        f"YOLO checkpoint: {MODEL_PATH}\n"
        f"Crop margin: {CROP_MARGIN:.0%} per side\n"
        "Use the same YOLO checkpoint and crop margin for final inference.\n",
        encoding="utf-8",
    )
    save_contact_sheets(records)
    detected = sum(row["status"] == "ok" for row in records)
    print(f"Training images processed: {len(records)}")
    print(f"Images with a detector crop: {detected}")
    print(f"Images without a detector crop: {len(records) - detected}")
    print(f"Contact sheets: {math.ceil(len(records) / (SHEET_COLUMNS * SHEET_ROWS))}")
    print(f"Audit output: {OUTPUT_DIR}")
    print(f"Crop margin used: {CROP_MARGIN:.0%} per side")
    print("Keep this same crop margin during final inference/deployment.")


if __name__ == "__main__":
    main()
