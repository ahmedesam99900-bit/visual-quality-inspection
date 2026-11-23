"""Prepare PatchCore train and validation crops using the frozen YOLO ROI.

Run from the repository root:
    .venv/Scripts/python.exe src/prepare_patchcore_data.py

The reserved test split is intentionally not read by this script.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = ROOT / "splits" / "pcb1_v1_draft.csv"
VISA_ROOT = ROOT / "data" / "visa"
YOLO_MODEL = ROOT / "outputs" / "train_runs" / "pcb1_yolo_cpu" / "weights" / "best.pt"
TRAIN_AUDIT = ROOT / "outputs" / "patchcore_orientation_audit"
OUTPUT = ROOT / "outputs" / "patchcore_dataset_dev"

IMGSZ = 320
CONFIDENCE = 0.25
CROP_MARGIN = 0.01
EXPECTED = {("train", "normal"): 804, ("val", "normal"): 100}
EXPECTED_VAL_ANOMALY = 40


def load_manifest() -> list[dict[str, str]]:
    with SPLIT_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Deliberately exclude test here. It stays untouched until final evaluation.
    rows = [row for row in rows if row["split"] in {"train", "val"}]
    counts = {(split, label): sum(r["split"] == split and r["label"] == label for r in rows)
              for split, label in EXPECTED}
    if counts != EXPECTED:
        raise ValueError(f"Unexpected train/validation counts: {counts}")
    val_anomalies = sum(r["split"] == "val" and r["label"] != "normal" for r in rows)
    if val_anomalies != EXPECTED_VAL_ANOMALY:
        raise ValueError(f"Expected {EXPECTED_VAL_ANOMALY} validation anomalies, found {val_anomalies}")
    for row in rows:
        image = VISA_ROOT / row["image"]
        if not image.is_file():
            raise FileNotFoundError(f"Image missing: {image}")
        if row["label"] != "normal":
            mask = VISA_ROOT / row["mask"]
            if not row["mask"] or not mask.is_file():
                raise FileNotFoundError(f"Anomaly mask missing for {image}: {mask}")
    return rows


def crop_bounds(box: tuple[float, float, float, float], width: int, height: int):
    x1, y1, x2, y2 = box
    margin_x = (x2 - x1) * CROP_MARGIN
    margin_y = (y2 - y1) * CROP_MARGIN
    left = max(0, int(x1 - margin_x))
    top = max(0, int(y1 - margin_y))
    right = min(width, int(x2 + margin_x + 0.999))
    bottom = min(height, int(y2 + margin_y + 0.999))
    if right <= left or bottom <= top:
        raise ValueError(f"Invalid crop bounds: {(left, top, right, bottom)}")
    return left, top, right, bottom


def destination_path(row: dict[str, str]) -> Path:
    group = "good" if row["label"] == "normal" else "anomaly"
    source = Path(row["image"])
    return OUTPUT / row["split"] / group / f"{source.stem}.jpg"


def copy_train_crop(row: dict[str, str]) -> dict[str, str]:
    # The audit crop map is cached to avoid rereading the CSV for every image.
    crop_name = TRAIN_CROPS.get(Path(row["image"]).name)
    if crop_name is None:
        raise ValueError(f"No reviewed training crop recorded for {row['image']}")
    source_crop = TRAIN_AUDIT / "crops" / crop_name
    if not source_crop.is_file():
        raise FileNotFoundError(f"Training crop missing: {source_crop}")
    dest = destination_path(row)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_crop, dest)
    return {"split": row["split"], "label": row["label"], "source": row["image"],
            "crop": str(dest.relative_to(OUTPUT)), "mask": "", "confidence": "",
            "box_xyxy": "reused approved 1% crop"}


def predict_crop(model: YOLO, row: dict[str, str]) -> dict[str, str]:
    image_path = VISA_ROOT / row["image"]
    result = model.predict(source=str(image_path), imgsz=IMGSZ, device="cpu",
                           conf=CONFIDENCE, max_det=10, batch=1, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError(f"YOLO found no product in validation image: {image_path}")
    scores = result.boxes.conf.cpu().tolist()
    best = max(range(len(scores)), key=scores.__getitem__)
    box = tuple(float(v) for v in result.boxes.xyxy[best].cpu().tolist())
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_size = image.size
        bounds = crop_bounds(box, *image.size)
        crop = image.crop(bounds)
    dest = destination_path(row)
    dest.parent.mkdir(parents=True, exist_ok=True)
    crop.save(dest, quality=95)

    mask_rel = ""
    if row["label"] != "normal":
        mask_path = VISA_ROOT / row["mask"]
        with Image.open(mask_path) as mask_image:
            mask_image = mask_image.convert("L")
            if mask_image.size != image_size:
                raise ValueError(f"Mask/image dimensions differ: {mask_path}")
            mask_crop = mask_image.crop(bounds)
        mask_dest = OUTPUT / row["split"] / "masks" / f"{Path(row['image']).stem}.png"
        mask_dest.parent.mkdir(parents=True, exist_ok=True)
        mask_crop.save(mask_dest)
        mask_rel = str(mask_dest.relative_to(OUTPUT))

    return {"split": row["split"], "label": row["label"], "source": row["image"],
            "crop": str(dest.relative_to(OUTPUT)), "mask": mask_rel,
            "confidence": f"{scores[best]:.5f}",
            "box_xyxy": ",".join(f"{v:.2f}" for v in box)}


def main() -> None:
    if not SPLIT_CSV.is_file() or not YOLO_MODEL.is_file():
        raise FileNotFoundError("Split CSV or trained YOLO checkpoint is missing.")
    if not (TRAIN_AUDIT / "audit.csv").is_file():
        raise FileNotFoundError(f"Reviewed training crop audit not found: {TRAIN_AUDIT}")
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists; remove or rename it before running: {OUTPUT}")

    global TRAIN_CROPS
    with (TRAIN_AUDIT / "audit.csv").open(encoding="utf-8-sig", newline="") as handle:
        TRAIN_CROPS = {row["image"]: row["crop_file"] for row in csv.DictReader(handle)}
    rows = load_manifest()
    model = YOLO(str(YOLO_MODEL))
    output_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        if row["split"] == "train":
            output_rows.append(copy_train_crop(row))
        else:
            output_rows.append(predict_crop(model, row))
        if index % 20 == 0:
            print(f"Prepared {index}/{len(rows)} train/validation images", flush=True)

    with (OUTPUT / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    (OUTPUT / "preprocessing.txt").write_text(
        f"YOLO checkpoint: {YOLO_MODEL}\nInference size: {IMGSZ}\n"
        f"Crop margin: {CROP_MARGIN:.0%} per side\n"
        "Training: normal images only. Validation: normal and anomalous images with cropped masks.\n"
        "Test split was not accessed.\n",
        encoding="utf-8",
    )
    print(f"Prepared rows: {len(output_rows)}")
    print(f"Training normal crops: {sum(r['split']=='train' for r in output_rows)}")
    print(f"Validation crops: {sum(r['split']=='val' for r in output_rows)}")
    print(f"Validation anomaly masks: {sum(bool(r['mask']) for r in output_rows)}")
    print(f"Output: {OUTPUT}")


TRAIN_CROPS: dict[str, str] = {}

if __name__ == "__main__":
    main()
