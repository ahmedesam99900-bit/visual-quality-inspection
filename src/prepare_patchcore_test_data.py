"""Prepare the reserved PCB1 test crops using the frozen YOLO ROI pipeline.

Run from the project root after copying this file to src/:
    .venv/Scripts/python.exe src/prepare_patchcore_test_data.py

This only creates outputs/patchcore_dataset_dev/test and never modifies the
training or validation folders. It must be run once, before final test scoring.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
SPLIT_CSV = ROOT / "splits" / "pcb1_v1_draft.csv"
VISA_ROOT = ROOT / "data" / "visa"
YOLO_MODEL = ROOT / "outputs" / "train_runs" / "pcb1_yolo_cpu" / "weights" / "best.pt"
DATA_ROOT = ROOT / "outputs" / "patchcore_dataset_dev"
TEST_ROOT = DATA_ROOT / "test"
IMGSZ = 320
CONFIDENCE = 0.25
CROP_MARGIN = 0.01
EXPECTED = {"normal": 100, "anomaly": 60}


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


def load_test_rows() -> list[dict[str, str]]:
    with SPLIT_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["split"] == "test"]
    counts = {
        "normal": sum(r["label"] == "normal" for r in rows),
        "anomaly": sum(r["label"] != "normal" for r in rows),
    }
    if counts != EXPECTED:
        raise ValueError(f"Expected 100 normal and 60 anomalous test images, found {counts}")
    for row in rows:
        image = VISA_ROOT / row["image"]
        if not image.is_file():
            raise FileNotFoundError(f"Test image missing: {image}")
        if row["label"] != "normal":
            mask = VISA_ROOT / row["mask"]
            if not row["mask"] or not mask.is_file():
                raise FileNotFoundError(f"Test mask missing for {image}: {mask}")
    return rows


def main() -> None:
    if not SPLIT_CSV.is_file() or not YOLO_MODEL.is_file():
        raise FileNotFoundError("Split CSV or frozen YOLO checkpoint is missing")
    for folder in (DATA_ROOT / "train" / "good", DATA_ROOT / "val" / "good",
                   DATA_ROOT / "val" / "anomaly", DATA_ROOT / "val" / "masks"):
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected existing development data folder: {folder}")
    if TEST_ROOT.exists():
        raise FileExistsError(
            f"Test output already exists; inspect it before deciding what to do: {TEST_ROOT}"
        )

    rows = load_test_rows()
    model = YOLO(str(YOLO_MODEL))
    audit: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        image_path = VISA_ROOT / row["image"]
        result = model.predict(source=str(image_path), imgsz=IMGSZ, device="cpu",
                               conf=CONFIDENCE, max_det=10, batch=1, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            raise RuntimeError(f"YOLO found no product in test image: {image_path}")
        scores = result.boxes.conf.cpu().tolist()
        best = max(range(len(scores)), key=scores.__getitem__)
        box = tuple(float(v) for v in result.boxes.xyxy[best].cpu().tolist())
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_size = image.size
            bounds = crop_bounds(box, *image_size)
            crop = image.crop(bounds)

        category = "good" if row["label"] == "normal" else "anomaly"
        stem = Path(row["image"]).stem
        image_dest = TEST_ROOT / category / f"{stem}.jpg"
        image_dest.parent.mkdir(parents=True, exist_ok=True)
        crop.save(image_dest, quality=95)

        mask_rel = ""
        if category == "anomaly":
            with Image.open(VISA_ROOT / row["mask"]) as mask_image:
                mask_image = mask_image.convert("L")
                if mask_image.size != image_size:
                    raise ValueError(f"Mask/image dimensions differ: {row['mask']}")
                mask_crop = mask_image.crop(bounds)
            mask_dest = TEST_ROOT / "masks" / f"{stem}.png"
            mask_dest.parent.mkdir(parents=True, exist_ok=True)
            mask_crop.save(mask_dest)
            mask_rel = str(mask_dest.relative_to(DATA_ROOT))

        audit.append({
            "split": "test", "label": row["label"], "source": row["image"],
            "crop": str(image_dest.relative_to(DATA_ROOT)), "mask": mask_rel,
            "confidence": f"{scores[best]:.5f}",
            "box_xyxy": ",".join(f"{v:.2f}" for v in box),
        })
        if index % 20 == 0:
            print(f"Prepared {index}/{len(rows)} test crops", flush=True)

    with (TEST_ROOT / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit[0].keys())
        writer.writeheader()
        writer.writerows(audit)
    (TEST_ROOT / "preprocessing.txt").write_text(
        f"YOLO checkpoint: {YOLO_MODEL}\nInference size: {IMGSZ}\n"
        f"Confidence threshold: {CONFIDENCE}\nCrop margin: {CROP_MARGIN:.0%} per side\n"
        "Only the reserved test split was processed. Model settings were not changed.\n",
        encoding="utf-8",
    )
    print("Prepared test normals: 100")
    print("Prepared test anomalies with masks: 60")
    print(f"Test crops: {TEST_ROOT}")


if __name__ == "__main__":
    main()
