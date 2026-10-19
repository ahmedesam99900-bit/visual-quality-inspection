from pathlib import Path
from collections import Counter
import csv
import hashlib
import random
import shutil


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "visa"
SPLIT_PATH = ROOT / "splits" / "pcb1_v1_draft.csv"
DESTINATION = ROOT / "data" / "yolo_pcb1_pilot"
MANIFEST_PATH = DESTINATION / "selection.csv"
SEED = 42

FIELDNAMES = ["image", "source_image", "source_label", "split"]

EXTRA_VALIDATION_PATHS = [
    "pcb1/Data/Images/Normal/0742.JPG",
    "pcb1/Data/Images/Normal/0745.JPG",
]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def file_hash(path):
    """Calculate a file fingerprint without loading the entire file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def select_samples(rows, split, normal, count):
    """Select a reproducible sample from one split and condition."""
    candidates = sorted(
        [
            row for row in rows
            if row["split"] == split
            and (row["label"] == "normal") == normal
        ],
        key=lambda row: row["image"],
    )

    if len(candidates) < count:
        raise ValueError(f"Not enough images in {split}")

    return random.Random(SEED).sample(candidates, count)


def main():
    rows = read_csv(SPLIT_PATH)
    rows_by_image = {row["image"]: row for row in rows}

    if len(rows_by_image) != len(rows):
        raise ValueError("The source split contains duplicate image paths")

    selected = {
        "train": select_samples(rows, "train", normal=True, count=40),
        "val": (
            select_samples(rows, "val", normal=True, count=10)
            + select_samples(rows, "val", normal=False, count=10)
        ),
    }

    # Include reviewed normal samples with downward-facing connector pins.
    for image_path in EXTRA_VALIDATION_PATHS:
        if image_path not in rows_by_image:
            raise ValueError(f"Sample is missing from the split: {image_path}")

        sample = rows_by_image[image_path]

        if sample["split"] != "val" or sample["label"] != "normal":
            raise ValueError(f"Invalid validation sample: {image_path}")

        if not any(row["image"] == image_path for row in selected["val"]):
            selected["val"].append(sample)

    # Build the expected manifest before making any changes.
    manifest = []

    for split, group in selected.items():
        for row in group:
            if row["split"] != split:
                raise ValueError(f"Split mismatch: {row['image']}")

            source = DATA_ROOT / row["image"]

            if not source.is_file():
                raise FileNotFoundError(source)

            condition = "normal" if row["label"] == "normal" else "anomaly"
            filename = f"pcb1_{condition}_{source.stem}.jpg"

            manifest.append({
                "image": f"images/{split}/{filename}",
                "source_image": row["image"],
                "source_label": row["label"],
                "split": split,
            })

    source_paths = [row["source_image"] for row in manifest]
    destination_paths = [row["image"] for row in manifest]

    if len(set(source_paths)) != len(source_paths):
        raise ValueError("A source image was selected more than once")

    if len(set(destination_paths)) != len(destination_paths):
        raise ValueError("Duplicate destination filenames")

    counts = Counter(row["split"] for row in manifest)

    if counts != Counter({"train": 40, "val": 22}):
        raise ValueError(f"Unexpected selection counts: {dict(counts)}")

    # Allow an existing manifest only when its records match this selection.
    expected_by_image = {row["image"]: row for row in manifest}

    if MANIFEST_PATH.exists():
        existing = read_csv(MANIFEST_PATH)
        existing_paths = [row["image"] for row in existing]

        if len(set(existing_paths)) != len(existing_paths):
            raise ValueError("The existing manifest contains duplicate records")

        for row in existing:
            if expected_by_image.get(row["image"]) != row:
                raise ValueError(
                    f"Existing manifest record does not match: {row['image']}"
                )

    # Reject unexpected images instead of silently mixing datasets.
    for split in selected:
        image_dir = DESTINATION / "images" / split

        if image_dir.exists():
            for path in image_dir.iterdir():
                if path.is_file() and path.suffix.lower() in {
                    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
                }:
                    relative_path = path.relative_to(DESTINATION).as_posix()

                    if relative_path not in expected_by_image:
                        raise ValueError(f"Unexpected image: {path}")

    # Verify existing copies before adding any missing images.
    for row in manifest:
        source = DATA_ROOT / row["source_image"]
        destination = DESTINATION / row["image"]

        if destination.exists():
            if not destination.is_file():
                raise ValueError(f"Destination is not a file: {destination}")

            if file_hash(source) != file_hash(destination):
                raise ValueError(
                    f"Existing image differs from its source: {destination}"
                )

    added = 0

    for row in manifest:
        source = DATA_ROOT / row["source_image"]
        destination = DESTINATION / row["image"]
        destination.parent.mkdir(parents=True, exist_ok=True)

        if not destination.exists():
            shutil.copy2(source, destination)
            added += 1

    # Replace the manifest only after all image copies have succeeded.
    temporary_manifest = MANIFEST_PATH.with_suffix(".csv.tmp")

    with temporary_manifest.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(manifest)

    temporary_manifest.replace(MANIFEST_PATH)

    print(f"YOLO train images: {counts['train']}")
    print(f"YOLO validation images: {counts['val']}")
    print("Test images used: 0")
    print(f"New images copied: {added}")

    for split in ["train", "val"]:
        group = [row for row in manifest if row["split"] == split]
        normal_count = sum(
            row["source_label"] == "normal" for row in group
        )
        anomaly_count = len(group) - normal_count
        print(f"{split}: normal={normal_count}, anomaly={anomaly_count}")

    print(f"Manifest saved: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()