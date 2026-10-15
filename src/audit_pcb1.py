from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "visa"

with (DATA_ROOT / "pcb1" / "image_anno.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    rows = list(csv.DictReader(file))

sizes = Counter()
fingerprints = defaultdict(list)
errors = []
checked_masks = 0

for row in rows:
    image_path = DATA_ROOT / row["image"]

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        sizes[image.size] += 1

        # البصمة تعتمد على الأبعاد والبكسلات، مش اسم الملف.
        digest = hashlib.sha256()
        digest.update(str(image.size).encode("ascii"))
        digest.update(image.tobytes())
        fingerprints[digest.hexdigest()].append(row["image"])

        if row["label"] != "normal":
            if not row["mask"]:
                raise ValueError("Anomalous image has no mask path")

            with Image.open(DATA_ROOT / row["mask"]) as source:
                mask = np.array(source)

            if mask.ndim != 2:
                raise ValueError(f"Mask is not single-channel: {mask.shape}")

            if mask.shape != (image.height, image.width):
                raise ValueError("Image and mask dimensions differ")

            if not np.any(mask > 0):
                raise ValueError("Anomalous image has an empty mask")

            checked_masks += 1

    except Exception as error:
        errors.append(f"{row['image']}: {error}")

duplicates = [
    paths for paths in fingerprints.values()
    if len(paths) > 1
]

print(f"CSV rows: {len(rows)}")
print(f"Image sizes (width, height): {dict(sizes)}")
print(f"Valid anomaly masks: {checked_masks}")
print(f"Errors: {len(errors)}")
print(f"Exact duplicate groups: {len(duplicates)}")

for error in errors:
    print("ERROR:", error)

for paths in duplicates:
    print("DUPLICATES:", paths)