from pathlib import Path
from collections import Counter
import csv
import random


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "visa"
OUTPUT_PATH = ROOT / "splits" / "pcb1_v1_draft.csv"
SEED = 42

with (DATA_ROOT / "pcb1" / "image_anno.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    rows = list(csv.DictReader(file))

# Reserve samples inspected in the contact sheet for development.
with (ROOT / "outputs" / "pcb1_preview_samples.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    reviewed = {row["image"] for row in csv.DictReader(file)}

# Include additional samples inspected before generating the contact sheet.
reviewed.update(
    f"pcb1/Data/Images/Normal/{name}.JPG"
    for name in ["0000", "0001", "0500", "1003"]
)
reviewed.update(
    f"pcb1/Data/Images/Anomaly/{name}.JPG"
    for name in ["000", "015", "041", "085"]
)

image_paths = {row["image"] for row in rows}

if len(image_paths) != len(rows):
    raise ValueError("CSV contains repeated image paths")

if not reviewed.issubset(image_paths):
    raise ValueError("A reviewed image is missing from the dataset")

normal = sorted(
    [row for row in rows if row["label"] == "normal"],
    key=lambda row: row["image"],
)
anomaly = sorted(
    [row for row in rows if row["label"] != "normal"],
    key=lambda row: row["image"],
)

if (len(normal), len(anomaly)) != (1004, 100):
    raise ValueError("Unexpected dataset counts")


def reserve_reviewed(group):
    """Reserve reviewed samples and shuffle the rest reproducibly."""
    known = [row for row in group if row["image"] in reviewed]
    remaining = [row for row in group if row["image"] not in reviewed]
    random.Random(SEED).shuffle(remaining)
    return known, remaining


known_normal, other_normal = reserve_reviewed(normal)
known_anomaly, other_anomaly = reserve_reviewed(anomaly)

# Assign reviewed normals to training and reviewed anomalies to validation.
train_slots = 804 - len(known_normal)
val_anomaly_slots = 40 - len(known_anomaly)

if train_slots < 0 or val_anomaly_slots < 0:
    raise ValueError("Reviewed samples exceed the allocated split size")

groups = {
    "train": known_normal + other_normal[:train_slots],
    "val": (
        other_normal[train_slots:train_slots + 100]
        + known_anomaly
        + other_anomaly[:val_anomaly_slots]
    ),
    "test": (
        other_normal[train_slots + 100:]
        + other_anomaly[val_anomaly_slots:]
    ),
}

assigned = [row["image"] for group in groups.values() for row in group]

if len(assigned) != len(rows) or len(set(assigned)) != len(rows):
    raise ValueError("Split has missing or repeated assignments")

if reviewed.intersection(row["image"] for row in groups["test"]):
    raise ValueError("Reviewed images leaked into test")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Exclusive creation prevents overwriting an existing split.
with OUTPUT_PATH.open("x", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(
        file, fieldnames=["image", "label", "mask", "split"]
    )
    writer.writeheader()

    for split, group in groups.items():
        for row in group:
            writer.writerow({**row, "split": split})

for split, group in groups.items():
    normal_count = sum(row["label"] == "normal" for row in group)
    defect_counts = Counter(
        defect.strip()
        for row in group
        if row["label"] != "normal"
        for defect in row["label"].split(",")
    )

    print(
        f"{split}: normal={normal_count}, "
        f"anomaly={len(group) - normal_count}"
    )
    print(f"  Defect counts: {dict(sorted(defect_counts.items()))}")

print(f"Saved: {OUTPUT_PATH}")