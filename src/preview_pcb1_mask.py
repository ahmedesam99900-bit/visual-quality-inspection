from pathlib import Path
import csv

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "visa"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

with (DATA_ROOT / "pcb1" / "image_anno.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    rows = list(csv.DictReader(file))

# Read the mask path from the dataset annotations.
sample = next(
    row for row in rows
    if row["image"] == "pcb1/Data/Images/Anomaly/099.JPG"
)

with Image.open(DATA_ROOT / sample["image"]) as source:
    original = source.convert("RGB")

with Image.open(DATA_ROOT / sample["mask"]) as source:
    mask_array = np.array(source)

if mask_array.ndim != 2:
    raise ValueError(f"Expected a single-channel mask: {mask_array.shape}")

if mask_array.shape != (original.height, original.width):
    raise ValueError("Image and mask dimensions do not match.")

# Treat every nonzero mask value as an anomalous pixel.
binary_mask = Image.fromarray((mask_array > 0).astype(np.uint8) * 255)

red = Image.new("RGB", original.size, "red")
tinted = Image.blend(original, red, alpha=0.45)
overlay = Image.composite(tinted, original, binary_mask)

# Preserve the original resolution for detailed visual inspection.
comparison = Image.new(
    "RGB", (original.width * 2, original.height + 40), "white"
)
comparison.paste(original, (0, 40))
comparison.paste(overlay, (original.width, 40))

draw = ImageDraw.Draw(comparison)
draw.text((10, 10), "Original", fill="black")
draw.text(
    (original.width + 10, 10),
    "Ground truth overlay",
    fill="black",
)

output_path = OUTPUT_DIR / "pcb1_mask_check.png"
comparison.save(output_path)

print("Mask values:", np.unique(mask_array).tolist())
print(f"Saved: {output_path}")