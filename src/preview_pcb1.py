from pathlib import Path
import csv

from PIL import Image, ImageOps, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "visa"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# مسارات الصور في CSV تبدأ بـ pcb1، لذلك نربطها بـ DATA_ROOT.
with (DATA_ROOT / "pcb1" / "image_anno.csv").open(
    encoding="utf-8-sig", newline=""
) as file:
    rows = list(csv.DictReader(file))

normal = [row for row in rows if row["label"] == "normal"]
anomaly = [row for row in rows if row["label"] != "normal"]


def select_spread_samples(group, count=6):
    """عينات موزعة على القائمة للمعاينة، وليست تقسيمًا للتدريب."""
    return [
        group[index * (len(group) - 1) // (count - 1)]
        for index in range(count)
    ]


selected = select_spread_samples(normal) + select_spread_samples(anomaly)

# أول صفين للسليم، وآخر صفين للمعيب.
cell_width, cell_height = 400, 350
sheet = Image.new("RGB", (cell_width * 3, cell_height * 4), "white")
draw = ImageDraw.Draw(sheet)

for index, row in enumerate(selected):
    image_path = DATA_ROOT / row["image"]

    with Image.open(image_path) as source:
        thumbnail = ImageOps.contain(
            source.convert("RGB"),
            (380, 290),
        )

    x = (index % 3) * cell_width
    y = (index // 3) * cell_height

    sheet.paste(thumbnail, (x + (cell_width - thumbnail.width) // 2, y + 40))
    draw.text(
        (x + 10, y + 10),
        f"{image_path.name} | {row['label']}",
        fill="black",
    )

preview_path = OUTPUT_DIR / "pcb1_preview.jpg"
sheet.save(preview_path, quality=95)

# نحتفظ بقائمة العينات التي عاينّاها لاستبعادها من الاختبار النهائي.
with (OUTPUT_DIR / "pcb1_preview_samples.csv").open(
    "w", encoding="utf-8", newline=""
) as file:
    writer = csv.DictWriter(file, fieldnames=["image", "label", "mask"])
    writer.writeheader()
    writer.writerows(selected)

print(f"Preview saved: {preview_path}")