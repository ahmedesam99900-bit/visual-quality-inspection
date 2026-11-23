"""Train and validate PatchCore on the prepared PCB1 development split.

Run from the repository root:
    .venv/Scripts/python.exe src/train_patchcore_pcb1.py

Only train/good is used to build the PatchCore memory bank. The supplied
validation images are used for validation; the reserved test split is not read.
"""

from __future__ import annotations

import csv
from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "outputs" / "patchcore_dataset_dev"
OUTPUT_ROOT = ROOT / "outputs" / "patchcore_runs" / "pcb1_patchcore_cpu"
EXPECTED_TRAIN = 804
EXPECTED_VAL_NORMAL = 100
EXPECTED_VAL_ANOMALY = 40


def verify_prepared_data() -> None:
    manifest_path = DATA_ROOT / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared-data manifest missing: {manifest_path}")
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    train_count = sum(row["split"] == "train" and row["label"] == "normal" for row in rows)
    val_normal_count = sum(row["split"] == "val" and row["label"] == "normal" for row in rows)
    val_anomaly_count = sum(row["split"] == "val" and row["label"] != "normal" for row in rows)
    if (train_count, val_normal_count, val_anomaly_count) != (
        EXPECTED_TRAIN, EXPECTED_VAL_NORMAL, EXPECTED_VAL_ANOMALY
    ):
        raise ValueError(
            "Unexpected train/validation counts: "
            f"train={train_count}, val_normal={val_normal_count}, "
            f"val_anomaly={val_anomaly_count}"
        )

    for rel_dir in ("train/good", "val/good", "val/anomaly", "val/masks"):
        folder = DATA_ROOT / rel_dir
        if not folder.is_dir():
            raise FileNotFoundError(f"Required data folder missing: {folder}")


def main() -> None:
    verify_prepared_data()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    datamodule = Folder(
        name="pcb1_patchcore_dev",
        root=DATA_ROOT,
        normal_dir="train/good",
        abnormal_dir="val/anomaly",
        normal_test_dir="val/good",
        mask_dir="val/masks",
        normal_split_ratio=0.0,
        # Images are JPG; ground-truth anomaly masks are PNG.
        extensions=(".jpg", ".png"),
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
        test_split_mode="from_dir",
        val_split_mode="same_as_test",
        seed=42,
    )

    pre_processor = Patchcore.configure_pre_processor(image_size=(160, 272))
    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        pre_processor=pre_processor,
    )

    engine = Engine(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        num_sanity_val_steps=0,
        default_root_dir=OUTPUT_ROOT,
        logger=False,
    )
    engine.fit(model=model, datamodule=datamodule)
    print(f"PatchCore validation run complete. Output folder: {OUTPUT_ROOT}")
    print(f"Best checkpoint: {engine.best_model_path}")


if __name__ == "__main__":
    main()
