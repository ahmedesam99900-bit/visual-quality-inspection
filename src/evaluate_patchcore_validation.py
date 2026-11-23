"""Evaluate the saved PCB1 PatchCore checkpoint on the validation split only."""

from __future__ import annotations

import csv
from pathlib import Path

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "outputs" / "patchcore_dataset_dev"
CHECKPOINT = (
    ROOT / "outputs" / "patchcore_runs" / "pcb1_patchcore_cpu" / "Patchcore"
    / "pcb1_patchcore_dev" / "v1" / "weights" / "lightning" / "model.ckpt"
)
OUTPUT_ROOT = ROOT / "outputs" / "patchcore_validation_eval"


def verify_data() -> None:
    manifest = DATA_ROOT / "manifest.csv"
    if not manifest.is_file() or not CHECKPOINT.is_file():
        raise FileNotFoundError("Validation manifest or trained PatchCore checkpoint is missing.")
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normals = sum(row["split"] == "val" and row["label"] == "normal" for row in rows)
    anomalies = sum(row["split"] == "val" and row["label"] != "normal" for row in rows)
    if (normals, anomalies) != (100, 40):
        raise ValueError(f"Expected 100 normal and 40 anomalous validation images; found {normals}/{anomalies}.")


def main() -> None:
    verify_data()
    datamodule = Folder(
        name="pcb1_patchcore_validation",
        root=DATA_ROOT,
        normal_dir="train/good",
        abnormal_dir="val/anomaly",
        normal_test_dir="val/good",
        mask_dir="val/masks",
        normal_split_ratio=0.0,
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
        default_root_dir=OUTPUT_ROOT,
        logger=False,
    )
    results = engine.test(model=model, datamodule=datamodule, ckpt_path=CHECKPOINT)
    print("Validation metrics returned by Anomalib:")
    print(results)
    print(f"Validation evaluation outputs: {OUTPUT_ROOT}")
    print("This run uses validation data only; the reserved test split remains untouched.")


if __name__ == "__main__":
    main()
