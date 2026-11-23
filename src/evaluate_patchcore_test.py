"""Score frozen PatchCore on the reserved PCB1 test split exactly once."""

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
TEST_ROOT = DATA_ROOT / "test"
OUTPUT_ROOT = ROOT / "outputs" / "patchcore_test_eval"


def verify_data() -> None:
    manifest = TEST_ROOT / "manifest.csv"
    if not manifest.is_file() or not CHECKPOINT.is_file():
        raise FileNotFoundError("Prepared test manifest or trained checkpoint is missing")
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normals = sum(r["label"] == "normal" for r in rows)
    anomalies = sum(r["label"] != "normal" for r in rows)
    masks = sum(bool(r["mask"]) for r in rows)
    if (normals, anomalies, masks) != (100, 60, 60):
        raise ValueError(f"Expected 100 normal, 60 anomalous, 60 masks; got {normals}/{anomalies}/{masks}")
    for rel_dir in ("train/good", "test/good", "test/anomaly", "test/masks"):
        if not (DATA_ROOT / rel_dir).is_dir():
            raise FileNotFoundError(f"Required folder missing: {DATA_ROOT / rel_dir}")


def main() -> None:
    verify_data()
    datamodule = Folder(
        name="pcb1_patchcore_test",
        root=DATA_ROOT,
        normal_dir="train/good",
        abnormal_dir="test/anomaly",
        normal_test_dir="test/good",
        mask_dir="test/masks",
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
    engine = Engine(accelerator="cpu", devices=1, default_root_dir=OUTPUT_ROOT, logger=False)
    results = engine.test(model=model, datamodule=datamodule, ckpt_path=CHECKPOINT)
    print("Final reserved-test metrics (no tuning performed):")
    print(results)
    print(f"Test evaluation outputs: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
