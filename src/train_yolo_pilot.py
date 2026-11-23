"""Train a YOLO detector on the reviewed VisA PCB1 pilot dataset.

Run from the repository root with:
    .\.venv\Scripts\python.exe .\src\train_yolo_pilot.py

The script starts from the local COCO-pretrained YOLO11n weights, validates
the pilot files, trains on CPU by default, and writes each run under outputs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from time import perf_counter

import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "yolo_pcb1_pilot" / "data.yaml"
DEFAULT_WEIGHTS = ROOT / "yolo11n.pt"
DEFAULT_PROJECT = ROOT / "outputs" / "train_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO11n on the reviewed PCB1 detection pilot."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="pcb1_yolo_cpu")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    return parser.parse_args()


def resolve_dataset_path(data_yaml: Path, dataset: dict, key: str) -> Path:
    value = Path(dataset[key])
    if value.is_absolute():
        return value
    root_value = dataset.get("path", data_yaml.parent)
    root = Path(root_value)
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root / value


def validate_dataset(data_yaml: Path) -> None:
    """Check class metadata, expected files, labels, and image-label pairing."""
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data_yaml}")

    with data_yaml.open(encoding="utf-8") as file:
        dataset = yaml.safe_load(file)

    if dataset.get("names") != {0: "hc_sr04"}:
        raise ValueError("Expected the single class mapping {0: 'hc_sr04'}.")

    for split, expected_count in (("train", 40), ("val", 22)):
        image_dir = resolve_dataset_path(data_yaml, dataset, split)
        label_dir = image_dir.parent.parent / "labels" / split
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

        images = {
            path.stem
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        }
        labels = {path.stem for path in label_dir.glob("*.txt")}
        if len(images) != expected_count:
            raise ValueError(
                f"Expected {expected_count} {split} images, found {len(images)}."
            )
        if images != labels:
            missing = sorted(images - labels)
            extra = sorted(labels - images)
            raise ValueError(
                f"Image/label names differ for {split}. Missing labels={missing}; "
                f"extra labels={extra}."
            )

        for label_path in label_dir.glob("*.txt"):
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                fields = line.split()
                if len(fields) != 5 or fields[0] != "0":
                    raise ValueError(
                        f"Invalid single-class YOLO label at {label_path}:{line_number}."
                    )
                try:
                    box = [float(value) for value in fields[1:]]
                except ValueError as error:
                    raise ValueError(
                        f"Non-numeric box value at {label_path}:{line_number}."
                    ) from error
                if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in box):
                    raise ValueError(
                        f"Normalized box values must be within [0, 1] at "
                        f"{label_path}:{line_number}."
                    )

        print(f"Validated {split}: {len(images)} images and {len(labels)} labels.")


def main() -> None:
    args = parse_args()
    data_yaml = args.data.resolve()
    weights = args.model.resolve()
    project = args.project.resolve()
    run_dir = project / args.name

    if not weights.is_file():
        raise FileNotFoundError(f"Initial model weights do not exist: {weights}")
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1 or args.workers < 0:
        raise ValueError("epochs, batch, and imgsz must be positive; workers cannot be negative.")
    validate_dataset(data_yaml)
    if run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists; choose another --name: {run_dir}"
        )

    print(f"Starting weights: {weights}")
    print(f"Dataset config: {data_yaml}")
    print(f"Run output: {run_dir}")
    print(
        f"Settings: epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, "
        f"device={args.device}, workers={args.workers}"
    )

    model = YOLO(str(weights))
    started = perf_counter()
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project),
        name=args.name,
        exist_ok=False,
        seed=42,
        deterministic=True,
        amp=False,
        cache=False,
        plots=True,
        val=True,
        save=True,
        verbose=True,
    )
    elapsed = perf_counter() - started

    print(f"Training elapsed seconds: {elapsed:.1f}")
    print(f"Run directory: {run_dir}")
    print(f"Best weights: {run_dir / 'weights' / 'best.pt'}")
    print(f"Metrics: {run_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
