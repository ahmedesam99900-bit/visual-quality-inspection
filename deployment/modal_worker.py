"""Private Modal GPU worker for training new product models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import modal

from cloud import APP_NAME, MAX_MODEL_BYTES, VOLUME_NAME, dataset_location, result_location


HERE = Path(__file__).resolve().parent
VOLUME_PATH = Path("/vqi-storage")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("numpy==2.2.6", "Pillow==12.2.0")
    .env({"PYTHONPATH": "/opt/vqi-training"})
    .add_local_file(HERE / "cloud.py", "/opt/vqi-training/cloud.py")
    .add_local_file(HERE / "trainer.py", "/opt/vqi-training/trainer.py")
)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="L4",
    cpu=4,
    memory=8192,
    volumes={str(VOLUME_PATH): volume},
    timeout=60 * 60,
    max_containers=1,
)
def train_on_gpu(job_id: str) -> dict:
    from trainer import train_model

    dataset = VOLUME_PATH / dataset_location(job_id)
    result_dir = VOLUME_PATH / result_location(job_id, modal.current_function_call_id())
    volume.reload()
    if not dataset.is_dir():
        raise FileNotFoundError("The staged dataset is unavailable in the Modal volume.")

    with TemporaryDirectory(prefix="vqi-train-") as local:
        artifact = Path(local) / "model.npz"
        metrics = train_model(dataset, artifact, device="cuda", progress=lambda message: print(message, flush=True))
        size = artifact.stat().st_size
        if not 0 < size <= MAX_MODEL_BYTES:
            raise ValueError("The trained model exceeds the supported artifact size.")
        with artifact.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        manifest = {"size_bytes": size, "sha256": digest, "metrics": metrics}

        result_dir.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".publish-", dir=result_dir) as staged:
            pending = Path(staged)
            shutil.copyfile(artifact, pending / "model.npz")
            (pending / "result.json").write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
            (pending / "model.npz").replace(result_dir / "model.npz")
            (pending / "result.json").replace(result_dir / "result.json")
    volume.commit()
    return manifest
