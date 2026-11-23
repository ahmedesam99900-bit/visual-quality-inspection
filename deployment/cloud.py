"""Submit and recover private GPU training jobs through the Modal SDK."""

from __future__ import annotations

import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory


APP_NAME = "visual-quality-inspection-training"
VOLUME_NAME = "vqi-training-artifacts"
FUNCTION_NAME = "train_on_gpu"
MAX_MANIFEST_BYTES = 64 * 1024
MAX_MODEL_BYTES = 64 * 1024 * 1024
DATASET_FOLDERS = ("train/normal", "val/normal", "val/anomaly", "test/normal", "test/anomaly")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class CloudTrainingError(RuntimeError):
    """A cloud training job could not be started or completed."""


class RetryableCloudError(CloudTrainingError):
    """Connectivity or a cloud service interrupted an existing job check."""


def config_reason() -> str | None:
    """Describe a missing local prerequisite without contacting Modal."""
    if find_spec("modal") is None:
        return "Install the Modal SDK with requirements-cloud.txt to enable cloud training."

    profile_path = Path(os.environ.get("MODAL_CONFIG_PATH", "~/.modal.toml")).expanduser()
    has_tokens = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
    has_oauth = bool(os.environ.get("MODAL_OAUTH_REFRESH_TOKEN"))
    if not (has_tokens or has_oauth or profile_path.is_file()):
        return "Connect a Modal workspace with modal setup to enable cloud training."
    return None


def configured() -> bool:
    """Check local prerequisites. Submission authenticates the actual deployment."""
    return config_reason() is None


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise ValueError("Invalid cloud job identifier.")
    return value


def dataset_location(job_id: str) -> str:
    return f"jobs/{_identifier(job_id)}/dataset"


def result_location(job_id: str, call_id: str) -> str:
    return f"jobs/{_identifier(job_id)}/runs/{_identifier(call_id)}"


def _dataset_files(root: Path):
    for relative in DATASET_FOLDERS:
        folder = root / relative
        if not folder.is_dir():
            continue
        if not folder.resolve().is_relative_to(root):
            raise ValueError("Training folders must stay inside the staged dataset.")
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("Training images must stay inside the staged dataset.")
            yield path


def submit(dataset_dir: Path, job_id: str) -> str:
    """Upload validated staged images and return the durable Modal function call ID."""
    if reason := config_reason():
        raise CloudTrainingError(reason)

    import modal

    dataset_dir = dataset_dir.resolve(strict=True)
    remote_dir = dataset_location(job_id)
    files = list(_dataset_files(dataset_dir))
    if not files:
        raise ValueError("The staged dataset contains no supported images.")

    try:
        # Resolve both resources before uploading any files.
        function = modal.Function.from_name(APP_NAME, FUNCTION_NAME).hydrate()
        volume = modal.Volume.from_name(VOLUME_NAME).hydrate()
        with volume.batch_upload() as upload:
            for path in files:
                relative = path.relative_to(dataset_dir).as_posix()
                upload.put_file(path, f"/{remote_dir}/{relative}")
        return function.spawn(job_id).object_id
    except modal.exception.AuthError as error:
        raise CloudTrainingError("Modal authentication failed. Run modal setup and restart the app.") from error
    except modal.exception.NotFoundError as error:
        raise CloudTrainingError(
            "The Modal training worker is unavailable. Deploy deployment/modal_worker.py in the active workspace."
        ) from error
    except modal.exception.PermissionDeniedError as error:
        raise CloudTrainingError("The active Modal profile cannot use this training deployment.") from error
    except modal.exception.Error as error:
        raise CloudTrainingError(f"Could not submit the Modal job: {error}") from error


def _download_manifest(volume, remote_dir: str) -> dict:
    content = bytearray()
    for chunk in volume.read_file(f"{remote_dir}/result.json"):
        content.extend(chunk)
        if len(content) > MAX_MANIFEST_BYTES:
            raise CloudTrainingError("The cloud result metadata exceeds the supported size.")
    return json.loads(content)


def _download_model(volume, remote_dir: str, manifest: dict, model_path: Path) -> dict:
    if not isinstance(manifest, dict):
        raise CloudTrainingError("Modal returned invalid training metadata.")
    size = manifest.get("size_bytes")
    digest = manifest.get("sha256")
    metrics = manifest.get("metrics")
    if (
        type(size) is not int
        or not 0 < size <= MAX_MODEL_BYTES
        or not isinstance(digest, str)
        or not re.fullmatch(r"[a-f0-9]{64}", digest)
        or not isinstance(metrics, dict)
    ):
        raise CloudTrainingError("Modal returned invalid training metadata.")

    if model_path.is_file() and model_path.stat().st_size == size:
        with model_path.open("rb") as existing:
            if hashlib.file_digest(existing, "sha256").hexdigest() == digest:
                return metrics

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".modal-", dir=model_path.parent) as directory:
        temporary = Path(directory) / "model.npz"
        hashed = hashlib.sha256()
        downloaded = 0
        with temporary.open("wb") as output:
            for chunk in volume.read_file(f"{remote_dir}/model.npz"):
                downloaded += len(chunk)
                if downloaded > size:
                    raise CloudTrainingError("The model is larger than declared by the training worker.")
                output.write(chunk)
                hashed.update(chunk)
        if downloaded != size or hashed.hexdigest() != digest:
            raise CloudTrainingError("The downloaded model failed its integrity check.")
        temporary.replace(model_path)
    return metrics


def poll(job_id: str, call_id: str, model_path: Path) -> dict | None:
    """Return None while pending, or validate and save the completed artifact."""
    import modal

    remote_dir = result_location(job_id, call_id)
    volume = modal.Volume.from_name(VOLUME_NAME)
    try:
        try:
            manifest = modal.FunctionCall.from_id(call_id).get(timeout=0)
        except modal.exception.OutputExpiredError:
            # The durable volume outlives Modal's function output retention.
            manifest = _download_manifest(volume, remote_dir)
        except modal.exception.FunctionTimeoutError as error:
            raise CloudTrainingError("The Modal GPU training job timed out.") from error
        except (TimeoutError, modal.exception.TimeoutError) as error:
            if type(error) in (TimeoutError, modal.exception.TimeoutError):
                return None
            raise
        return _download_model(volume, remote_dir, manifest, model_path)
    except (modal.exception.ConnectionError, modal.exception.ServiceError, modal.exception.InternalError) as error:
        raise RetryableCloudError("Could not reach Modal. The job can be checked again.") from error
    except modal.exception.Error as error:
        raise CloudTrainingError(f"Modal GPU training failed: {error}") from error
