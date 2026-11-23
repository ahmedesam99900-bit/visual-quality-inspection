"""Train a spatial one-class patch model for consistently framed products."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F


FORMAT_VERSION = 1
IMAGE_SIZE = 192
PATCH_SIZE = 8
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
FEATURES = 14
REFERENCE_LIMIT = 48
BATCH_SIZE = 12
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
MIN_TRAIN_IMAGES = 3
# Minimum scale for near-constant features, in the normalized [0, 1] input range.
MIN_FEATURE_SCALE = 0.025
open_image = Image.open


def _paths(dataset: Path, split: str, label: str) -> list[Path]:
    folder = dataset / split / label
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ) if folder.is_dir() else []


def _load_rgb(path: Path) -> Image.Image:
    with open_image(BytesIO(path.read_bytes()), formats=("JPEG", "PNG")) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("Use a single-frame JPEG or PNG image.")
        return ImageOps.exif_transpose(source).convert("RGB")


def _device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The requested CUDA GPU is unavailable.")
    return device


def _embeddings(images: list[Image.Image], device: torch.device) -> torch.Tensor:
    resized = np.stack([
        np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR))
        for image in images
    ])
    pixels = torch.from_numpy(resized).to(device=device, dtype=torch.float32)
    pixels = pixels.permute(0, 3, 1, 2).div_(255)

    def pool(values: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(values, PATCH_SIZE)

    color = pool(pixels)
    detail = (pool(pixels.square()) - color.square()).clamp_min(0).sqrt()
    gray = pixels.mean(dim=1, keepdim=True)
    horizontal = F.pad(gray.diff(dim=3).abs(), (0, 1, 0, 0), mode="replicate")
    vertical = F.pad(gray.diff(dim=2).abs(), (0, 0, 0, 1), mode="replicate")
    edge_mean = pool(torch.cat((horizontal, vertical), dim=1))
    edge_strength = pool(torch.cat((horizontal.square(), vertical.square()), dim=1)).sqrt()
    context = F.avg_pool2d(color, kernel_size=3, stride=1, padding=1, count_include_pad=False)
    local_contrast = (color - context).abs()
    return torch.cat(
        (color, detail, edge_mean, edge_strength, context, local_contrast.mean(dim=1, keepdim=True)), dim=1,
    )


def _batches(paths: list[Path], device: torch.device):
    for start in range(0, len(paths), BATCH_SIZE):
        batch = paths[start:start + BATCH_SIZE]
        images = []
        for path in batch:
            try:
                images.append(_load_rgb(path))
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
                raise ValueError(f"Unreadable dataset image: {path.name}") from error
        yield start, _embeddings(images, device)


def _training_bank(paths, device, progress):
    # Select uniformly across the sorted dataset, while estimating feature scales
    # from every normal training image. Standardizing features balances color
    # and texture contributions to the nearest-neighbor distance.
    positions = set(np.linspace(0, len(paths) - 1, min(REFERENCE_LIMIT, len(paths)), dtype=int).tolist())
    bank = []
    counts = 0
    feature_sum = torch.zeros(FEATURES, device=device)
    feature_square_sum = torch.zeros_like(feature_sum)
    for start, features in _batches(paths, device):
        feature_sum += features.sum(dim=(0, 2, 3))
        feature_square_sum += features.square().sum(dim=(0, 2, 3))
        counts += features.size(0) * GRID_SIZE * GRID_SIZE
        bank.extend(feature for index, feature in enumerate(features) if start + index in positions)
        if start == 0 or start + features.size(0) == len(paths) or start % (5 * BATCH_SIZE) == 0:
            progress(f"Learning normal patches: {start + features.size(0)}/{len(paths)} images")
    variance = feature_square_sum / counts - (feature_sum / counts).square()
    scales = variance.clamp_min(MIN_FEATURE_SCALE ** 2).sqrt()
    return torch.stack(bank), scales


class _PatchDistance:
    """Compare each patch against normal training patches near the same position."""

    def __init__(self, bank: torch.Tensor, scales: torch.Tensor):
        self.scales = scales[None, :, None, None]
        bank = bank / self.scales
        padded = F.pad(bank, (1, 1, 1, 1), mode="replicate")
        neighborhoods = F.unfold(padded, kernel_size=3)
        references = neighborhoods.view(bank.size(0), FEATURES, 9, -1)
        self.references = references.permute(3, 0, 2, 1).reshape(
            GRID_SIZE * GRID_SIZE, bank.size(0) * 9, FEATURES,
        ).contiguous()

    def predict(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = features / self.scales
        positions = normalized.flatten(start_dim=2).permute(2, 0, 1).contiguous()
        distances = torch.cdist(positions, self.references) / np.sqrt(FEATURES)
        nearest = distances.amin(dim=2).T.reshape(-1, GRID_SIZE, GRID_SIZE)
        anomaly_maps = F.avg_pool2d(
            nearest[:, None], kernel_size=3, stride=1, padding=1, count_include_pad=False,
        )[:, 0]
        scores = anomaly_maps.flatten(start_dim=1).topk(3, dim=1).values.mean(dim=1)
        return scores, anomaly_maps


def _predict_split(paths, distance, device):
    scores = []
    maps = []
    for _, features in _batches(paths, device):
        batch_scores, batch_maps = distance.predict(features)
        scores.extend(batch_scores.cpu().tolist())
        maps.append(batch_maps.cpu().numpy())
    return np.asarray(scores, dtype=np.float64), np.concatenate(maps)


def _choose_threshold(normal, anomaly) -> float:
    values = np.unique(np.concatenate((normal, anomaly)))
    # Prefer the higher threshold for tied balanced accuracy.
    middle = values[:-1] + np.diff(values) / 2
    thresholds = np.concatenate(([np.nextafter(values[0], -np.inf)], middle, values[-1:]))
    true_negative_rate = np.searchsorted(np.sort(normal), thresholds, side="right") / len(normal)
    false_negative_rate = np.searchsorted(np.sort(anomaly), thresholds, side="right") / len(anomaly)
    objective = true_negative_rate - false_negative_rate
    best = np.flatnonzero(np.isclose(objective, objective.max(), atol=1e-12, rtol=0))[-1]
    return float(thresholds[best])


def _metrics(normal, anomaly, threshold) -> dict:
    true_negative = int(np.count_nonzero(normal <= threshold))
    true_positive = int(np.count_nonzero(anomaly > threshold))
    false_positive = len(normal) - true_negative
    false_negative = len(anomaly) - true_positive
    ranked = np.sort(normal)
    below = np.searchsorted(ranked, anomaly, side="left")
    at_or_below = np.searchsorted(ranked, anomaly, side="right")
    auroc = float(np.mean((below + at_or_below) / (2 * len(normal))))
    return {
        "normal": len(normal),
        "anomaly": len(anomaly),
        "accuracy": (true_negative + true_positive) / (len(normal) + len(anomaly)),
        "balanced_accuracy": (true_negative / len(normal) + true_positive / len(anomaly)) / 2,
        "auroc": auroc,
        "true_normal": true_negative,
        "true_anomaly": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def _check_duplicates(groups: dict[str, list[Path]]) -> None:
    seen = {}
    for group, paths in groups.items():
        for path in paths:
            digest = sha256(path.read_bytes()).digest()
            previous = seen.setdefault(digest, group)
            if previous != group:
                raise ValueError(f"Duplicate image across groups {previous} and {group}: {path.name}")


@torch.inference_mode()
def train_model(
    dataset_dir: Path,
    model_path: Path,
    device: str = "auto",
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Fit on normal images, calibrate on validation, then measure held-out test data."""
    if progress is None:
        def progress(_message: str) -> None:
            pass

    train = _paths(dataset_dir, "train", "normal")
    val_normal = _paths(dataset_dir, "val", "normal")
    val_anomaly = _paths(dataset_dir, "val", "anomaly")
    test_normal = _paths(dataset_dir, "test", "normal")
    test_anomaly = _paths(dataset_dir, "test", "anomaly")
    if len(train) < MIN_TRAIN_IMAGES or not val_normal or not val_anomaly:
        raise ValueError("Provide at least 3 train/normal images and images in both validation classes.")
    if bool(test_normal) != bool(test_anomaly):
        raise ValueError("An optional test split requires both test/normal and test/anomaly.")
    progress("Checking for duplicate images across data groups")
    _check_duplicates({
        "train/normal": train,
        "val/normal": val_normal,
        "val/anomaly": val_anomaly,
        "test/normal": test_normal,
        "test/anomaly": test_anomaly,
    })

    target = _device(device)
    bank, scales = _training_bank(train, target, progress)
    distance = _PatchDistance(bank, scales)
    progress("Calibrating image and pixel thresholds on validation data")
    normal_scores, normal_maps = _predict_split(val_normal, distance, target)
    anomaly_scores, _ = _predict_split(val_anomaly, distance, target)
    image_threshold = _choose_threshold(normal_scores, anomaly_scores)
    # With no pixel masks, estimate a pixel threshold from normal validation
    # patches and report image metrics separately.
    pixel_threshold = float(np.quantile(normal_maps, 0.995))
    pixel_scale = max(pixel_threshold, 0.01)
    image_scale = max(image_threshold, 0.01)
    test_metrics = None
    if test_normal:
        progress("Evaluating the held-out test split")
        normal_test, _ = _predict_split(test_normal, distance, target)
        anomaly_test, _ = _predict_split(test_anomaly, distance, target)
        test_metrics = _metrics(normal_test, anomaly_test, image_threshold)

    progress("Saving the trained model")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_name(f".{model_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as output:
            np.savez_compressed(
                output,
                version=FORMAT_VERSION,
                bank=bank.cpu().numpy(),
                scales=scales.cpu().numpy(),
                image_threshold=image_threshold,
                image_scale=image_scale,
                pixel_threshold=pixel_threshold,
                pixel_scale=pixel_scale,
            )
        os.replace(temporary, model_path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "algorithm": "spatial_patch_bank",
        "training_images": len(train),
        "reference_images": bank.size(0),
        "device": str(target),
        "image_threshold": image_threshold,
        "pixel_threshold": pixel_threshold,
        "pixel_threshold_source": "99.5th percentile of normal validation patches",
        "validation": _metrics(normal_scores, anomaly_scores, image_threshold),
        "test": test_metrics,
    }


class ProductInspector:
    """Apply a trained product model to an entire fixed-framing photo."""

    def __init__(self, model_path: Path, device: str = "cpu"):
        self.device = _device(device)
        with np.load(model_path, allow_pickle=False) as model:
            if model["version"].item() != FORMAT_VERSION:
                raise ValueError("Unsupported product model version.")
            bank = model["bank"]
            scales = model["scales"]
            if (bank.ndim != 4 or not 1 <= bank.shape[0] <= REFERENCE_LIMIT
                    or bank.shape[1:] != (FEATURES, GRID_SIZE, GRID_SIZE)
                    or scales.shape != (FEATURES,) or not np.isfinite(bank).all()
                    or not np.isfinite(scales).all() or (scales <= 0).any()):
                raise ValueError("Invalid product model features.")
            self.image_threshold = float(model["image_threshold"].item())
            self.pixel_threshold = float(model["pixel_threshold"].item())
            self.image_scale = float(model["image_scale"].item())
            self.pixel_scale = float(model["pixel_scale"].item())
            if (
                not np.isfinite((self.image_threshold, self.pixel_threshold, self.image_scale, self.pixel_scale)).all()
                or min(self.image_scale, self.pixel_scale) <= 0
            ):
                raise ValueError("Invalid product model thresholds.")
        self.distance = _PatchDistance(
            torch.as_tensor(bank, dtype=torch.float32, device=self.device),
            torch.as_tensor(scales, dtype=torch.float32, device=self.device),
        )

    @torch.inference_mode()
    def inspect(self, source: Path):
        from inspect_pcb1 import Inspection

        try:
            image = _load_rgb(source)
        except FileNotFoundError:
            return Inspection("cannot_inspect", reason="missing_image")
        except ValueError:
            return Inspection("cannot_inspect", reason="multiple_frames")
        except (OSError, SyntaxError, Image.DecompressionBombError):
            return Inspection("cannot_inspect", reason="invalid_image")
        scores, maps = self.distance.predict(_embeddings([image], self.device))
        score = float(scores[0].item())
        anomaly_map = maps[0].cpu().numpy()
        normalized_map = np.clip(0.5 + (anomaly_map - self.pixel_threshold) / (2 * self.pixel_scale), 0, 1)
        image_score = float(np.clip(0.5 + (score - self.image_threshold) / (2 * self.image_scale), 0, 1))
        box = (0, 0, image.width, image.height)
        crop = BytesIO()
        image.save(crop, format="JPEG", quality=95)
        return Inspection(
            status="anomaly" if score > self.image_threshold else "normal",
            original=image,
            detection={"box_xyxy": box, "crop_box_xyxy": box, "confidence": None},
            anomaly_score=image_score,
            image_threshold=0.5,
            pixel_threshold=0.5,
            crop_jpeg=crop.getvalue(),
            anomaly_map=Image.fromarray(np.rint(normalized_map * 255).astype(np.uint8)),
            mask=Image.fromarray((anomaly_map > self.pixel_threshold).astype(np.uint8) * 255),
        )
