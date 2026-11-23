"""Run a PCB1 v1 inspection on one JPEG or PNG image."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
import math
import os
from pathlib import Path
import sys
from typing import Literal
from uuid import uuid4

from PIL import Image, ImageDraw, ImageOps


VERSION = "pcb1/v1"
YOLO_WEIGHTS = Path("outputs/train_runs/pcb1_yolo_cpu/weights/best.pt")
PATCHCORE_WEIGHTS = Path(
    "outputs/patchcore_runs/pcb1_patchcore_cpu/Patchcore/pcb1_patchcore_dev/v1/weights/lightning/model.ckpt"
)
# Keep Pillow's original reader before Ultralytics adds its optional HEIF fallback.
open_image = Image.open

MESSAGES = {
    "normal": "سليمة حسب نتيجة الفحص",
    "anomaly": "فيها عيب محتمل حسب نتيجة الفحص",
    "missing_image": "ملف الصورة مش موجود.",
    "invalid_image": "تعذّر قراءة الصورة. استخدم صورة سليمة بصيغة JPEG أو PNG.",
    "multiple_frames": "الملف فيه أكتر من إطار. اختار صورة واحدة ثابتة.",
    "no_product_detected": "الموديل مقدرش يحدد مكان وحدة الـPCB في الصورة.",
    "invalid_detection": "الموديل حدّد مكان للوحدة، لكن حدود القص غير صالحة.",
}


@dataclass
class Inspection:
    status: Literal["normal", "anomaly", "cannot_inspect"]
    reason: str | None = None
    original: Image.Image | None = None
    detection: dict | None = None
    anomaly_score: float | None = None
    image_threshold: float | None = None
    pixel_threshold: float | None = None
    crop_jpeg: bytes | None = field(default=None, repr=False)
    anomaly_map: Image.Image | None = None
    mask: Image.Image | None = None

    def save(self, source: Path, output: Path) -> None:
        """Publish the complete result in a new directory."""
        if output.exists():
            raise FileExistsError(f"Output already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        # A regular directory inherits permissions so the operator can open the result.
        stage = output.with_name(f".pcb1-{uuid4().hex}")
        stage.mkdir()
        try:
            files = self._write_images(stage)
            result = {
                "inspection_version": VERSION,
                "source_image": str(source),
                "status": self.status,
                "reason": self.reason,
                "message_ar": MESSAGES[self.reason or self.status],
                "detection": self.detection,
                "anomaly_score": self.anomaly_score,
                "image_threshold": self.image_threshold,
                "pixel_threshold": self.pixel_threshold,
                "mask_has_regions": bool(self.mask.getbbox()) if self.mask is not None else None,
                "files": files,
            }
            (stage / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            stage.rename(output)
        finally:
            if stage.exists():
                for file in stage.iterdir():
                    file.unlink()
                stage.rmdir()

    def _write_images(self, output: Path) -> dict[str, str]:
        if self.crop_jpeg is None:
            return {}

        (output / "crop.jpg").write_bytes(self.crop_jpeg)
        with open_image(BytesIO(self.crop_jpeg)) as saved_crop:
            crop = saved_crop.convert("RGB")

        detected = self.original.copy()
        draw = ImageDraw.Draw(detected)
        left, top, right, bottom = self.detection["crop_box_xyxy"]
        draw.rectangle(
            (left, top, right - 1, bottom - 1),
            outline="#1de2f1",
            width=max(2, min(detected.size) // 200),
        )
        detected.save(output / "detected.png")

        heat = self.anomaly_map.resize(crop.size, Image.Resampling.BILINEAR)
        # Fixed score scale across images. The legend uses the same five stops.
        import numpy as np

        anchors = np.array([[34, 49, 115], [29, 147, 207], [61, 211, 167],
                            [251, 212, 81], [222, 61, 54]])
        stops = np.linspace(0, 255, len(anchors))
        levels = np.arange(256)
        palette = np.stack([np.interp(levels, stops, channel) for channel in anchors.T], axis=1)
        color_map = Image.fromarray(palette.round().astype(np.uint8)[np.asarray(heat)])
        Image.blend(crop, color_map, 0.68).save(output / "heatmap.png")

        self.mask.resize(crop.size, Image.Resampling.NEAREST).save(output / "mask.png")
        return {
            "detected": "detected.png",
            "crop": "crop.jpg",
            "heatmap": "heatmap.png",
            "mask": "mask.png",
        }


class Pcb1Inspector:
    """Reuse the detector and anomaly model across inspections in one process."""

    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()

    @cached_property
    def preprocessing(self):
        path = self.project_root / "src/prepare_patchcore_data.py"
        if not path.is_file():
            raise FileNotFoundError(f"PCB1 preprocessing script missing: {path}")
        spec = spec_from_file_location("pcb1_preprocessing", path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        if (module.IMGSZ, module.CONFIDENCE, module.CROP_MARGIN) != (320, 0.25, 0.01):
            raise ValueError("The project crop settings differ from the settings used by PCB1 v1.")
        return module

    @cached_property
    def detector(self):
        from ultralytics import YOLO

        return YOLO(str(self._weights(YOLO_WEIGHTS)))

    @cached_property
    def patchcore(self):
        from anomalib.models import Patchcore

        return Patchcore.load_from_checkpoint(
            self._weights(PATCHCORE_WEIGHTS),
            map_location="cpu",
            weights_only=True,
            pre_trained=False,
        ).eval()

    def _weights(self, relative_path: Path) -> Path:
        path = self.project_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"PCB1 model file missing: {path}")
        return path

    def inspect(self, source: Path) -> Inspection:
        """Inspect a local image without writing intermediate images."""
        try:
            encoded = source.read_bytes()
            with open_image(BytesIO(encoded), formats=("JPEG", "PNG")) as image:
                if getattr(image, "n_frames", 1) != 1:
                    return Inspection("cannot_inspect", reason="multiple_frames")
                orientation = image.getexif().get(274, 1)
                original = ImageOps.exif_transpose(image).convert("RGB")
        except FileNotFoundError:
            return Inspection("cannot_inspect", reason="missing_image")
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
            return Inspection("cannot_inspect", reason="invalid_image")

        import cv2
        import numpy as np

        # Match the project's YOLO decoder for regular PCB1 images. For
        # rotated phone photos, use the same upright pixels for both stages.
        detection_image = None
        if orientation == 1:
            detection_image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if detection_image is None or detection_image.shape[:2] != (original.height, original.width):
            detection_image = np.ascontiguousarray(np.asarray(original)[:, :, ::-1])

        preprocessing = self.preprocessing
        result = self.detector.predict(
            source=detection_image,
            imgsz=preprocessing.IMGSZ,
            conf=preprocessing.CONFIDENCE,
            device="cpu",
            max_det=10,
            batch=1,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return Inspection("cannot_inspect", reason="no_product_detected")

        selected = int(result.boxes.conf.argmax())
        box = tuple(result.boxes.xyxy[selected].tolist())
        confidence = float(result.boxes.conf[selected])
        if not all(math.isfinite(value) for value in (*box, confidence)):
            return Inspection("cannot_inspect", reason="invalid_detection")
        try:
            bounds = preprocessing.crop_bounds(box, *original.size)
        except ValueError:
            return Inspection("cannot_inspect", reason="invalid_detection")

        buffer = BytesIO()
        original.crop(bounds).save(buffer, format="JPEG", quality=95)
        crop_jpeg = buffer.getvalue()
        return self._score_crop(
            original,
            crop_jpeg,
            {"box_xyxy": box, "crop_box_xyxy": bounds, "confidence": confidence},
        )

    def _score_crop(self, original: Image.Image, crop_jpeg: bytes, detection: dict) -> Inspection:
        import torch
        from torchvision.transforms.v2.functional import to_dtype, to_image

        # Decode the JPEG crop exactly as the prepared validation data was read.
        with open_image(BytesIO(crop_jpeg)) as crop:
            image = to_dtype(to_image(crop.convert("RGB")), torch.float32, scale=True)

        model = self.patchcore
        with torch.inference_mode():
            # Match Engine.test: the export-oriented forward disables resize antialiasing.
            resized = model.pre_processor.transform(image.unsqueeze(0))
            prediction = model.post_processor(model.model(resized))
        anomaly_score = float(prediction.pred_score.item())
        image_threshold = float(model.post_processor.normalized_image_threshold)
        pixel_threshold = float(model.post_processor.normalized_pixel_threshold)
        anomaly_map = prediction.anomaly_map[0, 0].cpu()
        if not (
            all(math.isfinite(value) for value in (anomaly_score, image_threshold, pixel_threshold))
            and torch.isfinite(anomaly_map).all().item()
        ):
            raise RuntimeError("PatchCore returned a non-finite score, threshold, or map.")

        mask = prediction.pred_mask[0, 0].to(torch.uint8).cpu() * 255
        return Inspection(
            status="anomaly" if prediction.pred_label.item() else "normal",
            original=original,
            detection=detection,
            anomaly_score=anomaly_score,
            image_threshold=image_threshold,
            pixel_threshold=pixel_threshold,
            crop_jpeg=crop_jpeg,
            anomaly_map=Image.fromarray(anomaly_map.clamp(0, 1).mul(255).round().byte().numpy()),
            mask=Image.fromarray(mask.numpy()),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Photo to inspect (JPEG or PNG)")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project with the existing PCB1 models")
    parser.add_argument("--output", type=Path, help="New folder for this inspection result")
    args = parser.parse_args()

    source = args.image.resolve()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = (args.output or Path.cwd() / "outputs/inspections" / timestamp).resolve()
    if output.exists():
        print(f"مجلد النتيجة موجود بالفعل. اختار مجلد جديد.\n{output}", file=sys.stderr)
        return 1

    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    try:
        result = Pcb1Inspector(args.project_root).inspect(source)
        result.save(source, output)
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        print(f"حصلت مشكلة أثناء التشغيل. تفاصيلها\n{error}", file=sys.stderr)
        return 1

    if result.status == "cannot_inspect":
        print(f"تعذّر الفحص. {MESSAGES[result.reason]}")
    else:
        print(MESSAGES[result.status])
        print(f"درجة الشذوذ {result.anomaly_score:.3f}")
        if result.status == "anomaly" and not result.mask.getbbox():
            print("الموديل لاحظ اختلاف في الصورة، لكن القناع ما حدّدش منطقة واضحة.")
    print(f"النتيجة محفوظة في\n{output / 'result.json'}")
    return 2 if result.status == "cannot_inspect" else 0


if __name__ == "__main__":
    raise SystemExit(main())
