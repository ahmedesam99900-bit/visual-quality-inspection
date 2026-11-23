"""Exercise the complete local workspace over HTTP without PCB1 weights or Modal."""

from base64 import b64decode
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import time
import unittest
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw


APP_PATH = Path(__file__).resolve().parents[1] / "deployment" / "app.py"
STARTUP_LINE = re.compile(r"Visual Inspection is running at (http://127\.0\.0\.1:\d+)/")


def product_image(variation, defect=False):
    """Create distinct, consistently framed images of the same synthetic product."""
    image = Image.new("RGB", (192, 192), (32, 36, 39))
    painter = ImageDraw.Draw(image)
    painter.rectangle((25, 40, 167, 151), fill=(51, 125 + variation, 132))
    for center in (53, 137):
        painter.ellipse((center - 13, 65, center + 13, 91), fill=(190, 193, 196))
    for left in range(68, 124, 11):
        painter.rectangle((left, 151, left + 2, 171), fill=(228, 198, 127))
    if defect:
        painter.rectangle((78, 93, 111, 127), fill=(233, 24, 21 + variation))
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


class WorkspaceIntegration(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="inspection-integration-")
        self.addCleanup(temporary.cleanup)
        self.scratch = Path(temporary.name)
        self.project = self.scratch / "project"
        self.project.mkdir()
        self.process = None
        self.addCleanup(self.stop_server)
        self.start_server()

    def start_server(self):
        self.log = []
        environment = os.environ.copy()
        for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "MODAL_OAUTH_REFRESH_TOKEN"):
            environment.pop(key, None)
        environment["MODAL_CONFIG_PATH"] = str(self.scratch / "empty-modal.toml")
        self.process = subprocess.Popen(
            [sys.executable, "-B", "-u", str(APP_PATH), "--project-root", str(self.project),
             "--workspace", str(self.scratch / "workspace"), "--port", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.reader = Thread(target=self.log.extend, args=(self.process.stdout,), daemon=True)
        self.reader.start()
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline and self.process.poll() is None:
            log = "".join(self.log)
            match = STARTUP_LINE.search(log)
            if match:
                self.origin = match[1]
                return
            time.sleep(0.1)
        self.fail("Local app did not start\n" + "".join(self.log))

    def stop_server(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.reader.join(timeout=10)
        self.process.stdout.close()
        self.process = None

    def request(self, path, data=None, binary=False, expected=200):
        headers = {}
        if data is not None:
            headers["Origin"] = self.origin
            headers["Content-Type"] = "application/octet-stream" if binary else "application/json"
            if not binary:
                data = json.dumps(data).encode("utf-8")
        try:
            response = urlopen(Request(self.origin + path, data=data, headers=headers), timeout=15)
        except HTTPError as error:
            response = error
        with response:
            payload = json.load(response)
            self.assertEqual(response.status, expected, (path, payload))
            return payload

    def product(self, key):
        products = self.request("/api/workspace")["products"]
        return next(product for product in products if product["id"] == key)

    def upload(self, product, group, image, expected=200):
        query = urlencode({"group": group, "filename": "view.png"})
        return self.request(f"/api/products/{product}/images?{query}", image, binary=True, expected=expected)

    def wait_for_job(self, product, key):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            job = next(job for job in self.product(product)["jobs"] if job["id"] == key)
            if job["status"] not in ("queued", "running"):
                self.assertEqual(job["status"], "succeeded", job)
                return job["version_id"]
            time.sleep(0.2)
        self.fail("Training did not finish\n" + "".join(self.log))

    def test_product_training_inspection_and_restart(self):
        product = self.request("/api/products", {"name": "Synthetic test product"})["id"]
        endpoint = f"/api/products/{product}"
        inspection = f"/inspect?{urlencode({'product_id': product})}"
        self.request(f"{endpoint}/train", {"provider": "local"}, expected=400)
        self.upload(product, "train_normal", b"invalid image", expected=400)

        shared = product_image(0)
        self.assertTrue(self.upload(product, "train_normal", shared)["added"])
        self.assertFalse(self.upload(product, "train_normal", shared)["added"])
        duplicate = self.upload(product, "val_normal", shared, expected=400)
        self.assertIn("another group", duplicate["error"])
        for group, variants, defect in (
            ("train_normal", (1, 2, 3), False),
            ("val_normal", (4, 5), False),
            ("val_anomaly", (4, 5), True),
            ("test_normal", (6,), False),
            ("test_anomaly", (6,), True),
        ):
            for variant in variants:
                self.assertTrue(self.upload(product, group, product_image(variant, defect))["added"])

        job = self.request(f"{endpoint}/train", {"provider": "local"})
        version = self.wait_for_job(product, job["id"])
        state = self.product(product)
        self.assertIsNone(state["active_version"])
        self.assertEqual(state["dataset"], {
            "train_normal": 4, "val_normal": 2, "val_anomaly": 2, "test_normal": 1, "test_anomaly": 1,
        })
        metrics = state["versions"][0]["metrics"]
        self.assertEqual(metrics["training_images"], 4)
        self.assertEqual(metrics["test"]["accuracy"], 1.0)
        self.request(inspection, product_image(6), binary=True, expected=400)
        self.assertEqual(
            self.request(f"{endpoint}/activate", {"version_id": version})["active_version"], version,
        )

        for defect, status in ((False, "normal"), (True, "anomaly")):
            result = self.request(inspection, product_image(7, defect), binary=True)
            self.assertEqual(result["inspection_version"], version)
            self.assertEqual(result["status"], status)
            mime, encoded = result["images"]["heatmap"].split(",", 1)
            self.assertEqual(mime, "data:image/png;base64")
            with Image.open(BytesIO(b64decode(encoded, validate=True))) as heatmap:
                self.assertEqual(heatmap.size, (192, 192))
        failed = self.request(inspection, b"invalid image", binary=True)
        self.assertEqual(failed["status"], "cannot_inspect")
        self.assertEqual(failed["reason"], "invalid_image")

        self.stop_server()
        self.start_server()
        self.assertEqual(self.product(product)["active_version"], version)
        self.assertEqual(
            self.request(inspection, product_image(8, True), binary=True)["status"], "anomaly",
        )


if __name__ == "__main__":
    unittest.main()
