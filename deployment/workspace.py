"""Persist image datasets, training jobs, and model versions for the local app."""

from contextlib import contextmanager, closing
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import shutil
import sqlite3
from threading import Event, Thread
from uuid import uuid4

from inspect_pcb1 import VERSION, open_image
from PIL import Image

GROUPS = ("train_normal", "val_normal", "val_anomaly", "test_normal", "test_anomaly")
STARTER_ID = "starter-pcb1"
STARTER_VERSION = "starter-pcb1-v1"
BUILTIN = {
    "id": "pcb1",
    "name": "HC-SR04 / PCB1",
    "builtin": True,
    "active_version": VERSION,
    "versions": [{"id": VERSION, "name": VERSION, "metrics": None}],
    "dataset": dict.fromkeys(GROUPS, 0),
    "jobs": [],
}


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Workspace:
    def __init__(self, root, pcb_available, starter=None):
        self.root = root
        self.pcb_available = pcb_available
        self.starter = starter
        self.root.mkdir(parents=True, exist_ok=True)
        self._wake = Event()
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS products (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    active_version TEXT, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS images (
                    product_id TEXT NOT NULL REFERENCES products(id),
                    digest TEXT NOT NULL, group_name TEXT NOT NULL,
                    suffix TEXT NOT NULL, filename TEXT NOT NULL, body BLOB NOT NULL,
                    PRIMARY KEY (product_id, digest)
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(id),
                    status TEXT NOT NULL, provider TEXT NOT NULL, message TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, call_id TEXT, version_id TEXT
                );
                CREATE TABLE IF NOT EXISTS job_images (
                    job_id TEXT NOT NULL REFERENCES jobs(id), digest TEXT NOT NULL,
                    PRIMARY KEY (job_id, digest)
                );
                CREATE TABLE IF NOT EXISTS versions (
                    id TEXT PRIMARY KEY, product_id TEXT NOT NULL REFERENCES products(id),
                    name TEXT NOT NULL, created_at TEXT NOT NULL, metrics TEXT NOT NULL
                );
            """)
        if self.starter is not None:
            self._install_starter()

    def _install_starter(self):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM products WHERE id = ?", (STARTER_ID,)).fetchone():
                return
            metrics = json.loads((self.starter / "metrics.json").read_text(encoding="utf-8"))
            folder = self.root / "models"
            folder.mkdir(exist_ok=True)
            model = folder / (STARTER_VERSION + ".npz")
            pending = folder / (STARTER_VERSION + ".tmp")
            shutil.copyfile(self.starter / "model.npz", pending)
            pending.replace(model)
            now = timestamp()
            db.execute("INSERT INTO products VALUES (?, ?, ?, ?)", (STARTER_ID, "PCB1 · cropped starter", STARTER_VERSION, now))
            db.execute(
                "INSERT INTO versions VALUES (?, ?, ?, ?, ?)",
                (STARTER_VERSION, STARTER_ID, "Starter v1", now, json.dumps(metrics, allow_nan=False)),
            )

    @contextmanager
    def db(self):
        with closing(sqlite3.connect(self.root / "workspace.sqlite3", timeout=10)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys = ON")
            with db:
                yield db

    def start(self):
        with self.db() as db:
            db.execute(
                "UPDATE jobs SET status = 'failed', message = ?, updated_at = ? "
                "WHERE status = 'running' AND (provider = 'local' OR call_id IS NULL)",
                ("Training was interrupted. Start a new job to retry.", timestamp()),
            )
        Thread(target=self._run, name="training-worker", daemon=True).start()

    @staticmethod
    def _product(db, product_id):
        product = db.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
        if product is None:
            raise KeyError("Product not found.")
        return product

    @staticmethod
    def cloud_status():
        try:
            import cloud
            reason = cloud.config_reason()
        except (ImportError, OSError, RuntimeError):
            reason = "Connect a Modal account and deploy the training worker first."
        return {
            "available": reason is None,
            "reason": reason,
        }

    def describe(self):
        builtin = deepcopy(BUILTIN)
        builtin["available"] = self.pcb_available
        builtin["sample_available"] = self.pcb_available
        if not self.pcb_available:
            builtin["active_version"] = None
            builtin["versions"] = []
        products = [builtin]
        with self.db() as db:
            for row in db.execute("SELECT * FROM products ORDER BY created_at, name"):
                product = dict(row)
                key = (row["id"],)
                product["builtin"] = False
                product["available"] = True
                product["sample_available"] = product["id"] == STARTER_ID and self.starter is not None
                product["dataset"] = dict.fromkeys(GROUPS, 0)
                product["dataset"].update(dict(db.execute(
                    "SELECT group_name, COUNT(*) FROM images WHERE product_id = ? GROUP BY group_name", key
                )))
                product["versions"] = []
                for version in db.execute(
                    "SELECT id, name, created_at, metrics FROM versions WHERE product_id = ? ORDER BY rowid DESC", key
                ):
                    product["versions"].append({**dict(version), "metrics": json.loads(version["metrics"])})
                product["jobs"] = [dict(job) for job in db.execute(
                    "SELECT id, provider, status, message, version_id, created_at, updated_at "
                    "FROM jobs WHERE product_id = ? ORDER BY rowid DESC LIMIT 30", key
                )]
                products.append(product)
        return {"products": products, "cloud": self.cloud_status()}

    def create(self, name):
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 80:
            raise ValueError("Enter a product name between 1 and 80 characters.")
        product_id = uuid4().hex
        try:
            with self.db() as db:
                db.execute("INSERT INTO products VALUES (?, ?, NULL, ?)", (product_id, name.strip(), timestamp()))
        except sqlite3.IntegrityError as error:
            raise ValueError("A product with that name already exists.") from error
        return {"id": product_id}

    def upload(self, product_id, group, filename, payload):
        if group not in GROUPS:
            raise ValueError("Choose a training, validation, or test group.")
        try:
            with open_image(BytesIO(payload), formats=("JPEG", "PNG")) as image:
                if getattr(image, "n_frames", 1) != 1 or image.width * image.height > 24_000_000:
                    raise ValueError("Choose a single-frame image up to 24 megapixels.")
                suffix = ".jpg" if image.format == "JPEG" else ".png"
                image.load()
        except (OSError, SyntaxError, Image.DecompressionBombError) as error:
            raise ValueError("Choose a valid JPEG or PNG image.") from error
        digest = sha256(payload).hexdigest()
        with self.db() as db:
            self._product(db, product_id)
            changed = db.execute(
                "INSERT OR IGNORE INTO images VALUES (?, ?, ?, ?, ?, ?)",
                (product_id, digest, group, suffix, Path(filename).name[:200], payload),
            ).rowcount
            if not changed:
                previous = db.execute(
                    "SELECT group_name FROM images WHERE product_id = ? AND digest = ?", (product_id, digest)
                ).fetchone()["group_name"]
                if previous != group:
                    raise ValueError("This exact image is already in another group.")
        return {"added": bool(changed), "group": group}

    def queue(self, product_id, provider):
        if provider not in ("local", "modal"):
            raise ValueError("Choose a training provider.")
        if provider == "modal" and not self.cloud_status()["available"]:
            raise ValueError("Modal is unavailable. Connect it before starting cloud training.")
        key = uuid4().hex
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            self._product(db, product_id)
            if db.execute(
                "SELECT 1 FROM jobs WHERE product_id = ? AND status IN ('queued', 'running')", (product_id,)
            ).fetchone():
                raise ValueError("This product already has a training job in progress.")
            counts = dict(db.execute("SELECT group_name, COUNT(*) FROM images WHERE product_id = ? GROUP BY group_name", (product_id,)))
            if any(counts.get(group, 0) < size for group, size in zip(GROUPS, (3, 1, 1))):
                raise ValueError("Add at least 3 normal training images, 1 normal validation image, and 1 defect validation image.")
            if bool(counts.get("test_normal")) != bool(counts.get("test_anomaly")):
                raise ValueError("Include both normal and defect images if you use the test split.")
            now = timestamp()
            db.execute(
                "INSERT INTO jobs VALUES (?, ?, 'queued', ?, 'Waiting to start.', ?, ?, NULL, NULL)",
                (key, product_id, provider, now, now),
            )
            db.execute("INSERT INTO job_images SELECT ?, digest FROM images WHERE product_id = ?", (key, product_id))
        self._wake.set()
        return {"id": key, "status": "queued"}

    def activate(self, product_id, version_id):
        with self.db() as db:
            self._product(db, product_id)
            if not db.execute(
                "SELECT 1 FROM versions WHERE id = ? AND product_id = ?", (version_id, product_id)
            ).fetchone():
                raise KeyError("Version not found for this product.")
            db.execute("UPDATE products SET active_version = ? WHERE id = ?", (version_id, product_id))
        return {"active_version": version_id}

    def model(self, product_id, version_id=None):
        with self.db() as db:
            product = self._product(db, product_id)
            version_id = version_id or product["active_version"]
            if not version_id:
                raise ValueError("Activate or select a trained model before inspecting.")
            if db.execute(
                "SELECT 1 FROM versions WHERE id = ? AND product_id = ?", (version_id, product_id)
            ).fetchone() is None:
                raise KeyError("Version not found for this product.")
        return version_id, self.root / "models" / (version_id + ".npz")

    def _update(self, key, message, status="running", call_id=None):
        with self.db() as db:
            db.execute(
                "UPDATE jobs SET message = ?, status = ?, updated_at = ?, call_id = COALESCE(?, call_id) WHERE id = ?",
                (message, status, timestamp(), call_id, key),
            )

    def _publish(self, job, model_path, metrics):
        if not model_path.is_file():
            raise RuntimeError("Training completed without a model artifact.")
        key = job["id"]
        with self.db() as db:
            count = db.execute("SELECT COUNT(*) FROM versions WHERE product_id = ?", (job["product_id"],)).fetchone()[0]
            db.execute(
                "INSERT INTO versions VALUES (?, ?, ?, ?, ?)",
                (key, job["product_id"], f"v{count + 1}", timestamp(), json.dumps(metrics, allow_nan=False)),
            )
            db.execute(
                "UPDATE jobs SET status = 'succeeded', message = ?, version_id = ?, updated_at = ? WHERE id = ?",
                ("Model ready for review and activation.", key, timestamp(), key),
            )

    def _process(self, job):
        from tempfile import TemporaryDirectory

        key = job["id"]
        model_path = self.root / "models" / (key + ".npz")
        model_path.parent.mkdir(exist_ok=True)
        try:
            if job["call_id"]:
                import cloud

                metrics = cloud.poll(key, job["call_id"], model_path)
                if metrics is not None:
                    self._publish(job, model_path, metrics)
                return
            with TemporaryDirectory(prefix="dataset-", dir=self.root) as temporary:
                dataset = Path(temporary)
                with self.db() as db:
                    rows = db.execute(
                        "SELECT i.* FROM images i JOIN job_images j ON i.digest = j.digest "
                        "WHERE j.job_id = ? AND i.product_id = ?", (key, job["product_id"])
                    )
                    for row in rows:
                        split, label = row["group_name"].split("_", 1)
                        folder = dataset / split / label
                        folder.mkdir(parents=True, exist_ok=True)
                        (folder / (row["digest"] + row["suffix"])).write_bytes(row["body"])
                if job["provider"] == "modal":
                    import cloud

                    self._update(key, "Uploading dataset to your private Modal workspace.")
                    call_id = cloud.submit(dataset, key)
                    self._update(key, "Submitted to Modal GPU. Waiting for the model.", call_id=call_id)
                else:
                    from trainer import train_model

                    result = train_model(dataset, model_path, device="cpu", progress=lambda message: self._update(key, message))
                    self._publish(job, model_path, result)
        except Exception as error:
            retryable = False
            if job["call_id"]:
                import cloud
                retryable = isinstance(error, getattr(cloud, "RetryableCloudError", ()))
            if retryable:
                self._update(key, "Modal is temporarily unreachable. Retrying.")
            else:
                self._update(key, str(error)[:300] or type(error).__name__, status="failed")

    def _run(self):
        while True:
            self._wake.clear()
            with self.db() as db:
                db.execute("BEGIN IMMEDIATE")
                pending = list(db.execute("SELECT * FROM jobs WHERE status = 'running' AND call_id IS NOT NULL"))
                queued = db.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY rowid LIMIT 1").fetchone()
                if queued is not None:
                    db.execute(
                        "UPDATE jobs SET status = 'running', message = 'Preparing dataset.', updated_at = ? WHERE id = ?",
                        (timestamp(), queued["id"]),
                    )
            for job in pending:
                self._process(job)
            if queued is not None:
                self._process(queued)
            self._wake.wait(1 if queued is not None else 5)
