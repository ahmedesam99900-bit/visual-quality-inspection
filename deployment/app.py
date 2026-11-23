"""Local browser interface for the Visual Inspection workspace."""

from __future__ import annotations

import argparse
from base64 import b64encode
from functools import lru_cache, partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from urllib.parse import parse_qs, urlsplit
import webbrowser

from workspace import STARTER_ID, Workspace

from inspect_pcb1 import PATCHCORE_WEIGHTS, YOLO_WEIGHTS, Pcb1Inspector, VERSION


APP_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT = APP_DIR.parent
SAMPLE = Path("data/visa/pcb1/Data/Images/Anomaly/041.JPG")
ASSETS = {
    "/board.jpg": ("board.jpg", "image/jpeg"),
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MAX_BYTES = 20 * 1024 * 1024
MUTATIONS = {"images", "train", "activate"}


class RequestError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


@lru_cache(maxsize=3)
def custom_inspector(model_path):
    from trainer import ProductInspector

    return ProductInspector(model_path)

IMAGE_TYPES = {".jpg": "image/jpeg", ".png": "image/png"}


class InspectionHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, inspector, workspace, lock, **kwargs):
        self.inspector = inspector
        self.lock = lock
        self.workspace = workspace
        super().__init__(*args, **kwargs)

    @property
    def local_host(self):
        return f"127.0.0.1:{self.server.server_port}"

    def send_content(self, status, content, kind):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, status, value):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_content(status, payload, "application/json; charset=utf-8")

    def check_host(self):
        if self.headers.get("Host") == self.local_host:
            return True
        self.send_json(HTTPStatus.FORBIDDEN, {"error": "Open the local application address."})
        return False

    def do_GET(self):
        if not self.check_host():
            return

        request = urlsplit(self.path)
        path = request.path
        if path == "/api/workspace":
            self.send_json(HTTPStatus.OK, self.workspace.describe())
            return

        if path == "/sample.jpg":
            product = parse_qs(request.query).get("product_id", ["pcb1"])[0]
            if product == STARTER_ID and self.workspace.starter is not None:
                path = self.workspace.starter / "sample.jpg"
            elif product == "pcb1":
                path = self.inspector.project_root / SAMPLE
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "This product has no sample image."})
                return
            kind = "image/jpeg"
        elif path in ASSETS:
            asset, kind = ASSETS[path]
            path = APP_DIR / "web" / asset
        else:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Page not found."})
            return

        try:
            content = path.read_bytes()
        except OSError:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "The requested file is unavailable."})
            return
        self.send_content(HTTPStatus.OK, content, kind)

    def read_payload(self, media_type, limit):
        if self.headers.get_content_type() != media_type:
            raise RequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Unsupported request format.")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= limit:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "The request is empty or too large.")
        self.connection.settimeout(15)
        try:
            payload = self.rfile.read(length)
        except (TimeoutError, ConnectionError) as error:
            raise RequestError(HTTPStatus.REQUEST_TIMEOUT, "The upload timed out. Try again.") from error
        if len(payload) != length:
            raise RequestError(HTTPStatus.BAD_REQUEST, "Upload incomplete. Try again.")
        return payload

    def read_fields(self):
        try:
            fields = json.loads(self.read_payload("application/json", 16 * 1024))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ValueError("Invalid request data.") from error
        if not isinstance(fields, dict):
            raise ValueError("Invalid request data.")
        return fields

    def do_POST(self):
        if not self.check_host():
            return
        if self.headers.get("Origin") != f"http://{self.local_host}":
            self.send_json(HTTPStatus.FORBIDDEN, {"error": "Submit requests from the application."})
            return

        try:
            path = urlsplit(self.path)
            query = {key: values[0] for key, values in parse_qs(path.query).items()}
            if path.path == "/inspect":
                data = self.inspect_upload(
                    self.read_payload("application/octet-stream", MAX_BYTES),
                    query.get("product_id", "pcb1"),
                    query.get("version_id"),
                )
            elif path.path == "/api/products":
                data = self.workspace.create(self.read_fields().get("name"))
            else:
                data = self.product_action(path.path, query)
        except RequestError as error:
            self.send_json(error.status, {"error": str(error)})
        except KeyError as error:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": error.args[0]})
        except ValueError as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except Exception as error:
            self.log_error("Request failed: %s", error)
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Request failed. See the application window for details."})
        else:
            self.send_json(HTTPStatus.OK, data)

    def product_action(self, path, query):
        parts = path.strip("/").split("/")
        if len(parts) != 4 or parts[:2] != ["api", "products"] or parts[3] not in MUTATIONS:
            raise KeyError("Page not found.")
        _, _, product_id, action = parts
        if action == "images":
            payload = self.read_payload("application/octet-stream", MAX_BYTES)
            return self.workspace.upload(product_id, query.get("group"), query.get("filename", "image"), payload)

        fields = self.read_fields()
        if action == "train":
            return self.workspace.queue(product_id, fields.get("provider"))
        version_id = fields.get("version_id")
        if not isinstance(version_id, str):
            raise ValueError("Choose a model version.")
        return self.workspace.activate(product_id, version_id)

    def inspect_upload(self, payload, product_id, version_id):
        if product_id == "pcb1":
            if not self.workspace.pcb_available:
                raise ValueError("The PCB1 model weights are missing from this project.")
            if version_id not in (None, VERSION):
                raise KeyError("Unknown PCB1 model version.")
            model_path = None
            version_id = VERSION
        else:
            version_id, model_path = self.workspace.model(product_id, version_id)

        with TemporaryDirectory(prefix="pcb1-upload-") as directory:
            upload = Path(directory) / "image"
            upload.write_bytes(payload)
            # Keep one copy of the models loaded and run one inspection at a time.
            with self.lock:
                model = self.inspector if model_path is None else custom_inspector(model_path)
                inspection = model.inspect(upload)
            output = upload.with_name("result")
            inspection.save(upload, output)
            result = json.loads((output / "result.json").read_text(encoding="utf-8"))
            result.pop("source_image")
            result["inspection_version"] = version_id
            images = {}
            for name, filename in result.pop("files").items():
                path = output / filename
                encoded = b64encode(path.read_bytes()).decode("ascii")
                images[name] = f"data:{IMAGE_TYPES[path.suffix]};base64,{encoded}"
            result["images"] = images
            return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the page in the default browser.")
    parser.add_argument("--workspace", type=Path, help="Directory for products, datasets and training jobs.")
    parser.add_argument("--starter", type=Path, help="Install the included sample product into a new workspace.")
    args = parser.parse_args()
    root = args.project_root.resolve()
    pcb_available = all((root / path).is_file() for path in (
        YOLO_WEIGHTS, PATCHCORE_WEIGHTS, Path("src/prepare_patchcore_data.py")
    ))
    data_root = (args.workspace or root / "outputs" / "inspection_workspace").resolve()
    workspace_cache = data_root / "runtime"
    for variable, folder in (("YOLO_CONFIG_DIR", "yolo"), ("MPLCONFIGDIR", "matplotlib")):
        if variable not in os.environ:
            path = workspace_cache / folder
            path.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(path)
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")

    workspace = Workspace(data_root, pcb_available, args.starter)
    handler = partial(InspectionHandler, inspector=Pcb1Inspector(root), workspace=workspace, lock=Lock())
    try:
        with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
            workspace.start()
            url = f"http://127.0.0.1:{server.server_port}/"
            print(f"Visual Inspection is running at {url}", flush=True)
            if args.open:
                webbrowser.open(url)
            server.serve_forever()
    except KeyboardInterrupt:
        print("\nVisual Inspection stopped.")
    except OSError as error:
        parser.exit(1, f"Could not start Visual Inspection: {error}\n")


if __name__ == "__main__":
    main()
