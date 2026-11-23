"""Build the one-file Windows x64 installer from the project and its cached release inputs."""

import argparse
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import tempfile
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from PIL import Image, ImageDraw


SOURCE = Path(__file__).resolve().parent
PROJECT = SOURCE.parents[1]
OUTPUT = PROJECT / "outputs/windows-desktop"
SDK_FILES = (
    ("lib/net462/Microsoft.Web.WebView2.Core.dll", "Microsoft.Web.WebView2.Core.dll"),
    ("lib/net462/Microsoft.Web.WebView2.WinForms.dll", "Microsoft.Web.WebView2.WinForms.dll"),
    ("runtimes/win-x64/native/WebView2Loader.dll", "WebView2Loader.dll"),
    ("LICENSE.txt", "licenses/WebView2-SDK-LICENSE.txt"),
    ("NOTICE.txt", "licenses/WebView2-SDK-NOTICE.txt"),
)


def draw_icon(path):
    icon = Image.new("RGBA", (256, 256))
    draw = ImageDraw.Draw(icon)
    draw.rounded_rectangle((6, 6, 250, 250), radius=51, fill="#14283e")
    draw.rounded_rectangle((49, 48, 208, 190), radius=21, outline="#70dcf2", width=13)
    draw.ellipse((68, 83, 113, 128), outline="#ffffff", width=10)
    draw.ellipse((143, 83, 188, 128), outline="#ffffff", width=10)
    draw.line((76, 155, 180, 155), fill="#70dcf2", width=9)
    for x in (91, 116, 141, 166):
        draw.line((x, 191, x, 217), fill="#70dcf2", width=9)
    icon.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (128, 128), (256, 256)])


def compile_windows(csc, source, output, icon, references):
    subprocess.run(
        [str(csc), "/nologo", "/optimize+", "/target:winexe", "/platform:x64",
         "/out:" + str(output), "/win32icon:" + str(icon),
         *["/reference:" + str(reference) for reference in references], str(source)],
        check=True,
    )


def build(cache, release):
    csc = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
    release.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="build-", dir=release.parent) as directory:
        stage = Path(directory)
        payload = stage / "payload"
        payload.mkdir()
        print("Preparing the application runtime…", flush=True)
        with ZipFile(cache / "base-runtime.zip") as base:
            for entry in base.infolist():
                if entry.is_dir():
                    continue
                relative = PurePosixPath(entry.filename).relative_to("VQI")
                if ".." in relative.parts:
                    raise ValueError(f"Unexpected base archive path: {entry.filename}")
                if relative.parts[0] in ("deployment", "Open-Visual-Inspection.cmd", "START-HERE.txt"):
                    continue
                item = payload.joinpath(*relative.parts)
                item.parent.mkdir(parents=True, exist_ok=True)
                with base.open(entry) as source, item.open("wb") as target:
                    shutil.copyfileobj(source, target)
        shutil.copytree(PROJECT / "deployment", payload / "deployment",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copyfile(PROJECT / "distribution/Connect-Modal.cmd", payload / "Connect-Modal.cmd")
        shutil.copyfile(SOURCE / "START-HERE.txt", payload / "START-HERE.txt")

        with ZipFile(cache / "webview2.nupkg") as sdk:
            for source, target in SDK_FILES:
                path = payload / target
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(sdk.read(source))
        prerequisites = payload / "prerequisites"
        prerequisites.mkdir()
        name = "WebView2RuntimeX64.exe"
        shutil.copyfile(cache / name, prerequisites / name)
        icon = stage / "Studio.ico"
        draw_icon(icon)
        base_references = ("System.Windows.Forms.dll", "System.Drawing.dll")
        compile_windows(csc, SOURCE / "StudioWindow.cs", payload / "Visual Inspection Studio.exe", icon,
                        (*base_references, *(payload / name for _, name in SDK_FILES[:2])))
        bootstrap = stage / "Setup.exe"
        compile_windows(csc, SOURCE / "StudioSetup.cs", bootstrap, icon,
                        (*base_references, "System.IO.Compression.dll",
                         "System.IO.Compression.FileSystem.dll", "Microsoft.CSharp.dll"))

        print("Packaging the Windows installer…", flush=True)
        archive = stage / "payload.zip"
        with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=1) as files:
            for path in sorted(payload.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(payload).as_posix()
                    method = ZIP_STORED if relative.startswith("prerequisites/") else ZIP_DEFLATED
                    files.write(path, relative, compress_type=method)
        with ZipFile(archive) as files:
            if failure := files.testzip():
                raise ValueError(f"The installer archive is damaged: {failure}")
        with bootstrap.open("ab") as target, archive.open("rb") as source:
            offset = target.tell()
            shutil.copyfileobj(source, target, length=1024**2)
            target.write(struct.pack("<Q8s", offset, b"VQIZIP01"))
        result = release / "Visual-Inspection-Studio-Setup.exe"
        bootstrap.replace(result)
        print(f"{result} ({result.stat().st_size / (1024**2):.0f} MiB)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=OUTPUT / "build-cache")
    parser.add_argument("--release", type=Path, default=OUTPUT / "release")
    options = parser.parse_args()
    build(options.cache.resolve(), options.release.resolve())
