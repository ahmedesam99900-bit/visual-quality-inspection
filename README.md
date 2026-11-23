# Visual Quality Inspection

A local English web application for inspecting PCB1 and training versioned models for other visual inspection products.

## Install and share the Windows desktop application

Distribute `outputs/windows-desktop/release/Visual-Inspection-Studio-Setup.exe`. Open the installer and select **Install**. Launch **Visual Inspection Studio** from the Windows Start menu. The application runs in its own window and includes Python and a pretrained PCB1 starter model for consistently framed crops. It stores products, images and trained models in the current user's local application data folder.

For Modal training, use **Help > Connect Modal GPU** inside the desktop application. The setup and build files are in [distribution/windows](distribution/windows/README.md).

## Run on Windows

In the existing project, run `deployment\start_app.cmd`. It uses the project's Python 3.12 environment and opens the local application. Start with **Load sample image** and **Run inspection** in the PCB1 view.

On a clean checkout, create a Python 3.12 environment at `.venv` and install `requirements.txt` before running the launcher. For example, from the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cpu torchvision==0.29.1+cpu --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\deployment\start_app.cmd
```

Restore these local artifacts to enable the built-in PCB1 inspection:

| Artifact | Project-relative path |
| --- | --- |
| YOLO11n product detector | `outputs/train_runs/pcb1_yolo_cpu/weights/best.pt` |
| PCB1 PatchCore checkpoint | `outputs/patchcore_runs/pcb1_patchcore_cpu/Patchcore/pcb1_patchcore_dev/v1/weights/lightning/model.ckpt` |
| Sample inspection image | `data/visa/pcb1/Data/Images/Anomaly/041.JPG` |

## Add a product

1. Create a product from the workspace and upload normal images to **Training**.
2. Upload labeled normal and defective images to **Validation**. If you add a test split, include both classes and reserve those images for measuring the trained version.
3. Start **Modal GPU** training after connecting the account. **Local CPU** is also available. The job uses the images present when you submit it.
4. Review the metrics, activate the version, and run an inspection with a new image. Activation is explicit. Previous versions stay available.

Custom models learn spatial patch features from normal images and use validation examples to set the image threshold. Use consistent product framing, distance and lighting across uploads and inspections. PCB1 uses its own YOLO crop followed by PatchCore. The heatmap shows normalized patch differences. Production acceptance needs representative photos and defect examples from the target camera.

## Connect cloud training

On Windows, run `deployment\connect_modal.cmd` and sign in to the Modal workspace you intend to use. You can also run the setup commands from the project root. [Modal setup](https://modal.com/docs/guide/getting-started).

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cloud.txt
.\.venv\Scripts\modal.exe setup
.\.venv\Scripts\modal.exe deploy deployment/modal_worker.py
```

Reopen the application after setup. The worker requests one L4 GPU for a job, up to one hour. Training images and the resulting artifact go to the private `vqi-training-artifacts` volume in the active Modal workspace. Job IDs persist locally so the app can resume checking submitted cloud jobs after restart. Manage retained cloud data and billing from the Modal workspace. [GPU requirements](https://modal.com/docs/guide/gpu) · [Modal volumes](https://modal.com/docs/guide/volumes).

## Source and checks

- `deployment/web/` contains the inspection, datasets, training, and model views.
- `deployment/app.py` serves the local web app. `workspace.py` manages SQLite state under `outputs/inspection_workspace/`.
- `deployment/inspect_pcb1.py` runs the built-in PCB pipeline and renders the heatmap. `trainer.py` creates custom model artifacts. `cloud.py` and `modal_worker.py` connect the GPU worker.
- `src/` contains the original PCB1 data and model scripts.
- The test suite exercises upload, training, activation and inference through the web API. Run it with `python -m unittest discover -s tests -v`. GitHub Actions runs the suite on pushes and pull requests.

The PCB1 banner crop is derived from [VisA PCB1](https://github.com/amazon-science/spot-diff), image 041, by Yang Zou, Jongheon Jeong, Latha Pemula, Dongqing Zhang, and Onkar Dabeer. It was cropped and tinted for display under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
