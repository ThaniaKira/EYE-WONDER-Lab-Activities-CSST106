# VisionLab ✿ — Computer Vision & Perception (CSST106)

A responsive, purple, floral-futuristic **Flask Web UI** bringing together seven computer-vision laboratory exercises. Built with custom CSS, a robot illustration, animated flower mascot, Python, OpenCV, scikit-learn, and optional deep learning dependencies.

> **Status:** runnable application and lab demos included. Advanced experiments require local model/data setup. **Do not describe missing custom YOLO training or unavailable SURF/DeepLab runs as completed** in a laboratory report.

## Run locally

Use Python 3.10–3.12 and create a virtual environment:

```powershell
# Windows (PowerShell)
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python app.py
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in your browser. Camera capture works on `localhost` or HTTPS with browser permission. A simple flower drawing is bundled as a reusable sample.

To run YOLO11 and semantic DeepLabV3, also install:

```bash
pip install -r requirements-advanced.txt
```

YOLO11 pretrained weights (`yolo11n.pt`) and DeepLabV3 weights may download automatically the first time they are used. Pretrained models may be slow on a CPU. The optional advanced packages may require compatible CPU/GPU-specific installation instructions from PyTorch.

## Seven interactive modules

| Module | Features | Practical notes |
|---|---|---|
| 01 Image Manipulation | Grayscale, resize, rotate, crop, selected pixel/BGR values and marker | Runs with core dependencies |
| 02 Edge Detection | Gaussian blur, Sobel gradient magnitude, Canny threshold sliders | Runs with core dependencies |
| 03 Histogram | Global equalization, CLAHE, color luminance equalization, contrast and histogram graphs | Runs with core dependencies |
| 04 Feature Detection | SURF/SIFT/ORB choice, keypoint visualizations, two-image descriptor matching | SIFT/ORB are the default runnable alternatives; SURF requires a nonfree-enabled OpenCV build |
| 05 PCA | Single-image compression, explained-variance chart, **LFW Eigenfaces + SVM** experiment | Actual LFW face dataset may download on first use |
| 06 YOLO11 | Object detection on uploaded image, optional video and camera still | Needs `ultralytics`; custom training is separate |
| 07 Segmentation | Otsu, adaptive threshold, region growing, watershed, Canny contours and optional DeepLabV3 | DeepLabV3 needs PyTorch/TorchVision and pretrained weights |

## Custom YOLO training and evaluation (Lab 06)

Your lab handout specifies **YOLOv11**, even though the uploaded filename mentions YOLOv8. The weight filename used by current Ultralytics code is `yolo11n.pt`.

1. Collect and annotate your real images using LabelImg/Roboflow and export in YOLO format (`class_id x_center y_center width height`, normalized).
2. Organize `datasets/objects/images/{train,val,test}` and `datasets/objects/labels/{train,val,test}`. Build a `data.yaml` referencing those paths, the class count, and class names. Keep test images unseen during training.
3. Train, validate, and optionally evaluate on your separate test split:

```bash
python scripts/train_yolo.py --data datasets/objects/data.yaml --epochs 50 --test
```

4. The script records precision, recall, mAP@.50 and mAP@.50:.95 in `visionlab_metrics.json`. Report **real metrics** measured with your own dataset. Training is not a button in this Web UI; it uses the included script.
5. To load your locally trained best weights in the Web UI:

```powershell
$env:VISIONLAB_YOLO_WEIGHTS="runs/visionlab/custom-detection/weights/best.pt"
python app.py
```

On macOS/Linux use `export VISIONLAB_YOLO_WEIGHTS="..."`.

## Limitations and honesty notes

- **SURF:** Standard `opencv-contrib-python` wheels often ship without the nonfree SURF implementation. The app returns an explicit error instead of silently labeling an ORB/SIFT result as SURF. Genuine SURF experiments need a custom, nonfree-enabled OpenCV build compatible with your system.
- **Eigenfaces:** The LFW experimental mode uses the required `min_faces_per_person=70`, 80/20 split, 150 whitened principal components (if feasible), and balanced RBF SVM. It computes test-set results each time; don't claim a fixed accuracy without running it. Single-image PCA is an extra exploratory technique, **not** a substitute for Eigenfaces.
- **DeepLabV3:** A pretrained semantic segmentation model runs only when its packages and model weights are installed; its classes depend on those included in the weights.
- **Video:** Uploaded videos are processed for up to 180 frames; outputs are stored in `media/` locally until manually removed. `mp4v` encoding is supported by many OpenCV installs but some browsers may require a compatible conversion. Do not upload private videos to a public demo.
- **File size:** 32 MB total request limit. Image resolution is scaled down for browser-friendly processing.
- This application works locally and does not deploy itself to GitHub or a hosting provider.

## GitHub steps

```bash
git init
git add .
git commit -m "Build VisionLab CSST106 web application"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/VisionLab-CSST106.git
git push -u origin main
```

Replace `YOUR-USERNAME` and create the repository on GitHub first. Keep personal datasets, private images, trained weights, and generated videos out of the repo; `.gitignore` excludes these.

## Printed hard-copy report checklist

For every laboratory (01–07), document: **objectives; background; tools and libraries; procedure; algorithm(s); implementation screenshots; actual output results; observations and limitations; answers to handout questions; conclusions.** Include a cover page, table of contents, references, and appendices where appropriate. Screenshot the relevant UI after each actual successful experiment and ensure printed results match your running code.

## Local project structure

```text
app.py                      Flask routes and API
modules/                    Python algorithms for seven labs
scripts/train_yolo.py       Optional custom training/validation CLI
static/css/style.css        Purple/floral futuristic design
static/js/app.js            Uploads, camera, processing and rendering
static/assets/              Original SVG robot, animated flower, demo image
templates/                  Dashboard, lab, guide and shared layout
models/                     Local model files (ignored)
datasets/                   Your private datasets (ignored)
media/                      Local processed videos (ignored)
reports/                    Your report images and report work
```

**Design direction:** dreamy lavender, plum gradients, pink accents, retro floral flourishes and a holographic robot. Artwork is original to this project and does not copy the provided inspiration screenshots.
