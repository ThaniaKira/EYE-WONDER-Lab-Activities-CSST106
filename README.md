# eye✿wonder — Computer Vision & Perception (CSST106)

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

