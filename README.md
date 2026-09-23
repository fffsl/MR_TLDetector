# MR_TLDetector

**Meteorological Resilient Transmission Line Defect Detection System**  

MR_TLDetector is a PyQt5-based desktop application for transmission line defect detection under challenging meteorological conditions. It integrates YOLO-based object detection, optional image preprocessing, visual result display, detection record management, and CSV export in a single graphical interface.  

## Features

- Graphical detection interface for image, image-folder, video, and camera inputs.
- YOLO model loading through user-selected weight files.
- Dataset configuration loading through user-selected YAML files.
- Optional meteorological image preprocessing, including dehazing, deraining, desnowing, low-light enhancement, contrast enhancement, brightness correction, white balance, sharpening, and denoising.
- Real-time display of detection boxes, confidence scores, object categories, coordinates, and inference time.
- Detection result filtering by object class.
- Export of detection records to CSV.

The functions of rows 1–10 in the function bar on the right side of the main interface are as follows:

- Function 1: Select Image File  # Single image input;
- Function 2: Select Image Folder # Batch image file input;
- Function 3: Select Video File # Video input;
- Function 4: Select Camera Source # Camera invocation;
- Function 5: IoU threshold: the IoU parameter during object detection; only when the IoU of the detection box is greater than this value will the result be displayed; default 0.45;
- Function 6: Confidence threshold: the conf parameter during object detection; only when the confidence of the detected object is greater than this value will the result be displayed; default 0.45;
- Function 7: Image inference size: the fixed size to which the image is resized during inference; default 640;
- Function 8: Dataset configuration file (.yaml);
- Function 9: The model to be used for inference (.pt), generally the optimal model after training;
- Function 10: Preprocessing algorithms, including DCP (dehazing), Restormer Derain, SnowFormer Desnow, LIME-Lite, CLAHE, OneRestore.

## Project Structure

```text
MR_TLDetector/
├── GUI.py                  # Main PyQt5 graphical application
├── MR_TLDetector/          # Packaged Python module
│   ├── __init__.py
│   ├── __main__.py         # Module entry point
│   ├── inference.py        # Reusable inference components
│   ├── paths.py            # Runtime path helpers
│   └── restoration.py      # Restormer, SnowFormer, OneRestore
├── external/               # the Quark Cloud Drive link： https://pan.quark.cn/s/02a6ee43fedd
│   ├── OneRestore/         # Universal restoration model code and checkpoints
│   ├── Restormer/          # Dedicated deraining model code and checkpoint
│   └── SnowFormer/         # Dedicated desnowing model code
├── Remote_sensing test     #  Remote sensing test
├── CTIR_test               #  CTIR test
├── UI/
│   └── main.ui             # Qt Designer interface file
├── icon/                   # Application icons and UI images
├── test/                   # Sample images and model/config files
├── output/                 # Runtime detection output
└── requirements.txt        # Python dependencies
```

## Environment

The project is intended for Python on Windows and uses PyQt5, OpenCV, NumPy, PyYAML, Ultralytics, and related runtime dependencies.

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Running the Application

Run from the project root:

```powershell
python GUI.py
```

## Basic Workflow
DCP (dehazing), Restormer Derain, SnowFormer Desnow, , CLAHE, OneRestore

1. Select a YAML data configuration file.
2. Select a YOLO weight file.
3. Choose an input source: image, image folder, video file, or camera.
4. Select preprocessing methods if the input is affected by fog, rain, snow, low light, noise, or other visual degradation.
5. Click **Start** to run detection.
6. Review detection results in the image display area and result table.
7. Export records with **Export Data** when needed.

## Supported Preprocessing Methods

- `DCP (Dehazing)`: dehazing based on dark-channel prior.
- `Restormer Derain`: rain interference reduction.Restormer deraining with the local `external/Restormer/Deraining/pretrained_models/deraining.pth` checkpoint
- `SnowFormer Desnow`: snow interference reduction. the GUI shows `SnowFormer Desnow` only when `external/SnowFormer/pretrained_models/SnowFormer_CSD.pth` is present. 
- `LIME-Lite`: low-light enhancement.
- `CLAHE(local contrast)`: local contrast enhancement.
- `OneRestore`: brightness correction.

  ## Supported Preprocessing Methods

- `DCP (Dehazing)`: dehazing based on dark-channel prior, atmospheric-light estimation from the brightest dark-channel pixels, and guided-filter transmission refinement.
- `Restormer Derain`: Restormer deraining with the local `external/Restormer/Deraining/pretrained_models/deraining.pth` checkpoint, followed by vertical rain-streak cleanup and mild brightening for the detector images.
- `Fast Desnow (Snow Spots)` / `SnowFormer Desnow`: the GUI shows `SnowFormer Desnow` only when `external/SnowFormer/pretrained_models/SnowFormer_CSD.pth` is present. Without that checkpoint, it shows `Fast Desnow (Snow Spots)` and uses a lightweight snow-spot mask, Telea inpainting, edge-preserving blending, and mild brightening.
- `LIME-Lite`: a lightweight illumination-map enhancement with highlight limiting. It uses the channel-maximum illumination estimate from LIME, but substitutes bilateral smoothing and a custom gain for LIME's structure-aware refinement.
- `CLAHE (Local Contrast)`: contrast-limited adaptive histogram equalization on the LAB luminance channel for local contrast enhancement.
- `OneRestore Auto`: OneRestore with the real-scene checkpoint and automatic degradation embedding, followed by conservative luminance, color-shift, highlight, and edge-preserving constraints to reduce visual distortion.

## Output

Detection outputs are saved under the `output/` directory by default. Each run creates a timestamped folder containing processed results and copied source images. CSV files exported from the interface are also saved in the corresponding output directory.

The output directory can be customized with the `INSULATOR_OUTPUT_DIR` environment variable.

## Notes

- The model weights and YAML configuration are selected at runtime, so the detection model can be replaced without changing the main interface code.
- The graphical interface text and font names have been internationalized where appropriate.
- Training scripts are not included in the current streamlined application directory. This repository focuses on application-side detection and deployment.
- Software packages that don't require environment setup can be loaded via GitHub Releases. Users only need to modify the images, weight files, and dataset configuration files to use it.
- It is recommended to go directly to Releases to download and use the already packaged exe software.

