# 🚗 Mini ADAS Vision

A hands-on computer vision project that uses a Hugging Face object detection model to build a simple ADAS-style road scene analysis application.

The application detects objects in a road image, performs basic spatial reasoning, estimates a heuristic risk level, and presents the result through an interactive Gradio UI.

> **Note:** This is an educational project for learning Hugging Face computer vision models and application-level reasoning. It is not a production ADAS or safety-critical system.

## 🎯 Project Goal

The goal of this project is to understand how an open-source Hugging Face vision model can be integrated into a complete application.

Instead of stopping at model inference, the project builds an application pipeline around the model:

```text
Road Image
    ↓
Hugging Face Object Detection
    ↓
Detected Objects
    ↓
Spatial Analysis
    ↓
Ego-Lane Analysis
    ↓
Proximity Estimation
    ↓
Heuristic Risk Scoring
    ↓
ADAS Warnings
    ↓
Annotated Image + Summary
    ↓
Gradio UI
```

## ✨ Features

- Object detection using Hugging Face Transformers
- DETR ResNet-50 object detection model
- Bounding-box extraction and visualization
- Ego-lane approximation using a trapezoidal image region
- Bottom-center based spatial reasoning
- Image-based proximity estimation
- Heuristic LOW / MEDIUM / HIGH risk classification
- Aggregated ADAS-style warnings
- Annotated output image
- Interactive Gradio interface
- Notebook-based learning path
- Testing across multiple road scenes

## 🤗 Hugging Face Model

This project uses:

**Model:** `facebook/detr-resnet-50`

DETR (DEtection TRansformer) is an object detection model that combines a convolutional backbone with a Transformer-based detection architecture.

The model is trained on the COCO dataset and can detect common objects such as:

- person
- bicycle
- car
- motorcycle
- bus
- truck
- and other COCO classes

For this project, the model's detections are used as the input to our custom ADAS reasoning layer.

## 🧠 Application Architecture

The Hugging Face model is responsible for **object perception**.

The application is responsible for **reasoning about those detections**.

This separation is one of the main concepts demonstrated by the project.

```text
                 ┌─────────────────────┐
                 │    Road Image       │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Hugging Face DETR   │
                 │ Object Detection    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Detection Results   │
                 │ label / score / box │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   ADAS Analyzer     │
                 ├─────────────────────┤
                 │ Ego-lane            │
                 │ Bottom-center       │
                 │ Proximity           │
                 │ Risk scoring        │
                 │ Warning generation  │
                 └──────────┬──────────┘
                            │
                    ┌───────┴────────┐
                    ▼                ▼
             Annotated Image    ADAS Summary
                    │                │
                    └───────┬────────┘
                            ▼
                       Gradio UI
```

## 🔍 How the ADAS Analysis Works

After object detection, each detected object is passed through a simple spatial and risk-analysis pipeline.

### 1. Detection Filtering

The model can detect many COCO object classes, but not every class is relevant to our ADAS-style analysis.

The application currently considers:

```text
person
bicycle
motorcycle
car
bus
truck
```

Other detected objects are ignored when generating ADAS warnings.

This keeps the warning layer focused on road participants and vehicles.

### 2. Bounding Box Analysis

Each detection contains a bounding box:

```text
[xmin, ymin, xmax, ymax]
```

From this box, we calculate the center:

```text
center_x = (xmin + xmax) / 2
center_y = (ymin + ymax) / 2
```

We also calculate the bottom-center:

```text
bottom_center_x = (xmin + xmax) / 2
bottom_center_y = ymax
```

The bottom-center is particularly useful for determining where an object is positioned relative to the road surface.

### 3. Ego-Lane Approximation

The project uses a trapezoidal region to approximate the ego-lane.

The region is defined using normalized image coordinates:

```text
(0.42, 0.40) ───────── (0.58, 0.40)
      \                    /
       \                  /
        \                /
         \              /
          \            /
           \          /
            \        /
             \      /
              \    /
               \  /
                \/
          (0.10, 1.00)   (0.90, 1.00)
```

The bottom-center of each detected object is tested against this polygon.

If the bottom-center falls inside the polygon:

```text
inside_ego_lane = True
```

Otherwise:

```text
inside_ego_lane = False
```

This is an image-space approximation rather than a real lane-detection system.

### 4. Proximity Estimation

The project does not estimate physical distance.

Instead, it calculates a relative proximity score using:

- vertical position of the object's bottom-center
- relative bounding-box height

Both values are normalized by the image height.

The heuristic is:

```text
proximity_score =
    0.6 × normalized_bottom_y
  + 0.4 × normalized_box_height
```

The score is then grouped into:

```text
score >= 0.50  → NEAR

score >= 0.40  → MEDIUM

score <  0.40  → FAR
```

This is useful for learning spatial reasoning from a single image, but it should not be interpreted as a distance measurement in meters.

### 5. Risk Scoring

The application combines several factors into a simple heuristic risk score:

```text
Risk Score =
    0.30 × Object Severity
  + 0.30 × Ego-Lane Presence
  + 0.25 × Proximity
  + 0.15 × Detection Confidence
```

The resulting score is normalized to the range:

```text
0.0 → 1.0
```

Risk levels are classified as:

```text
score >= 0.60  → HIGH

score >= 0.30  → MEDIUM

score <  0.30  → LOW
```

Object severity is represented using simple class weights. In the current educational heuristic, pedestrians, motorcycles and bicycles are assigned higher weights than cars, buses and trucks. These weights are application-specific assumptions and are not intended to represent real-world collision severity.

### 6. Warning Generation

Warnings are generated only for:

1. ADAS-relevant object classes
2. Objects inside the approximate ego-lane
3. Objects classified as MEDIUM or HIGH risk

Multiple detections of the same object type are aggregated into a single warning.

For example:

```text
HIGH: 4 pedestrians in ego-lane
HIGH: 2 motorcycles in ego-lane
MEDIUM: 1 bus in ego-lane
```

This makes the output easier to understand than displaying a separate warning for every detection.

### 7. Visualization

The application visualizes the analysis by displaying:

- detected bounding boxes
- object center
- object bottom-center
- approximate ego-lane
- risk level
- proximity
- risk score

Low-risk detections are hidden from the final visualization to reduce visual clutter.


## ⚙️ Installation

### Prerequisites

- Python 3.10+
- Git
- Windows, Linux, or macOS
- A CPU is sufficient for this educational project

### Clone the repository

```bash
git clone https://github.com/rushivyas-eng/mini-adas-vision.git
cd mini-adas-vision
```

### 🚀 Running the Application

The main interactive application is built using Gradio.
Start the application with:

```bash
python src/gradio_app.py
```

## 📓 Learning Notebooks

The project is organized as a progressive notebook-based learning path.

| Notebook | Purpose |
|---|---|
| `01_environment_test.ipynb` | Verify the Python environment and required libraries |
| `02_first_inference.ipynb` | Run the first Hugging Face object-detection inference |
| `03_manual_inference.ipynb` | Understand model inputs, logits, classes and bounding boxes |
| `04_application_test.ipynb` | Connect object detection with the ADAS application logic |
| `05_gradio_ui.ipynb` | Learn how to expose the application through Gradio |
| `06_application_testing.ipynb` | Validate the application across multiple road scenes |

Recommended order:

```text
01 → 02 → 03 → 04 → 05 → 06
```

The notebooks intentionally move from low-level model experimentation toward a complete application.

## 📁 Project Structure

```text
mini_adas_vision/
│
├── images/
│   └── road.jpg
│
├── notebooks/
│   ├── 01_environment_test.ipynb
│   ├── 02_first_inference.ipynb
│   ├── 03_manual_inference.ipynb
│   ├── 04_application_test.ipynb
│   ├── 05_gradio_ui.ipynb
│   └── 06_application_testing.ipynb
│
├── src/
│   ├── analyzer.py
│   ├── app.py
│   ├── detector.py
│   └── gradio_app.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

### Source files

**`detector.py`**

Responsible for loading the Hugging Face model and performing object detection.

**`analyzer.py`**

Contains the application-level ADAS reasoning:

- ego-lane analysis
- bottom-center calculation
- proximity estimation
- risk scoring
- warning generation

**`app.py`**

Provides the application-level testing and visualization logic.

**`gradio_app.py`**

Provides the interactive Gradio user interface.


## 🖼️ Example Output

The application produces an annotated image containing:

- detected objects
- bounding boxes
- object centers
- bottom-center points
- approximate ego-lane corridor
- risk levels
- proximity categories
- risk scores

The UI also provides an ADAS summary containing:

```text
Detections
High Risk
Medium Risk
Ego-Lane Objects
```

and aggregated warnings such as:

```text
⚠ HIGH: pedestrians in ego-lane
⚠ HIGH: cars in ego-lane
⚠ MEDIUM: bus in ego-lane
```

The exact results depend on the input image and the detections produced by the model.

## ⚠️ Limitations

This project is intentionally designed as a learning exercise and should not be considered a production ADAS system.

### Generic object detection model

The project uses `facebook/detr-resnet-50`, a general-purpose object detection model trained on COCO.

It is not an automotive-specific perception model.

### Fixed ego-lane approximation

The ego-lane is represented using a fixed trapezoidal region in image coordinates.

It does not dynamically detect road lanes.

Changes in camera position, road geometry, or perspective can therefore affect the result.

### No physical distance estimation

The `NEAR`, `MEDIUM`, and `FAR` categories are derived from image geometry.

They do not represent physical distances such as:

```text
NEAR = 5 meters
```

No depth sensor, stereo camera, LiDAR, or monocular depth model is used.

### Heuristic risk scoring

The risk score is an application-specific heuristic.

It is not:

- collision probability
- time-to-collision
- braking distance
- vehicle trajectory prediction
- safety-certified ADAS logic

### Single-image analysis

The application analyzes individual images.

It does not perform:

- object tracking
- multi-frame analysis
- velocity estimation
- trajectory prediction

### Detection limitations

Detection quality can vary depending on:

- lighting
- image quality
- object size
- occlusion
- camera perspective
- crowded scenes

### No safety guarantees

The output should only be used for experimentation and learning.

It must not be used to make real-world driving or safety decisions.

## 🎓 What I Learned

Through this project, I explored the complete path from a pretrained computer vision model to a usable application.

Key learning areas included:

- Loading pretrained Hugging Face vision models
- Understanding image processors and model inputs
- Interpreting object-detection outputs
- Working with class probabilities and bounding boxes
- Converting normalized bounding boxes into image coordinates
- Building application-specific spatial reasoning
- Using polygon-based region checks
- Designing simple proximity heuristics
- Combining multiple signals into a risk score
- Aggregating model detections into meaningful warnings
- Visualizing computer vision results with Matplotlib
- Building an interactive interface using Gradio
- Structuring a computer vision project into reusable Python modules
- Testing the application across different road scenes

## 🏁 Project Outcome

This project demonstrates how a pretrained Hugging Face computer vision model can be combined with application-level reasoning to create a small end-to-end vision application.

The final pipeline is:

```text
Road Image
    ↓
Hugging Face DETR
    ↓
Object Detection
    ↓
Spatial Analysis
    ↓
Ego-Lane + Proximity
    ↓
Heuristic Risk Scoring
    ↓
ADAS Warnings
    ↓
Annotated Visualization
    ↓
Gradio Application