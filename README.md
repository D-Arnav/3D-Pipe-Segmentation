# 3D Pipe Segmentation and Cylinder Fitting

## Problem Setup

### Goal

Given a pointcloud construction of an industrial area with pipes, fit a cylinder to pipes present.

### Approach

1. Reconstruct 3D scene from pointclouds and capture image and camera configuration (intrensic, extrensic) via virtual pinhole camera.
2. Perform 2D pipe segmentation from camera rendering via RoboFlow API.
3. Project pointcloud onto the mask to identify points on the segmentation mask.
4. Fit cylinder to these points.
5. Overlay cylinder onto the existing pointcloud and visualize.

## Setup

- Python Version 3.9.25

### Requirements

```txt
inference_sdk==1.1.2
numpy==2.4.4
open3d==0.18.0
opencv_python==4.10.0.84
scipy==1.17.1
```

### RoboFlow API Key
- Enter free API key from [Roboflow](https://roboflow.com) into config dictionary (line 19 of `main.py`)

### Dataset

- [Pump.e57](http://www.libe57.org/data.html) converted to .ply
- Already included in this repo

### Running the code

```bash
python main.py
```