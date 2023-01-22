# Project: Object Tracking Using Deep Sort and YOLOv3

## Introduction
This is an implement of MOT tracking algorithm deep sort. Deep sort is basicly the same with sort but added a CNN model to extract features in image of human part bounded by a detector. This CNN model is indeed a RE-ID model and the detector used in [PAPER](https://arxiv.org/abs/1703.07402) is FasterRCNN.

I re-implemented the CNN feature extraction model with PyTorch, and changed the CNN model a little bit. Also, I use **YOLOv3** to generate bboxes instead of FasterRCNN.

### STEPS:

Clone the repository

```bash
https://github.com/aibi10/Deep-Sort-Object-Tracking.git
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n deepsort python=3.7.10 -y
```

```bash
conda activate deepsort
```

### STEP 02- install the requirements

```bash
pip install -r requirements.txt
```

### STEP 03- install the following dependencies from conda

```bash
conda install pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch -y
```

### STEP 04 - run this command 

```bash
python detectorTracker.py
```

Voila, your web cam will start.


### STEP 05 - Go to line 13 and put your own video name, if you want to feed some video to the model
```bash
python detectorTracker.py
```

Voila!!! This time your input video will have bounding boxes using deep sort

```bash
Author: Abhishek Singh
Data Scientist
Email: isingh.abhishek10@gmail.com
```
