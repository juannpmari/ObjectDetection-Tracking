Dataset: https://motchallenge.net/data/MOT16/

# Description:
This benchmark contains 14 challenging video sequences (7 training, 7 test) in unconstrained environments filmed with both static and moving cameras. Tracking and evaluation are done in image coordinates. All sequences have been annotated with high accuracy, strictly following a well-defined protocol.

## Technologies used:
Python, Pytorch, YOLOv8, Faster-RCNN

## Annotation format:
https://motchallenge.net/instructions/
Each video directory contains two subdirectories, one with the frames and the other with a file gt.txt with annotations.
This file containes many lines, each with the following information:
* <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
'frame' is the frame number and 'id' is the id of the tracked object. 'bb_left', 'bb_top', 'bb_width', 'bb_height' are the top-left corner coordinates, width and height of the bounding box, respectively. 'conf' is the detection confidence. 'x', 'y', 'z' are the 3D world coordinates of the detection.

For the object detection project, we only need the first 6 columns. The rest are for 3D tracking. The conf value can be used to ignore a line if it's 0.
bounding box coordinates are in pixels, while for yolo model we need them normalized to [0,1]. Yolo format requires:
* <object-class> <x> <y> <width> <height>
where x,y are the center of the bounding box, and width, height are the width and height of the bounding box, all normalized to [0,1].


# Project composition:
    * train.py: trains the models and saves them to output path
    * test.py: runs the models on the test set and saves the metrics (f1, mAP) to output path
    * main.py: displays frames sequentially performing object detection with specified model. Optionally tracks the objects.
    Note: inference is performed offline, so main.py only provides visualization. Future integration with AWS is intended
    * utils.py: contains utility functions, such as loading the dataset, drawing bounding boxes, etc.

# Installation
    * python -m venv virtualenv
    * virtualenv/bin/activate
    * Install the requirements: pip install -r requirements.txt
    * Download the dataset and extract it to the data folder

# Object detection
YoloV8[https://github.com/ultralytics/ultralytics] is the state-of-the-art object detection model. So we'll train it on the MOT16 dataset. The model is trained on the training set and evaluated on the test set.

# Future work:
    * Faster-rcnn training. Comparison with YOLOv8
    * integration with AWS to allow for online inference


# Work status:
[x] Analyze annotation format
[...] dataset transformation to yolo format
[] draw transformed annotations on video to check results

[] model training