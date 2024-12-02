# Project description
This project demonstrates the use of a variety of computer vision models for object detection in different environments. Model performance in terms of detection accuracy and inference time are compared for one-stage detectors (YOLOv8) and two-stage detectors (Faster-RCNN).
Training and inference under different hyperparameters are also compared.
All the code is written in Python, and CUDA is used for GPU acceleration during training and inference stages.

## Concepts and technologies involved:
### Containerization
* Docker
* Orchestration: Docker-compose

### ML and computer vision
* Framework: Pytorch
* Models: YOLOv8, Faster-RCNN
* CUDA for GPU acceleration
* Evaluation metrics: mAP, recall, precission, F1-score

### Advanced Python concepts used:
* Concurrent programming: multithreading and multiprocessing
* Decorators: @staticmethod, @classmethod, @property
* Iterators and generators

## Dataset description
Dataset used: https://motchallenge.net/data/MOT16/
This benchmark contains 14 challenging video sequences (7 training, 7 test) in unconstrained environments filmed with both static and moving cameras. Tracking and evaluation are done in image coordinates. All sequences have been annotated with high accuracy, strictly following a well-defined protocol.
The first part of the project will be focused on one of these video sequences, MOT16-02, for the sake of computational resources, where the only class to detect are persons. As a challenge, there's an importante variation in object size (there's people both near and far away from the camera), as well as the presence of numerous objects.
Addition of more videos and object classes is intended for the future, depending on resource availability, as well as integration with cloud environments.

### Annotation format:
https://motchallenge.net/instructions/
Each video directory contains two subdirectories, one with the frames and the other with a file gt.txt with annotations.
This file containes many lines, each with the following information:
* 'frame' 'id', 'bb_left', <'bb_top'>, <'bb_width'>, <'bb_height'>, <'conf'>, <'x'>, <'y'>, <'z'>
'frame' is the frame number and 'id' is the id of the tracked object. 'bb_left', 'bb_top', 'bb_width', 'bb_height' are the top-left corner coordinates, width and height of the bounding box, respectively. 'conf' is the detection confidence. 'x', 'y', 'z' are the 3D world coordinates of the detection.

For the object detection project, we only need the first 6 columns. The rest are for 3D tracking. The conf value can be used to ignore a line if it's 0.
bounding box coordinates are in pixels, while for yolo model we need them normalized to [0,1]. Yolo format requires:
* <object-class> <x> <y> <width> <height>
where x,y are the center of the bounding box, and width, height are the width and height of the bounding box, all normalized to [0,1].


## Project composition:
### train.py
trains the models and saves them to output path, together with training metrics.
Arguments:
* -m: pretrained model path
* -d: dataset's data.yaml file
* -c: configuration .yaml file 
* -p: project, output results will be saved to project/name
* -n: name, output results will be saved to project/name
example usage: python app/train.py -m models/yolov8n.pt -d dataset/data.yaml -c default_copy.yaml -p yolov8 -n run2

### inference.py
performs inference over a set of images and saves the predictions to output path.
Note: inference is performed offline, so only resulting images and videos are provided. Future integration with AWS is intended to allow for online inference.


### utils.py
contains several useful functions:
* transform2yolo: transforms dataset from input format to yolo format, which YOLOv8 requires for traininig, as specified in "Annotation format" section
* train_test_split: splits input dataset into 3 subsets for trainining, validation and testing
* annotation_visualizer: draws annotated bounding boxes on the images and displays them as a video

### preprocess_data.py
orchestrates the flow for image preprocessing until achieving a dataset ready for traininig and evaluation.

## Installation
All the code is dockerized, so it's only necessary to run 'docker-compose up' command.
Hardware requirements: model training and inference was carried out with an Nvidia RTX3080 and CUDA 11. If you don't have GPU, there are videos showing model inference results.


## Object detection results

### YOLOv8
YOLOv8[https://github.com/ultralytics/ultralytics] is the state-of-the-art object detection model. So we'll train it on the MOT16 dataset. The model is trained on the training set and evaluated on the test set.


# Future work:
* tracker integration (DeepSORT) to track objects
* Faster-rcnn training. Comparison with YOLOv8
* integration with AWS to allow for online inference
