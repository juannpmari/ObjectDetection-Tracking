Dataset: https://motchallenge.net/data/MOT16/

Description: This benchmark contains 14 challenging video sequences (7 training, 7 test) in unconstrained environments filmed with both static and moving cameras. Tracking and evaluation are done in image coordinates. All sequences have been annotated with high accuracy, strictly following a well-defined protocol.

Project composition:
    * train.py: trains the models and saves them to output path
    * test.py: runs the models on the test set and saves the metrics (f1, mAP) to output path
    * main.py: displays frames sequentially performing object detection with specified model. Optionally tracks the objects.
    * utils.py: contains utility functions, such as loading the dataset, drawing bounding boxes, etc.

Installation
    * python -m venv virtualenv
    * virtualenv/bin/activate
    * Install the requirements: pip install -r requirements.txt
    * Download the dataset and extract it to the data folder

Object detection
YoloV8[https://github.com/ultralytics/ultralytics] is the state-of-the-art object detection model. So we'll train it on the MOT16 dataset. The model is trained on the training set and evaluated on the test set. 
    