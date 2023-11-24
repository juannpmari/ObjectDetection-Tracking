from utils import TrainingUtils
from utils import DetectionUtils 

#TODO: use argparser to call this file from cmd and decide what to do: transform annotations to yolo, visualize them

# print("transforming data to yolo format...")
# TrainingUtils().transform2yolo(move_imgs=True)
# TrainingUtils().train_test_split()

DetectionUtils.annotation_visualizer('data')