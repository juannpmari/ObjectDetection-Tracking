import os
from pathlib import Path
import cv2

class DetectionUtils:

    def video_visualizer(data_path):
        '''Visualize the video frames in the video_path'''
        video_path = Path(data_path,'img1')
        for item in os.listdir(video_path):
            img = cv2.imread(data_path + item)
            cv2.imshow('img', img)
            cv2.waitKey(100) # wait for 100 milliseconds
    
        
