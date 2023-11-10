from utils import TrainingUtils
from utils import DetectionUtils 

# print("transforming data to yolo format...")
# TrainingUtils().transform2yolo()
DetectionUtils.annotation_visualizer('data')

# if __name__=='__main__':
    # video_path = 'data/train/MOT16-02' #En realidad sería el path a los videos con las predicciones hechas
    # du.video_visualizer(video_path)    
