from utils import TrainingUtils as tu
from utils import DetectionUtils as du

tu().transform2yolo()

# if __name__=='__main__':
    # video_path = 'data/train/MOT16-02' #En realidad sería el path a los videos con las predicciones hechas
    # du.video_visualizer(video_path)    
