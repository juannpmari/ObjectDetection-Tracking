from datetime import datetime
import os
from pathlib import Path
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import PIL
import time
#TODO: use argparser to launch inference with certain hyperam and a certain model.

class Predict:

    def __init__(self,model) -> None:
        self.output_path = None
        self.model = YOLO(model)

    @property
    def output_path(self):
        return output_path
    
    @output_path.setter
    def output_path(self,value):
        if type(value) != str:
            raise Exception("Path must be str")
        self.output_path = value
   
    @classmethod
    def predict(cls,image):
        '''
            Runs predictions over an image/set of images and saves them to a new image/video
            image is a generator object, so we use context manager with
        '''
      
        with image as im:
            pred_result = cls.model(image)
        #Save to image or video in output_path
        #TODO: implementar esta parte
        print(f"Predicted image saved to f{cls.output_path}")
            
    
    @staticmethod
    def load_image(img_path):
       return PIL.Image.open(img_path)
    
    @classmethod
    def load_image_batch(cls,img_batch_path):
        '''Loads all images in img_path to memory for further processing'''
        img_list = os.listdir(img_batch_path)
        img_list = [Path(img_batch_path,im) for im in img_list]
        
        #Use multithreading as it's an i/o bound task
        with ThreadPoolExecutor() as executor:
            results = executor.map(cls.load_image,img_list)
        return results

    #Use classmethod decorator to avoid the need for instantiating the class when calling predict_batch
    @classmethod
    def predict_batch(cls,img_path):
        images = cls.load_image_batch(img_path)

        #Using multiprocessing
        with ProcessPoolExecutor() as executor:
            executor.map(cls.predict,images)
        
        # # Using sequential programming
        # predictions = []
        # for im in images:
        #     predictions.append(cls.predict(im))
            
        #Depending on the computational cost of predict and the overhead of creating multiple processes, one option will be faster than the other
        # in this case, as predicting is expensive, multiprocessing speeds up execution
  


if __name__=='__main__':
    start = time.perf_counter()

    img_path = 'data/yolo/test/images'
    output_path = f'data/yolo/inf_{datetime.now().strftime("%Y%m%d")}'

    predictor = Predict('/models/yolov8l.pt')
    predictor.output_path = output_path
    Predict.predict_batch(img_path)

    finish = time.perf_counter()
    print(f'Execution time {round(finish-start,2)}')
