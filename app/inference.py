import argparse
from datetime import datetime
import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import PIL
import time

class Predict:

    def __init__(self,model,output_path) -> None:
        self._output_path = output_path
        os.makedirs(Path(self._output_path),exist_ok=True)
        self.model = YOLO(model)

    @property
    def output_path(self):
        return self._output_path
    
    @output_path.setter
    def output_path(self,value):
        # if type(value) != str:
        #     raise Exception("Path must be str")
        self._output_path = value
   
    def predict(self,image):
        '''
            Runs predictions over an image/set of images and saves them to a new image/video
            image is a generator object, so we use context manager with
        '''
        with image as im:
            try:
                pred_result = self.model(np.array(im))
                print(type(pred_result))
            except Exception as e:
                print(f"error {e}")

        # Show the results
        for r in pred_result:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = PIL.Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            im.save(f'{self._output_path}/{datetime.now().strftime("%Y%m%d-%H%M%S%f")}.jpg')  # save image
       
        print(f"Predicted image saved to {self._output_path}")
            
    
    @staticmethod
    def load_image(img_path):
       return PIL.Image.open(img_path)
    
    # @classmethod
    def load_image_batch(self,img_batch_path):
        '''Loads all images in img_path to memory for further processing'''
        img_list = os.listdir(img_batch_path)
        img_list = [Path(img_batch_path,im) for im in img_list]
        
        #Use multithreading as it's an i/o bound task
        with ThreadPoolExecutor() as executor:
            results = executor.map(self.load_image,img_list)
        return results

    def predict_batch(self,img_path):
        images = self.load_image_batch(img_path)

        #Using multiprocessing
        # with ProcessPoolExecutor(max_workers=2) as executor:
        #     executor.map(self.predict,images)
        
        # # Using sequential programming
        predictions = []
        for im in images:
            predictions.append(self.predict(im))
            
        #Depending on the computational cost of predict and the overhead of creating multiple processes, one option will be faster than the other  




ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to yolov8 model for inference, eg: dir/yolov8n.pt")
ap.add_argument("-i", "--images", type=str, required=True,
                help="path to images")
ap.add_argument("-p", "--project", type=str, required=True,
                help="result will be saved to ./project/name")
ap.add_argument("-n", "--name", type=str, required=True,
                help="result will be saved to ./project/name")

args = vars(ap.parse_args())
path_inf_model = args["model"]
images = args['images']
project = args['project']
name = args['name']


if __name__=='__main__':
    start = time.perf_counter()

    output_path = f'/dataset/yolo/inf_{datetime.now().strftime("%Y%m%d")}'

    predictor = Predict(path_inf_model,Path('/dataset',project,name))
    predictor.predict_batch(images)

    finish = time.perf_counter()
    print(f'Execution time {round(finish-start,2)}')


# python app/inference.py -m app/yolov8/first_run/weights/best.pt -i /dataset/test/images -p yolov8 -n first_run_inference