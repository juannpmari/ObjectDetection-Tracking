from ultralytics import YOLO
#TODO: use argparser to launch inference with certain hyperam and a certain model.

def predict(image,output_path,model = 'yolov8.pt'):
    '''Runs predictions over an image/set of images and saves them to a new image/video'''
    model = YOLO(model)
    pred_result = model(image)
    #Save to image or video in output_path

#Borrar:
# if __name__=='__main__':
#     video_path = 'data/test/MOT16-02' #En realidad sería el path a los videos con las predicciones hechas
#     du.video_visualizer(video_path)    
