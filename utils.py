import os
from pathlib import Path
import re
import shutil
import cv2

class TrainingUtils:
    
    def _transform_annotations(self,ann_path,img_list,output_ann_path):
        '''Transforms data annotatios (all in the same .txt) to yolo format (one .txt for each image)'''
        #TODO: optimize this code, it's very slow
        pattern = r'\d+'  # Matches one or more digits
        
        with open(Path(ann_path,'gt.txt')) as ann_file:
            
            for line in ann_file:
                matches = re.findall(pattern, line)
                img_idx = int(matches[0]) #First value refers to frame number
                # output_ann_file_path = Path(output_ann_path,img_list[img_idx]).split('.')[:-1]
                filename, extension = os.path.splitext(Path(output_ann_path,img_list[img_idx-1]))
                output_ann_file_path = filename + ".txt"
                with open(output_ann_file_path, 'a') as output_file:
                    #TODO: normalize coordinates to [0,1] and move x,y to the center of the box
                    new_line = f'{matches[6]} {matches[2]} {matches[3]} {matches[4]} {matches[5]}\n'
                    output_file.write(new_line)


    def transform2yolo(self,data_path:str = './data',subsets:list = ['test','train']):
        '''Transforms annotations to yolo format'''

        dst_path = Path(data_path,'yolo')
        # os.makedirs(dst_path,exist_ok=True)

        for subset in subsets:
            dst_subset_path = Path(dst_path,subset)
            # os.makedirs(subset_path,exist_ok=True)
            os.makedirs(Path(dst_subset_path,'images'),exist_ok=True)

            output_ann_path = Path(dst_subset_path,'labels')
            os.makedirs(output_ann_path,exist_ok=True)

            #TODO: use rglob instead. This can also be done using multiprocessing (map())
            for folder in os.listdir(Path(data_path,subset)):
                img_path=Path(data_path,subset,folder,'img1')

                #Move images
                # for file in os.listdir(img_path):
                #     shutil.copy(Path(img_path,file),Path(dst_subset_path,'images'))

                #Transform annotations
                if subset in ['train','valid']:
                    ann_path=Path(data_path,subset,folder,'gt')
                    self._transform_annotations(ann_path,img_list=os.listdir(img_path),output_ann_path=output_ann_path)
                
        

class DetectionUtils:

    def video_visualizer(data_path):
        '''Visualize the video frames in the video_path'''
        video_path = Path(data_path,'img1')
        for item in os.listdir(video_path):
            img = cv2.imread(data_path + item)
            cv2.imshow('img', img)
            cv2.waitKey(100) # wait for 100 milliseconds
    
    def annotation_visualizer(data_path):
        '''Visualize the video frames in the video_path'''
        video_path = Path(data_path,'yolo/train/images')
        
        images = []

        for item in os.listdir(video_path):
            img = cv2.imread(f'data/yolo/train/images/{item}')
            
            ann_file = open(f'data/yolo/train/labels/{item.split(".")[0]}.txt', 'r')
            lines = ann_file.readlines()
            lines = [line.split(' ') for line in lines]
            for line in lines:
                # print(len(line))
                bbox_x = int((line[1]))
                bbox_y = int((line[2]))
                bbox_w = int((line[3]))
                bbox_h = int((line[4]))

                img = cv2.rectangle(img,(bbox_x,bbox_y),(bbox_x+bbox_w,bbox_y+bbox_h),color=(255,0,0))
            images.append(img)

            cv2.imshow('img', img)
            cv2.waitKey(20)

        # height, width, layers = images[0].shape
        # video = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width,height))

        # for image in images:
        #     video.write(image)
        
        # cv2.destroyAllWindows()
        # video.release()
