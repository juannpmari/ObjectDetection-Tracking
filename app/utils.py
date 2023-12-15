from datetime import datetime
import itertools
import os
from pathlib import Path
import random
import re
import shutil
import time
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class TrainingUtils:
    
    @staticmethod
    def _transform_one_annotation(line,idx,pattern,img_path,img_list,output_ann_path):
        print(f"Opening with line {idx}")
        matches = re.findall(pattern, line)
        img_idx = int(matches[0]) #First value refers to frame number
        filename, extension = os.path.splitext(Path(output_ann_path,img_list[img_idx-1]))
        h,w = cv2.imread(str(Path(img_path,img_list[img_idx-1]))).shape[:2]
        output_ann_file_path = filename + ".txt"
        with open(output_ann_file_path, 'a') as output_file:

            #Normalize values to [0,1], and then move x,y to the center of the box
            bbox_x_tl = float(matches[2])/w #bbox x top left
            bbox_y_tl = float(matches[3])/h #bbox y top left
            bbox_w = float(matches[4])/w #bbox width
            bbox_h = float(matches[5])/h

            new_line = f'0 {bbox_x_tl+bbox_w/2} {bbox_y_tl+bbox_h/2} {float(matches[4])/w} {float(matches[5])/h}\n'
            output_file.write(new_line)

    @classmethod
    def _transform_annotations(cls,ann_path,img_path,output_ann_path):
        '''Transforms data annotatios (all in the same .txt) to yolo format (one .txt for each image)'''
        #TODO: optimize this code, it's very slow
        pattern = r'\d+'  # Matches one or more digits
        img_list = os.listdir(img_path)
        
        start = time.perf_counter()

        with open(Path(ann_path,'gt.txt')) as ann_file:
            lines = [line for line in ann_file]
            idxs = [idx for idx,_ in enumerate(lines)]
            with ThreadPoolExecutor() as executor:
                executor.map(cls._transform_one_annotation,lines,idxs,
                             itertools.repeat(pattern,len(lines)),
                             itertools.repeat(img_path,len(lines)),
                             itertools.repeat(img_list,len(lines)),
                             itertools.repeat(output_ann_path,len(lines)))
                
            ## Sequential alternative (more than 2x slower)
            # for line in tqdm(ann_file, desc="Transforming annotations"):
            #     matches = re.findall(pattern, line)
            #     img_idx = int(matches[0]) #First value refers to frame number
            #     filename, extension = os.path.splitext(Path(output_ann_path,img_list[img_idx-1]))
            #     h,w = cv2.imread(str(Path(img_path,img_list[img_idx-1]))).shape[:2]
            #     output_ann_file_path = filename + ".txt"
            #     with open(output_ann_file_path, 'a') as output_file:

            #         #Normalize values to [0,1], and then move x,y to the center of the box
            #         bbox_x_tl = float(matches[2])/w #bbox x top left
            #         bbox_y_tl = float(matches[3])/h #bbox y top left
            #         bbox_w = float(matches[4])/w #bbox width
            #         bbox_h = float(matches[5])/h

            #         new_line = f'{matches[6]} {bbox_x_tl+bbox_w/2} {bbox_y_tl+bbox_h/2} {float(matches[4])/w} {float(matches[5])/h}\n'
            #         output_file.write(new_line)
        print("Done converting to yolo")
        finish = time.perf_counter()
        print(f'time {round(finish-start,2)}')


    @classmethod
    def transform2yolo(cls,data_path:str = './data',subsets:list = ['test','train'],move_imgs = False):
        '''Transforms annotations to yolo format'''

        dst_path = Path(data_path,'yolo')
        # os.makedirs(dst_path,exist_ok=True)

        for subset in subsets:
            dst_subset_path = Path(dst_path,subset)
            # os.makedirs(subset_path,exist_ok=True)
            os.makedirs(Path(dst_subset_path,'images'),exist_ok=True)

            output_ann_path = Path(dst_subset_path,'probando')#'labels')
            os.makedirs(output_ann_path,exist_ok=True)

            #TODO: use rglob instead. This can also be done using multiprocessing (map())
            for folder in os.listdir(Path(data_path,subset)):
                img_path=Path(data_path,subset,folder,'img1')

                #Move images
                if move_imgs:
                    with ThreadPoolExecutor() as executor:
                        source_list = [Path(img_path,file) for file in os.listdir(img_path)]
                        dest_list = [Path(dst_subset_path,'images') for i in range(len(source_list))]
                        executor.map(shutil.copy,source_list,dest_list)
                    
                    #Sequential alternative
                    # for file in os.listdir(img_path):
                    #     shutil.copy(Path(img_path,file),Path(dst_subset_path,'images'))

                #Transform annotations
                if subset in ['train','valid']:
                    ann_path=Path(data_path,subset,folder,'gt')
                    cls._transform_annotations(ann_path,img_path=img_path,output_ann_path=output_ann_path)
                
    @staticmethod
    def train_test_split(data_path:str = './data/yolo/train',train_perc:float = 0.8,valid_perc:float=0.1):
        '''Split data into train, validation and test sets'''
        
        output_dataset = f'dataset_{datetime.now().strftime("%Y%m%d")}'
        

        full_img_list = os.listdir(Path(data_path,'images'))
        full_label_list = os.listdir(Path(data_path,'labels'))

        data = list(zip(full_img_list, full_label_list))
        random.shuffle(data)

        train_size = int(train_perc*len(full_img_list))
        valid_size = int(valid_perc*len(full_img_list))

        train_data = data[:train_size]
        val_data = data[train_size:train_size+valid_size]
        test_data = data[train_size+valid_size:]
        

        data_types = ['images','labels']
        for i in range(0,len(data_types)):
            os.makedirs(Path(output_dataset,'train',data_types[i]),exist_ok=True)
            os.makedirs(Path(output_dataset,'valid',data_types[i]),exist_ok=True)
            os.makedirs(Path(output_dataset,'test',data_types[i]),exist_ok=True)
            list(map(lambda x:shutil.copy(Path(data_path,data_types[i],x[i]),Path(output_dataset,'train',data_types[i])),train_data))
            list(map(lambda x:shutil.copy(Path(data_path,data_types[i],x[i]),Path(output_dataset,'valid',data_types[i])),val_data))
            list(map(lambda x:shutil.copy(Path(data_path,data_types[i],x[i]),Path(output_dataset,'test',data_types[i])),test_data))     

        #Create data.yaml file
        with open('data.yaml','w+') as file:
            file.write(f"train: {Path(output_dataset,'train')}")
            file.write(f"val: {Path(output_dataset,'valid')}")
            file.write(f"test: {Path(output_dataset,'test')}")
            file.write('nc: 1')
            file.write("classes: ['person']")

           

class DetectionUtils:

    def video_visualizer(data_path):
        '''Visualize the video frames in the video_path'''
        video_path = Path(data_path,'img1')
        for item in os.listdir(video_path):
            img = cv2.imread(data_path + item)
            cv2.imshow('img', img)
            cv2.waitKey(100) # wait for 100 milliseconds
    
    def annotation_visualizer(data_path, save_video=True):
        '''Visualize the video frames in the video_path with annotations'''
        video_path = Path(data_path, 'yolo/train/images')

        fps = 30
        first_img = cv2.imread(f'data/yolo/train/images/{os.listdir(video_path)[0]}')  # Get size from first image
        size_y, size_x, _ = first_img.shape

        video_writer = cv2.VideoWriter('./annotations.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size_y, size_x))

        for item in os.listdir(video_path):
            img = cv2.imread(f'data/yolo/train/images/{item}')

            ann_file = open(f'data/yolo/train/labels/{item.split(".")[0]}.txt', 'r')
            lines = ann_file.readlines()
            lines = [line.split(' ') for line in lines]

            for line in lines:
                bbox_x_c = int(float(line[1])*size_x)
                bbox_y_c = int(float(line[2])*size_y)
                bbox_w = int(float(line[3])*size_x)
                bbox_h = int(float(line[4])*size_y)
                
                img = cv2.rectangle(img, (bbox_x_c-int(bbox_w/2), bbox_y_c-int(bbox_h/2)), (bbox_x_c + int(bbox_w/2), bbox_y_c + int(bbox_h/2)), color=(255, 0, 0))
                img = cv2.putText(img, line[0], (bbox_x_c, bbox_y_c - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.imwrite(f'data/annot_images/{item}', img)

            video_writer.write(img)
            cv2.imshow('img', img)
            cv2.waitKey(20)
            
        video_writer.release()
        cv2.destroyAllWindows()
