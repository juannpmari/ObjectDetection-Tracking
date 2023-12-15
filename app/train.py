from ultralytics import YOLO
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to pretrained yolov8 model, eg: dir/yolov8n.pt")
ap.add_argument("-d", "--data", type=str, required=True,
                help="path to dataset's data.yaml ")
ap.add_argument("-c", "--cfg", type=str, required=True,
                help="path to cfg.yaml file with training config")
ap.add_argument("-p", "--project", type=str, required=True,
                help="result will be saved to ./project/name")
ap.add_argument("-n", "--name", type=str, required=True,
                help="result will be saved to ./project/name")

args = vars(ap.parse_args())
path_pretrained_model = args["model"]
cfg = args["cfg"]
data = args['data']
project = args['project']
name = args['name']

if __name__ == "__main__":
    model = YOLO('/models/yolov8n.pt')
    results = model.train(cfg='/app/default_copy.yaml',data='/dataset/data.yaml',project=project,name=name)#, epochs=200, imgsz=640)
    model.export()

#Example usage: # example usage: python app/train.py -m models/yolov8n.pt -d dataset/data.yaml -c default_copy.yaml -p yolov8 -n run2