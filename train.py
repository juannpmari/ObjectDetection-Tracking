#TODO: use argsparser to launch a model trainining, with passes hyperparameters (one hyp shoul be model, yolo8 or faster-rcnn)
from ultralytics import YOLO
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to pretrained yolov8 model, eg: dir/yolov8n.pt")
ap.add_argument("-d", "--data", type=str, required=True,
                help="path to dataset's data.yaml ")
ap.add_argument("-c", "--cfg", type=str, required=True,
                help="path to cfg.yaml file with training config")
# ap.add_argument("--save", action='store_true',
#                 help="Save video")

args = vars(ap.parse_args())
# source = args["source"]
path_pretrained_model = args["model"]
cfg = args["cfg"]
data = args['data']

if __name__ == "__main__":
    model = YOLO(path_pretrained_model)
    model.train(cfg=cfg,data=data)