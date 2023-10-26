from ultralytics import YOLO


model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)
model.train(cfg='/yolo/default_copy.yaml',data="/yolo/dataset/ppe-detector-231024/data.yaml",project='ppe-detector',name='yolo8-231025-iou05')#,epochs=200)  # train the model
