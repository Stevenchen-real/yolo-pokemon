from ultralytics import YOLO

model = YOLO("yolo11n.pt")

if __name__ == '__main__' :
    model.train(data = 'dataset-Golden/data.yaml', epochs = 1500, batch = 4, imgsz = 1280, name = 'GF', patience = 0)