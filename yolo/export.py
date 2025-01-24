from ultralytics import YOLO

model = YOLO('runs/detect/GF-BEST/weights/best.pt')

model.export(format = "engine")