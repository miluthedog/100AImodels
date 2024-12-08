from ultralytics import YOLO

model = YOLO("facedetectYOLO/runs/detect/train2/weights/best.pt")

result = model(source=0,show=True,conf=0.7)