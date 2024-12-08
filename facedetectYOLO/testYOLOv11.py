from ultralytics import YOLO

model = YOLO("facedetectYOLO/best.pt")

result = model(source=0,show=True,conf=0.7)