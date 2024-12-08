from ultralytics import YOLO

model = YOLO('faceDetectYOLO/best.onnx', task='detect')

results = model(source=0, stream=True, conf = 0.1)