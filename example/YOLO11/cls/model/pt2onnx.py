from ultralytics import YOLO

model = YOLO("yolo11n-cls.pt")
model.export(format="onnx")