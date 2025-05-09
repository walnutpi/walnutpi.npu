from ultralytics import YOLO

model = YOLO("yolo11n-obb.pt")
model.export(format="onnx")