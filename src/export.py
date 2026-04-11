from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("yolov8_center.pt")

# Export the model to TensorRT
model.export(format="engine", half=True)