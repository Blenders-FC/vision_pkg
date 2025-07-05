from ultralytics import YOLO

# Load a YOLO11n PyTorch model
model = YOLO("best-rhoban-v1.pt")

# Export the model to TensorRT
model.export(format="engine", half=True)  # dla:0 or dla:1 corresponds to the DLA cores