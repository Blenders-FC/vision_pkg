from ultralytics import YOLO

# Load a YOLO11n PyTorch model
#model = YOLO("yolo11n.pt")

#results = model("bus.jpg")


# Export the model to TensorRT
#model.export(format="engine", half=True)  # dla:0 or dla:1 corresponds to the DLA cores

# Load the exported TensorRT model
trt_model = YOLO("yolo11n.engine")

# Run inference
results = trt_model("bus.jpg")


results[0].save(filename="bus_detected.jpg")