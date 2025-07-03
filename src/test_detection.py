from ultralytics import YOLO
import cv2

# Load the exported TensorRT model
trt_model = YOLO("best-nv2-62.engine")

# Open the video capture (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run inference on the frame
    results = trt_model(frame)

    # Visualize results on the frame
    annotated_frame = results[0].plot()

    # Show the annotated frame
    cv2.imshow('YOLOv11 TensorRT Detection', annotated_frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
