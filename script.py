from ultralytics import YOLO

model_v8  = YOLO("models/yolov8.pt")
model_v11 = YOLO("models/yolov11.pt")

print("YOLOv8 classes:", model_v8.names)
print("YOLOv11 classes:", model_v11.names)