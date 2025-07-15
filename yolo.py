# this code imports YOLOv8n (small model) and classifies and draws boxes around only cars, motorcylces, buses, and trucks


from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# COCO class IDs for vehicles: car=2, motorcycle=3, bus=5, truck=7
vehicle_class_ids = [2, 3, 5, 7]

# Run detection on webcam limiting to vehicle classes only and show results
model.predict(source=0, classes=vehicle_class_ids, show=True)
