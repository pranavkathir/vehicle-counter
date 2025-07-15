import cv2
from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# Vehicle classes (COCO IDs)
vehicle_class_ids = [2, 3, 5, 7]

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run detection, restrict to vehicle classes
    results = model.predict(source=frame, classes=vehicle_class_ids, conf=0.3, verbose=False)

    # Draw boxes manually
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{result.names[cls]} {conf:.2f}"

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame with detections
    cv2.imshow("YOLOv8 Vehicle Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
