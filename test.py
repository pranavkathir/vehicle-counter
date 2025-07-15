import cv2

# Open the default webcam (use 0, or 1 for external webcam)
#cap = cv2.VideoCapture(4)  # or 2, 3, etc.

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read one frame from the webcam
    ret, frame = cap.read()

    # If frame was not read successfully, break the loop
    if not ret:
        print("Failed to grab frame.")
        break

    # Display the frame
    cv2.imshow("Webcam Feed", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()
