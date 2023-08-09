import cv2

# Load YOLOv3 model and class names
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Open a video capture object
cap = cv2.VideoCapture(0)  # 0 for default camera, or specify video file path

while True:
    ret, frame = cap.read()  # Read a frame from the camera feed

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get output layer names
    layer_names = net.getUnconnectedOutLayersNames()

    # Forward pass and get predictions
    outs = net.forward(layer_names)

    # Initialize lists for detected objects' information
    class_ids = []
    confidences = []
    boxes = []

    # Process model outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = scores.argmax()  # Get the index of the highest score
            confidence = scores[class_id]  # Access the confidence value using the class_id index
            if confidence > 0.5:
                # YOLO returns coordinates as center_x, center_y, width, height
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])

                # Calculate top-left corner coordinates
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])

    # Apply non-maximum suppression to remove redundant boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    for i in range(len(boxes)):
        if i in indices:
            box = boxes[i]
            x, y, width, height = box
            label = classes[class_ids[i]]
            confidence = confidences[i]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

            # Display class label and confidence
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Display the annotated frame
    cv2.imshow("Real-Time Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Exit loop on 'q' key press
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
