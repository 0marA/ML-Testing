import time
import torch
import cv2
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='yolov5/runs/train/exp5/weights/best.pt')


camera = cv2.VideoCapture(0)

prev_frame, new_frame = 0, 0


while True:
    ret, frame = camera.read()


    if not ret:
        break
    
    new_frame = time.time()

    results = model(frame)

    # Get the detected objects
    detections = results.pandas().xyxy[0]

    annotated_frame = np.squeeze(results.render())

    # Check if any objects are detected
    if len(detections) > 0:
        # Get the first detected object
        detection = detections.iloc[0]

        # Extract the bounding box coordinates
        x1, y1, x2, y2 = map(
            int, detection[['xmin', 'ymin', 'xmax', 'ymax']].values)

        # Calculate center coordinates
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Draw a dot at the center of the detected object
        cv2.circle(annotated_frame, (center_x, center_y), radius=5,
                   color=(0, 255, 0), thickness=-1)

        cv2.putText(annotated_frame, str(center_x) + ', ' + str(center_y), (center_x -
                    10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    fps = str(int(1/(new_frame-prev_frame)))
    
    prev_frame = new_frame
    
    cv2.putText(annotated_frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3)

    cv2.imshow('Live Feed', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
