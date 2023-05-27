import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp3/weights/best.pt')

img = "./data/images/awake-4.jpg"

results = model(img)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    results = model(frame)

    cv2.imshow("frame", np.squeeze(results.render()))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
