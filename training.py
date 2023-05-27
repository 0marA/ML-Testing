import time
import os
import cv2
from pydub import AudioSegment
from pydub.playback import play
import keyboard

IMAGE_PATH = os.path.join('new-data', 'images')
labels = ['scissors', 'index-card']
number_imgs = 20

sound = AudioSegment.from_mp3("shutter.mp3")

cap = cv2.VideoCapture(1)
for label in labels:
    print('Collecting images for {}'.format(label))
    
    img_num = 0
    while img_num < number_imgs:
            if keyboard.is_pressed("space"):
                    img_num += 1
                    ret, frame = cap.read()

                    play(sound)
                    imgname = os.path.join(IMAGE_PATH, label +
                                        '-{}'.format(img_num) + '.jpg')
                    cv2.imwrite(imgname, frame)
                    print("Collected {}".format(imgname))
        
# python3 train.py --img 320 --batch 16 --epochs 500 --data dataset.yaml --weights yolov5s.pt
# conda activate torch-gpu            
