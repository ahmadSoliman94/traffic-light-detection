import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO


model = YOLO('./yolo_model.pt')


cap=cv2.VideoCapture('./video.mp4')




while (cap. isOpened()):
    ret, frame= cap.read()
    if ret:
        res=model.predict(frame,conf=0.3)
        res_plotted=res[0].plot()
        cv2. imshow('frame', res_plotted)
        if cv2.waitKey(20)& 0xFF== ord ('q'):
            break
cap. release()
cv2.destroyAllWindows()