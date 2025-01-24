from ultralytics import YOLO
from playsound import playsound
import cv2
import time

model = YOLO('best.engine', task = 'detect', verbose = False)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True :

    ret, frame = cap.read()

    if not ret :
        continue

    results = model.predict(frame, verbose = False, imgsz = 1280)

    cv2.imshow('switch', frame)

    cv2.waitKey(1)

    boxes = results[0].boxes

    size = boxes.cls.size(0)

    for i in range(0, size) :
        if (int)(boxes.cls[i]) == 0 and boxes.conf[i] > 0.8:
            x = (boxes.xyxyn[i][0] + boxes.xyxyn[i][2]) / 2
            y = (boxes.xyxyn[i][1] + boxes.xyxyn[i][3]) / 2
            x = (int)(1 + round((x.item() - 0.5) * 2))
            y = (int)(1 + round((y.item() - 0.5) * 2))
            print(boxes.conf[i].item())
            cv2.imwrite('capture/' + str((int)(time.time())) + '-' + str(int(boxes.conf[i].item() * 100)) + '.jpg', frame)
            ret, frame = cap.read()
            playsound('sound/' + str(y * 3 + x) + '.mp3')
            #time.sleep(2.5)
            break         