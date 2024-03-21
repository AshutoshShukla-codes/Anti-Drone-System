from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280) # id no. 3 is for width
cap.set(4, 720) # id no. 4 is for height

model = YOLO("droneModelNano.pt")

classNames = ["drone"]

while True:
    success, img = cap.read()
    results = model(img, stream=True) # Stream=True means it wil use generaters and it will be more efficient

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # To create opencv bounding box/rectangle over detected objects
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #print(x1, y1, x2, y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # To create cvzone bounding box/rectangle over detected objects
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100))/100 # Getting the confidence value

            # Class Name
            cls = int(box.cls[0])
            if conf > 0.90:
                # Putting the text on rectangle
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)
                cvzone.cornerRect(img, (x1, y1, w, h))  # To c reate cvzone bounding box/rectangle over detected objects

    cv2.imshow("Image", img)
    cv2.waitKey(1)