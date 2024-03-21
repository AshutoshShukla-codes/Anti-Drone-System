from ultralytics import YOLO
import cv2
import cvzone
import numpy as np
from gpiozero import AngularServo
import math
from time import sleep
from gpiozero.pins.pigpio import PiGPIOFactory

cap = cv2.VideoCapture(0)
cap.set(3, 640) # id no. 3 is for width
cap.set(4, 480) # id no. 4 is for height

model = YOLO("droneModelNano.pt")

classNames = ["drone"]


def getAdjustment(windowMax, x):
    normalised_adjustment = x / windowMax - 0.5
    adjustment_magnitude = abs(round(normalised_adjustment, 1))

    if normalised_adjustment > 0:
        adjustment_direction = -1
    else:
        adjustment_direction = 1

    return adjustment_magnitude, adjustment_direction


def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


# Initialise servos
pigpio_factory = PiGPIOFactory()

servo1 = AngularServo(18, pin_factory=pigpio_factory)
servo2 = AngularServo(23, pin_factory=pigpio_factory)
servo1_now = 0
servo2_now = 0
servo1.angle = servo1_now
servo2.angle = servo2_now
sleep(2)
print("Initialised servos.")

# Constants
cx = -1  # Have to change sign because this servo rotates in the wrong direction
cy = 1

Kp = 80
Kd = 10

while True:
    success, img = cap.read()
    results = model(img, stream=True) # Stream=True means it wil use generaters and it will be more efficient

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # To create opencv bounding box/rectangle over detected objects
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100))/100 # Getting the confidence value

            # Class Name
            cls = int(box.cls[0])
            if conf > 0.90:
                # Putting the text on rectangle
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1)
                cvzone.cornerRect(img, (x1, y1, w, h))  # To create cvzone bounding box/rectangle over detected objects
                # Calculate AB (pixel error)

                A = (0, 0)
                B = (w//2, h//2)
                window = cap.shape

                # Get adjustment
                xmag, xdir = getAdjustment(window[0], B[1])
                ymag, ydir = getAdjustment(window[1], B[0])

                if xmag != None:

                    # Proportional
                    adj_Kpx = cx * Kp * xdir * xmag
                    adj_Kpy = cy * Kp * ydir * ymag

                    # Derivative
                    xmag_old = xmag
                    ymag_old = ymag

                    adj_Kdx = cx * Kd * xdir * (xmag - xmag_old)
                    adj_Kdy = cy * Kd * ydir * (ymag - ymag_old)

                    # adustment
                    adjustment_x = adj_Kpx + adj_Kdx
                    adjustment_y = adj_Kpy + adj_Kdy
                    # servo
                    servo1_now = servo1_now + adjustment_x
                    servo2_now = servo2_now + adjustment_y

                    # Reset line of sight if instructed to look out of bounds
                    if (servo1_now > 90 or servo1_now < -90):
                        servo1_now = 0
                    if (servo2_now > 90 or servo2_now < -90):
                        servo2_now = 0

                    servo1.angle = servo1_now
                    servo2.angle = servo2_now
                    sleep(0.00001)

                xmag = 0
                xdir = 0
                ymag = 0
                ydir = 0

    cv2.imshow("Image", img)
    cv2.waitKey(1)