import cv2
import numpy as np
import os
import serial
import sqlite3
import time
import re

#Setup Communication path for arduino (In place of 'COM3' put the port to which your arduino is connected)
#arduino = serial.Serial('COM3', 9600) 
#time.sleep(2)
#print("Connected to arduino...")

db = sqlite3.connect("faceDataset.db")
cursor = db.cursor()


SpecChars = r"[\'\(\)\,\][0-9]"

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

#iniciate id counter
id = 1

while True:

    ret, img =cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        [output] = cursor.execute(f"SELECT * FROM users WHERE id = '{id}'")
        formatedOutput = re.sub(SpecChars, "", str(output))

        
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        formatedOutput, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            #id = formatedOutput
            confidence = "  {0}%".format(round(100 - confidence))
            [outMain] = cursor.execute(f"SELECT * FROM users WHERE id = '{formatedOutput}'")
            MainFormat = re.sub(SpecChars, "", str(outMain))
            print(MainFormat)
            #arduino.write(b'Nikita')
            time.sleep(0.5)

        else:
            #id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            print("Unknown        ")
            #arduino.write(id)
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (0, 255, 0), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera', img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
