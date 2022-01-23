import os
import re
import cv2
import time
import sqlite3
import numpy as np

# Connect to a database
db = sqlite3.connect("faceDataset.db")
cursor = db.cursor()


SpecChars = r"[\'\(\)\,\][0-9]"

# Initialize required modules
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')  # Get model
cascadePath = "haarcascade_frontalface_default.xml" # Connect nodes
faceCascade = cv2.CascadeClassifier(cascadePath)


font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

# Iniciate id counter
id = 1

while True:

    ret, img = cam.read()   # Read images from camera
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     # Convert image to grayscale
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        # Get name from DB by id
        [output] = cursor.execute(f"SELECT * FROM users WHERE id = '{id}'")
        formatedOutput = re.sub(SpecChars, "", str(output))
        
        # Show rectangle around a face
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        # Get result of recognizing in percentage ratio
        formatedOutput, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            confidence = "  {0}%".format(round(100 - confidence))   # Display percentage ratio
            [outMain] = cursor.execute(f"SELECT * FROM users WHERE id = '{formatedOutput}'")    # Get name
            MainFormat = re.sub(SpecChars, "", str(outMain))
            print(MainFormat)   # Display name
            time.sleep(0.1)

        # If face is not in DB 'Unknown' will be displayed
        else:
            confidence = "  {0}%".format(round(100 - confidence))
            print("Unknown")
            time.sleep(0.1)
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (0, 255, 0), 2)   # Display id on rectangle
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1) # Display confidence level under the rectangle
    
    # Shows image from camera
    cv2.imshow('camera', img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
