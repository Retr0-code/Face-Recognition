import cv2
import os
import re
import sqlite3


pattern = "****************"

SpecChars = r"[\'\(\)\,\][0-9]"
SpecChars2 = r"[\'\'\(\)\,\][0]"

#connecting to the db
db = sqlite3.connect("faceDataset.db")
cursor = db.cursor()


cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height



face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# For each person, enter one numeric face id
Name = input('\n enter user name end press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
Count = 0

#getting new id
[count] = cursor.execute("SELECT COUNT(name) FROM users")
formatedCount = re.sub(SpecChars2, "", str(count))
nameID = int(formatedCount) + 1



#formating
if len(Name) != len(pattern):

    LengthCount = len(pattern) - len(Name)

    for i in range(LengthCount):
        Name += " "

#adding Values in table        
cursor.execute(f"INSERT INTO users VALUES ('{Name}', '{nameID}')")

db.commit()


while(True):

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        Count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(nameID) + '.' + str(Count) + ".jpg", gray[y:y+h,x:x+w])

        #cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif Count >= 64: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
