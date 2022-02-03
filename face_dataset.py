import cv2
import os
import re
import sqlite3

# Регулярное выражение для очистки полученного имени из базы данных (для удаления спец. символов)
SpecChars2 = r"[\'\'\(\)\,\][0]"

# Connect the db
db = sqlite3.connect("faceDataset.db")
cursor = db.cursor()

nameID = 0      # Will store id of user
Count = 0       # Initialize individual sampling face count
face_detector = 0
cam = 0

def initDB():
    global db
    global cursor

    cursor.execute("CREATE TABLE IF NOT EXISTS users (name TEXT, id INTEGER)")
    cursor.execute("SELECT name FROM users WHERE id='1'")
    
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users VALUES ('Unknown', '1')")

def insert_data():
    global nameID
    # For each person, enter one numeric face id
    Name = input('\n enter user name end press <return> ==>  ')
    
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")
    
    # Getting new id
    [count] = cursor.execute("SELECT COUNT(name) FROM users")
    nameID = int(re.sub(SpecChars2, "", str(count))) + 1
    
    
    # Adding Values in table        
    cursor.execute(f"INSERT INTO users VALUES ('{Name}', '{nameID}')")
    
    # Commit changes
    db.commit()

def initCV2():
    global cam
    global face_detector

    # Initialize image capturing
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height
    
    # Connect nodes
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


if __name__ == "__main__":
    initDB()
    initCV2()
    insert_data()

    while(True):
    
        ret, img = cam.read()   # Get raw image from camera
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # Make monochrome image
        faces = face_detector.detectMultiScale(gray, 1.3, 5)    # Find face on image
    
        for (x,y,w,h) in faces:
    
            # Draws rectangle around a face
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            Count += 1
    
            # Save the captured images into the datasets folder
            cv2.imwrite("dataset/User." + str(nameID) + '.' + str(Count) + ".jpg", gray[y:y+h,x:x+w])
    
            # Uncomment enable video stream
            #cv2.imshow('image', img)
    
        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif Count >= 64: # Take 64 face sample and stop video
            break
    
    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()