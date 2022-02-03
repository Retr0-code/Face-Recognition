import cv2
import os
import re
import sqlite3

# Регулярное выражение для очистки полученного имени из базы данных (для удаления спец. символов)
SpecChars2 = r"[\'\'\(\)\,\][0]"

# подключение к базе данных 
db = sqlite3.connect("faceDataset.db")
cursor = db.cursor()

nameID = 0      # содержит id пользователя 
Count = 0       # инициализация лица 
face_detector = 0
cam = 0

def initDB():
    global db # обращение к глобальной переменной 
    global cursor # обращение к глобальной переменной

    cursor.execute("CREATE TABLE IF NOT EXISTS users (name TEXT, id INTEGER)") # создание таблицы при ее отсутствии 
    cursor.execute("SELECT name FROM users WHERE id='1'") # получение имени ползователя с id 1
    
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users VALUES ('Unknown', '1')") # если не существует, то вставляется "неизвестно"

def insert_data():
    global nameID # обращение к глобальной переменной 
    # получение имени пользоваеля 
    Name = input('\n enter user name end press <return> ==>  ') # содержит введенное имя 
    
    print("\n [INFO] Initializing face capture. Look the camera and wait ...") # вывод текста об ожидании
    
    # получение нового id 
    [count] = cursor.execute("SELECT COUNT(name) FROM users")
    nameID = int(re.sub(SpecChars2, "", str(count))) + 1
    
    
    # добавление имени и id в таблицу       
    cursor.execute(f"INSERT INTO users VALUES ('{Name}', '{nameID}')")
    
    # сохранение изменений 
    db.commit()

def initCV2():
    global cam # обращение к глобальной переменной
    global face_detector # обращение к глобальной переменной

    # настройка параметров камеры 
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # установка ширины
    cam.set(4, 480) # установка высоты
    
    # подключение нод
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# проверка на запуск, а не подключение 
if __name__ == "__main__":
    initDB()
    initCV2()
    insert_data()

    while(True):
    
        ret, img = cam.read()   # получение не обработанного изображения с камеры 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # перевод изображения в монохром 
        faces = face_detector.detectMultiScale(gray, 1.3, 5)    # получение лица с изображения 
    
        for (x,y,w,h) in faces:
    
            # создание рамки вокруг лица 
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            Count += 1
    
            # сохранение симков в папку "dataset" (User.id.порядок,jpg)
            cv2.imwrite("dataset/User." + str(nameID) + '.' + str(Count) + ".jpg", gray[y:y+h,x:x+w])
    
            # раскомментируйте, чтобы увидеть поток с камеры 
            # cv2.imshow('image', img) # показ изображения с камеры 
            
    
        k = cv2.waitKey(100) & 0xff # нажмите 'ESC' для выхода из программы
        if k == 27:
            break
        elif Count >= 64: # создание 64 образцов лица 
            break
    
    # закрытие программы
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()  # закрытие потока с камеры
    cv2.destroyAllWindows() # закрытие всех окон