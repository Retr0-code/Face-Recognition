import os
import re
import cv2
import time
import sqlite3
import numpy as np

# Подключение к базе данных 
db = sqlite3.connect("faceDataset.db")
cursor = db.cursor()

# Регулярное выражение для очистки полученного имени из базы данных (для удаления спец. символов)
SpecChars = r"[\'\(\)\,\][0-9]"

# Инициализация требуемях модулей
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')  # получение моделей обучения 
cascadePath = "haarcascade_frontalface_default.xml" # путь до каскадной таблицы
faceCascade = cv2.CascadeClassifier(cascadePath) # подключение нода(каскадная таблица)

# шрифт для отображения в интерфейсе
font = cv2.FONT_HERSHEY_SIMPLEX

# настройка параметров камеры 
cam = cv2.VideoCapture(0)
cam.set(3, 640) # установка ширины
cam.set(4, 480) # установка высоты 

# установка минимальной ширины и высоты окна с выводом изображения 
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

# хранение id из модели 
id = 1

while True:

    ret, img = cam.read()   # получение изображения с камеры 
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     # перевод изображения в монохром 
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       ) # получение изображения лица 

    for(x,y,w,h) in faces:
        
        [output] = cursor.execute(f"SELECT * FROM users WHERE id = '{id}'") # получение имени из бд
        formatedOutput = re.sub(SpecChars, "", str(output)) # форматирование полученного имени
        
        # показывает квадрат вокруг лица
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        # получение результата в процентном соотношении 
        formatedOutput, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # проверка процентного соотношения
        if (confidence < 100):
            confidence = "  {0}%".format(round(100 - confidence))   # отображение процентного соотношения 
            [outMain] = cursor.execute(f"SELECT * FROM users WHERE id = '{formatedOutput}'")    # получение имени
            MainFormat = re.sub(SpecChars, "", str(outMain))
            print(MainFormat)   # вывод имени 
            

        # Если лицо не в базе данных, то будет изображено "неизвестно"
        else:
            confidence = "  {0}%".format(round(100 - confidence)) # вывод процентного соотношения 
            print("Unknown") # вывод неизвестно
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (0, 255, 0), 2)   # вывод id на рамке
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1) # вывод процентного соотношения под рамкой 
    
        time.sleep(0.1) # задержка считывания изображения 100 мс            
    # вывод изображения с камеры 
    cv2.imshow('camera', img) 

    k = cv2.waitKey(10) & 0xff # нажмите 'ESC' для выхода из программы
    if k == 27:
        break

# закрытие программы 
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release() # закрытие потока с камеры
cv2.destroyAllWindows() # закрытие всех окон 
