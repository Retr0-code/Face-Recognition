import os
import cv2
import numpy as np
from PIL import Image

path = 'dataset'    # Содержит путь к изображениям лица

# Подключение нод (каскадной таблицы)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

# function to get the images and label data Функция получения изображений
def getImagesAndLabels(path):
    global detector     # Обращение к глобальной переменной "detector"
    
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]   # Получение всех путей в папке с изображениями
    faceSamples = []    # Содержит экземпляры (фото) для тренировки 
    ids = []        # Содержит id тренеруемых лиц (есть возможность тренероваться на нескольких лицах одновременно)

    for imagePath in imagePaths:    # Для каждого изображения в папке ... 

        PIL_img = Image.open(imagePath).convert('L')    # Перевод изображения в монохром
        img_numpy = np.array(PIL_img, 'uint8')       # Массив с изображениями

        id = int(os.path.split(imagePath)[-1].split(".")[1])    # Получение id лица пользователя
        faces = detector.detectMultiScale(img_numpy)    # Распознование лица по обработанным образцам

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])  # Добваление образца в массив
            ids.append(id)      # Довление id в массив

    return faceSamples, ids

if __name__ == "__main__":
    recognizer = cv2.face.LBPHFaceRecognizer_create()   # Подключение модуля распознования
    
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)        # Получение списка лиц и списка id
    recognizer.train(faces, np.array(ids))      # Запуск тренеровки
    
    # Сохранение модели в "trainer/trainer.yml"
    recognizer.write('trainer/trainer.yml') # Аналог recognizer.save()
    
    # Вывод кол-ва лиц распознаных на образцах
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))