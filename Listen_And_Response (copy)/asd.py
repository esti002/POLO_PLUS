import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import responser
import gpt_chatbot

opencv_home = cv2.__file__
folders = opencv_home.split(os.path.sep)[0:-1]

path = folders[0]
for folder in folders[1:]:
    path = path + "/" + folder

face_detector_path = path+"/data/haarcascade_frontalface_default.xml"

print("haar cascade configuration found here: ",face_detector_path)

if os.path.isfile(face_detector_path) != True:
    raise ValueError("Confirm that opencv is installed on your environment! Expected path ",face_detector_path," violated.")

haar_detector = cv2.CascadeClassifier(face_detector_path)

#model structure: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/age.prototxt
#pre-trained weights: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/dex_chalearn_iccv2015.caffemodel
age_model = cv2.dnn.readNetFromCaffe("/home/esti002/Codes/Python/Listen_And_Response/datasets/age.prototxt", "/home/esti002/Codes/Python/Listen_And_Response/datasets/dex_chalearn_iccv2015.caffemodel")

#model structure: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.prototxt
#pre-trained weights: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/gender.caffemodel
gender_model = cv2.dnn.readNetFromCaffe("/home/esti002/Codes/Python/Listen_And_Response/datasets/gender.prototxt", "/home/esti002/Codes/Python/Listen_And_Response/datasets/gender.caffemodel")

output_indexes = np.array([i for i in range(0, 101)])

def deneme(cinsiyet,yas):
    print("+++++++++++++++++++++++++++++++++++"+cinsiyet + str(yas))
    time.sleep(3)
    

def analysis():
    # Kamera yakalama başlatılıyor
    cap = cv2.VideoCapture(0)

    # Yüz tespiti için kullanılacak sınıflandırıcıyı yükleme
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Kameradan bir frame yakalama
        ret, frame = cap.read()
        
        # Frame'i gri tona dönüştürme
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Yüz tespiti
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            detected_face = frame[y:y+h, x:x+w]
            
            # Yaş modeli için görüntüyü yeniden boyutlandırma
            detected_face = cv2.resize(detected_face, (224, 224))
            img_blob = cv2.dnn.blobFromImage(detected_face)  # Caffe modeli (1, 3, 224, 224) şeklinde giriş bekler
            
            # ---------------------------
            # Yaş tahmini
            age_model.setInput(img_blob)
            age_dist = age_model.forward()[0]
            apparent_predictions = round(np.sum(age_dist * output_indexes), 2)
            
            # ---------------------------
            # Cinsiyet tahmini
            gender_model.setInput(img_blob)
            gender_class = gender_model.forward()[0]
            gender = 'Kadın' if np.argmax(gender_class) == 0 else 'Erkek'
            
            # ---------------------------
            # Tespit edilen yüzü gösterme
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            print(gender)
            print(apparent_predictions)
            
            
        # Görüntüyü ekranda gösterme
        cv2.imshow('Face Analysis', frame)
        
        # Çıkış için 'q' tuşuna basma kontrolü
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kamera yakalamayı serbest bırakma
    cap.release()
    cv2.destroyAllWindows()

analysis()
