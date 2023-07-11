"""
Se ha usado de base el codigo de clase de la semana 13
"""
import cv2
import pickle
import numpy as np
import time

duracion_deseada = 70  # Duración deseada en segundos
estacionamientos = [] #guardar espacios de almacenamiento

with open('espacios.pkl', 'rb') as file:
    estacionamientos = pickle.load(file) #leer los espacios guardados en espaciospkl

video = cv2.VideoCapture("video.mp4") #leer el video

inicio = time.time()  # Tiempo de inicio
while True:
    check, img = video.read()
    img = cv2.resize(img, (700, 700)) # redimensionar la imagen para ver mejor
    #filtros
    imgBN = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgTH = cv2.adaptiveThreshold(imgBN, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgTH, 5)
    kernel = np.ones((5,5), np.int8)
    imgDil = cv2.dilate(imgMedian, kernel)

    numero_de_fichas_movidas = 0
    for x, y, w, h in estacionamientos:
        espacio = imgDil[y:y+h, x:x+w]
        count = cv2.countNonZero(espacio)
        cv2.putText(img, str(count), (x,y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        if count < 1000:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            numero_de_fichas_movidas += 1
    cv2.imshow('video', img) #mostrar la imagen
    # cv2.imshow('video TH', imgTH)
    # cv2.imshow('video Median', imgMedian)
    # cv2.imshow('video Dilatada', imgDil)
    cv2.waitKey(10)
    tiempo_actual = time.time()  # Tiempo actual
    tiempo_transcurrido = tiempo_actual - inicio  # tiempo transcurrido
    if tiempo_transcurrido >= duracion_deseada:
        break  # terminar si el tiempo se completo

print("El numero de fichas movidas es: ", numero_de_fichas_movidas - 36)# 26 es el numro de fichas del table

