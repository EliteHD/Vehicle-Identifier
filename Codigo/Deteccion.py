import torch
import cv2
import numpy as np

#leemos el modelo                                       Colocar aqui el PATH de su ubicacion en la computadora
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Cars/model/RedNeuronal.pt',device='cpu',force_reload=True)

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()

    #Detecciones
    detect = model(frame)

    #FPS
    cv2.imshow('Detector de Carros',np.squeeze(detect.render()))

    #Leer teclado
    t = cv2.waitKey(1)
    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()

