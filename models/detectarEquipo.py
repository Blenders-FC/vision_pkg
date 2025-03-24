#importamos la biblioteca de visión de open cv 
import cv2 as cv
#importamos la biblioteca de numpy para operaciones con matrices
import numpy as np

#Creo un objeto videoCapture para abrir el video del equpo rojo
Rcap = cv.VideoCapture('redMask.mp4')
#Creo un objeto videoCapture para abrir el video del equpo azul
Acap = cv.VideoCapture('blueMask.mp4')

#Definimos los parametros de la máscara roja
rojoHigh=np.array([180,255,250],np.uint8)
rojoLow=np.array([80,170,65],np.uint8)
Rmask=[rojoLow,rojoHigh]

#Definimos los parametros de la máscara azul
azulHigh=np.array([150,220,255],np.uint8)
azulLow=np.array([90,0,120],np.uint8)
Amask=[azulLow,azulHigh]

#Creo el kernel para el traking
kernel = np.ones((5,5),np.uint8)#matriz de 5 por 5 con vals de 8 bits



def deteccionEquipo(src,mask):#Defino una función para hacer la detección del equipo (#video,#máscara)

#Creo un loop para leer fotograma por fotograma del video
    while True:
        t,frame = src.read()#Captura frame por frame

        # Excepción de error de lectura
        if not t:
            print("No puede leerse el frame")
            break
        
        op3=cv.cvtColor(frame,cv.COLOR_BGR2HSV)#cambiamos el frame a una escala HSV

        #defino la máscara
        op3mask=cv.inRange(op3,mask[0],mask[1])

        #limpiamos el ruido de la mascara
        op3maskCl=cv.morphologyEx(op3mask, cv.MORPH_OPEN,kernel)

        #visualizamos la máscara
        op3maskVis = cv.bitwise_and(frame,frame,mask=op3maskCl)

        

        #cuadro de detección
        x,y,w,h= cv.boundingRect(op3maskCl)# Obtenemos las coordenadas de los límites de la máscara
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)#creamos el rectangulo de deteción
        cv.circle(frame, ((x + w // 2), (y + h // 2)), 5, (255, 0, 0), -1)#creamos el circulo del centro del rectángulo


        #Visualizamos cada frame...
        cv.imshow('Original',frame)
        cv.imshow('mask',op3mask)
        cv.imshow('maskVis',op3maskVis)


        #Salir del ciclo al presionar espacio
        if cv.waitKey(30)==32:
            break


    # Cierra el video y las ventanas adicionales
    src.release()
    cv.destroyAllWindows()

deteccionEquipo(Rcap,Rmask)#Detección del equipo rojo
#deteccionEquipo(Acap,Amask)#Detección del equipo azul