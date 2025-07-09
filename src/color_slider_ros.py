from __future__ import print_function
import rospy
from std_msgs.msg import String
import cv2 as cv
import argparse #para procesar argumentos de la linea de comandos

window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H' #low hue
low_S_name = 'Low S' #low saturation
low_V_name = 'Low V' #low value
high_H_name = 'High H' #high hue
high_S_name = 'High S' #high saturation
high_V_name = 'High V' #high value

#definición de funciones de callback
def on_low_H_thresh_trackbar(val):
    global low_H 
    global high_H
    low_H = val 
    low_H = min(high_H-1, low_H) #evita que el mínimo supere al máximo
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)
def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H+1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)
def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S-1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)
def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S+1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)
def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V-1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)
def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V+1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)

# Valores máximos para los sliders HSV
max_value = 255
max_value_H = 180

# Inicialización de los valores HSV
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value

#parseo de argumentos
parser = argparse.ArgumentParser(description='Este código usa umbrales para detectar colores en especifico por medio de la cámara.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()

#captura de video
cap = cv.VideoCapture(args.camera)

#ventanas para mostrar los resultados
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)

#sliders de umbrales para ajustar el filtro HSV en tiempo real
cv.createTrackbar(low_H_name, window_detection_name , low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name , high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name , low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name , high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name , low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name , high_V, max_value, on_high_V_thresh_trackbar)

#frecuencia de procesamiento en Hz
rate = rospy.Rate(10)

#main loop
rospy.init_node('') #checar el nombre del nodo!!
while not rospy.is_shutdown():
    ret, frame = cap.read() #captura un frame de la cámara
    if not ret:
        rospy.logwarn("no se pudo leer la cámara")
        continue

    #RGB a HSV
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    #se le aplican los umbrales definidos
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))
    
    #video original vs resultado usando el filtro
    cv.imshow(window_capture_name, frame)
    cv.imshow(window_detection_name, frame_threshold)
    key = cv.waitKey(30)