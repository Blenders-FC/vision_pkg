#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np

#variables globales
frame = None
mask_red_clean = None
mask_blue_clean = None
mask_red = None
obstacle = False

# Para rojo (dos rangos porque el rojo está en ambos extremos del espacio HSV)
lower_red1 = np.array([0, 100, 100], np.uint8) 
upper_red1 = np.array([10, 255, 255], np.uint8) 
lower_red2 = np.array([170, 100, 100], np.uint8) 
upper_red2 = np.array([180, 255, 255], np.uint8) 

# Para azul
azul_low = np.array([100, 100, 100], np.uint8)
azul_high = np.array([140, 255, 255], np.uint8)

kernel = np.ones((5, 5), np.uint8)
bridge = CvBridge()

#------------------------------------------------------------ para recibir la imagen
def image_callback(msg):
    global frame
    try:
        # Convertir imagen ROS a OpenCV
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as e:
        rospy.logerr(f"Error al convertir la imagen: {e}")
        return

#------------------------------------------------------------ para detectar rojo y azul
def deteccionEquipo(subsection):
    global obstacle, mask_red_clean, mask_blue_clean, mask_red

    hsv = cv.cvtColor(subsection, cv.COLOR_BGR2HSV)
    
    # Procesamiento para color rojo (combinando ambos rangos)
    mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv.bitwise_or(mask_red1, mask_red2)
    mask_red_clean = cv.morphologyEx(mask_red, cv.MORPH_OPEN, kernel)
    
    # Procesamiento para color azul
    mask_blue = cv.inRange(hsv, azul_low, azul_high)
    mask_blue_clean = cv.morphologyEx(mask_blue, cv.MORPH_OPEN, kernel)
    
    obstacle = np.any(mask_red_clean) or np.any(mask_blue_clean)
    return obstacle

#------------------------------------------------------------ para mostrar las máscaras detectadas
def display_imagenes():
    mask_red_vis = cv.bitwise_and(frame, frame, mask=mask_red_clean)
    mask_blue_vis = cv.bitwise_and(frame, frame, mask=mask_blue_clean)
    combined_vis = cv.addWeighted(mask_red_vis, 1.0, mask_blue_vis, 1.0, 0)
    # Combinar ambas máscaras para visualización
    combined_mask = cv.bitwise_or(mask_red, mask_blue)
    
    
    contours, _ = cv.findContours(combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rects = []
    centers = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 100:
            x, y, w, h = cv.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
            rects.append((x, y, w, h))
            centers.append((cx, cy))

    groups = []
    distance_threshold = 50

    for (cx, cy) in centers:
        added = False
        for group in groups:
            for (gcx, gcy) in group:
                distance = np.sqrt((cx - gcx)**2 + (cy - gcy)**2)
                if distance < distance_threshold:
                    group.append((cx, cy))
                    added = True
                    break
            if added:
                break
        if not added:
            groups.append([(cx, cy)])

    for group in groups:
        group_indices = [centers.index(c) for c in group]
        group_rects = [rects[i] for i in group_indices]

        x_min = min([x for (x, y, w, h) in group_rects])
        y_min = min([y for (x, y, w, h) in group_rects])
        x_max = max([x + w for (x, y, w, h) in group_rects])
        y_max = max([y + h for (x, y, w, h) in group_rects])

        cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 3)
        cv.circle(frame, ((x_min + x_max) // 2, (y_min + y_max) // 2), 5, (255, 0, 0), -1)

    # Mostrar todas las visualizaciones
    cv.imshow("Original", frame)
    cv.imshow("Máscara Roja", mask_red_clean)
    cv.imshow("Máscara Azul", mask_blue_clean)
    cv.imshow("Máscara Combinada", combined_mask)
    cv.imshow("Visualización Combinada", combined_vis)

    if cv.waitKey(1) == 27:  # ESC para salir
        rospy.signal_shutdown("Usuario cerró la ventana")

#------------------------------------------------------------ para la navegación
def navigation ():
    global frame 

    if frame is None:
        return 

    height, width = frame.shape[:2] #regresa un arreglo de altura y ancho de imágen
    lower_half = frame [height // 2:, :] #para buscar sólo en la parte de abajo de la imagen, es decir por donde podría avanzar el robot, usando slicing
    
    xi, xf = 0, width
    yi, yf = 0, lower_half.shape[0]
    sections = 0
    max_sections = 12 #va a buscar en 12 zonas worst case scenario 
    
    while sections < max_sections:  #loop para que siga dividiendo y analizando frames hasta llegar a 12 (para que se pueda rendir vaya...) 
        half_x = (xf - xi) // 2 #la mitad del ancho de la imagen
        
        left_section = lower_half[yi:yf, xi:xi + half_x]
        left_obstacle = deteccionEquipo(left_section)

        if not left_obstacle:
            print(f"vía libre en la sección [{xi}, {xi + half_x}]")
            #Aquí hay que llamar al nodo de movimiento de ROS
            return
        right_section = lower_half[yi:yf, int(half_x):xf]
        right_obstacle = deteccionEquipo(right_section)

        if not right_obstacle:
            print (f"Vía libre en la sección [{xi + half_x}, {xf}]")
            #Aquí tmb llamar al nodo de movimiento de ROS
            return
            
        #si hay obstaculos en ambas secciones, analizamos una sección más pequeña
        xf = xi + half_x
        sections += 1
        print("analizando nueva sub zona")

    print("Tras 12 secciones, no se pudo encontrar vía libre!!")

def main():
    rospy.init_node('deteccion_equipo_node', anonymous=True)
    rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)
    rospy.loginfo("Nodo iniciado...")
    rate = rospy.Rate(20) #hay que ver que rate conviene
    while not rospy.is_shutdown():
        if frame is not None:
            display_imagenes()
            navigation()
        rate.sleep()
    
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

