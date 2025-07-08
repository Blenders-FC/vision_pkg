#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import UInt8  # Para el estado simple
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np

#------------------------------------------------------------ posibles estados
NO_DETECTA = 0
DERECHA = 1
IZQUIERDA = 2
CENTRO = 3

#------------------------------------------------------------ para recibir la imagen
def image_callback(msg):
    global frame
    try:
        # Convertir imagen ROS a OpenCV
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as e:
        rospy.logerr(f"Error al convertir la imagen: {e}")
        return
    
    #balance de blancos
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    l = cv.equalizeHist(l)
    lab = cv.merge((l, a, b))
    frame = cv.cvtColor(lab, cv.COLOR_LAB2BGR)
    
    if frame is not None:
            rospy.loginfo("Procesando frame...")
            navigation()
            display_imagenes()
    else:
        rospy.logwarn("Aún no hay frame disponible.")

#------------------------------------------------------------ para detectar rojo y azul
def deteccionEquipo(subsection):
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
    return obstacle, mask_red_clean, mask_blue_clean, mask_red, mask_blue

#------------------------------------------------------------ para mostrar las máscaras detectadas
def display_imagenes():
    _, mask_red_clean, mask_blue_clean, mask_red, mask_blue=deteccionEquipo(frame)
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

    estado_texto = ["No detecta", "Derecha", "Izquierda", "Centro"][estado]
    cv.putText(frame, f"Estado: {estado_texto}", (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Mostrar todas las visualizaciones
    cv.imshow("Original", frame)
    #cv.imshow("Máscara Roja", mask_red_clean)
    #cv.imshow("Máscara Azul", mask_blue_clean)
    #cv.imshow("Máscara Combinada", combined_mask)
    #cv.imshow("Visualización Combinada", combined_vis)
    final_img = bridge.cv2_to_imgmsg(frame, "bgr8")
    pub_img.publish(final_img)

    if cv.waitKey(1) == 27:  # ESC para salir
        rospy.signal_shutdown("Usuario cerró la ventana")

#------------------------------------------------------------ para la navegación
def navigation ():
    global frame 

    #hay que agregar una verificación por si no hay frame 
    if frame is None:
        rospy.logwarn("Esperando frame...")
        return

    height, width = frame.shape[:2]
    lower_half = frame[height // 2:, :width]
    xi, xf = 0, lower_half.shape[1]
    yi, yf = 0, lower_half.shape[0]

    max_sections = 20
    sections = 3

    print("Iniciando navegación en zona inferior")

    while sections <= max_sections:
        div_x = (xf - xi) // sections

        # Buscar primero desde el centro hacia la derecha
        for offset in range(sections // 2, sections):
            start = offset * div_x
            end = min((offset + 1) * div_x, xf)
            section = lower_half[yi:yf, start:end]
            obstacle, *_ = deteccionEquipo(section)
            if not obstacle:
                print(f"Vía libre en subzona [{start}, {end}] ({sections} divisiones)")
                cv.rectangle(lower_half, (start, 0), (end, yf), (0, 255, 0), 2)
                cv.circle(lower_half, (start, 0)//2, (end, yf)//2, (0, 255, 0), -1)
                if offset == (sections // 2)+1:
                    estado=CENTRO
                elif offset > (sections // 2)+1:
                    estado=DERECHA
                elif offset < (sections // 2)+1:
                    estado=IZQUIERDA
                break

        # Luego desde el centro hacia la izquierda
        for offset in reversed(range(0, sections // 2)):
            start = offset * div_x
            end = min((offset + 1) * div_x, xf)
            section = lower_half[yi:yf, start:end]
            obstacle, *_ = deteccionEquipo(section)
            if not obstacle:
                print(f"Vía libre en subzona [{start}, {end}] ({sections} divisiones)")
                cv.rectangle(lower_half, (start, 0), (end, yf), (0, 255, 0), 2)
                cv.circle(lower_half, (start, 0)//2, (end, yf)//2, (0, 255, 0), -1)
                if offset == (sections // 2)+1:
                    estado=CENTRO
                elif offset > (sections // 2)+1:
                    estado=DERECHA
                elif offset < (sections // 2)+1:
                    estado=IZQUIERDA
                break
        if estado!=NO_DETECTA:
            break
        print(f"Obstáculos en {sections} zonas, refinando...")
        sections += 2
        print("No se encontró vía libre tras múltiples divisiones")
    # Publicar el estado
    pub_state.publish(estado)
    return estado


if __name__ == '__main__':
    #variables iniciales
    estado = 0
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
    frame = None
    
    rospy.init_node('deteccion_equipo_node', anonymous=True)
    robot_id = rospy.get_param('robot_id', 1)
    pub_img = rospy.Publisher(f'/robotis_{robot_id}/ImgNavChallenge', Image, queue_size=1)
    pub_state = rospy.Publisher(f'/robotis_{robot_id}/via_libre_state', UInt8, queue_size=1, latch=True)
    pub_free_path = rospy.Publisher(f'/robotis_{robot_id}/free_path', Point, queue_size=1)
    
    rospy.loginfo("Nodo iniciado...")
    
    subimg = rospy.Subscriber('/usb_cam/image_raw', Image, image_callback)
    
    rospy.spin()
    cv.destroyAllWindows()

