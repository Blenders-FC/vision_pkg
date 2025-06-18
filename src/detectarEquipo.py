#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np

# Definimos las máscaras de color
lower_red1 = np.array([0, 100, 100], np.uint8) 
upper_red1 = np.array([10, 255, 255], np.uint8) 
lower_red2 = np.array([170, 100, 100], np.uint8) 
upper_red2 = np.array([180, 255, 255], np.uint8) 
Red1 = [lower_red1, upper_red1]
Red2 = [lower_red2, upper_red2]
Rmask = [Red1, Red2]

azulHigh = np.array([150, 220, 255], np.uint8)
azulLow = np.array([90, 0, 120], np.uint8)
Amask = [azulLow, azulHigh]

kernel = np.ones((5, 5), np.uint8)
bridge = CvBridge()

def deteccionEquipo(msg):
    try:
        # Convertir imagen ROS a OpenCV
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as e:
        rospy.logerr(f"Error al convertir la imagen: {e}")
        return

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, Rmask[0], Rmask[1])  # puedes cambiar a Amask para azul
    mask_clean = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask_vis = cv.bitwise_and(frame, frame, mask=mask_clean)

    contours, _ = cv.findContours(mask_clean, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

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

    cv.imshow("Original", frame)
    cv.imshow("Mask", mask)
    cv.imshow("Mask Visualizada", mask_vis)

    if cv.waitKey(1) == 27:  # ESC para salir
        rospy.signal_shutdown("Usuario cerró la ventana")

def main():
    rospy.init_node('deteccion_equipo_node', anonymous=True)
    rospy.Subscriber('/usb_cam/image_raw', Image, deteccionEquipo)
    rospy.loginfo("Nodo iniciado y escuchando imágenes...")
    rospy.spin()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
