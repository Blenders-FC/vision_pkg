#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

kernel = np.ones((5, 5), np.uint8)

def imageCallback(img_msg):
    try:
        frame = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")
        return

    frame_copy = frame.copy()
    gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)

    lab_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2HSV)
    lower_green = np.array([0, 40, 0])  # Lower bound for green
    upper_green = np.array([90, 255, 255])  # Upper bound for green
    mask_green = cv2.inRange(lab_frame, lower_green, upper_green)
    opening = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=1)
    # Sure background
    sure_bg = cv2.erode(opening, kernel, iterations=2)
    sure_bg = cv2.dilate(sure_bg, kernel, iterations=2)

    new_mask = cv2.bitwise_not(sure_bg)

    img = cv2.bitwise_and(frame_copy, frame_copy, mask=new_mask)
    print(img.shape)
    #img = sure_bg
    final_img = bridge.cv2_to_imgmsg(img,"rgb8")
    pub_img.publish(final_img)

if __name__ == "__main__":
    bridge = CvBridge()
    rospy.init_node('mask_image')
    robot_id = rospy.get_param('robot_id', 1)
    pub_img = rospy.Publisher(f'robotis_{robot_id}/ImgMask', Image, queue_size=1)
    rospy.loginfo("Hello ROS")
    subimg = rospy.Subscriber("/usb_cam/image_raw", Image, imageCallback)

    rospy.spin()
