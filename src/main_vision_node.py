#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import math
from yolo_utils import *
from openvino.runtime import Core
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Point
from std_msgs.msg import Float64


def detect(session, image_src, namesfile):
    global center

    IN_IMAGE_H = session.input(0).shape[2]
    IN_IMAGE_W = session.input(0).shape[3]

    # Input
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0

    # Compute   
    output_layer_boxes = session.output("boxes")
    output_layer_confs = session.output("confs")
    outputs_boxes = compiled_model([img_in])[output_layer_boxes]
    outputs_confs = compiled_model([img_in])[output_layer_confs]
    outputs = [outputs_boxes, outputs_confs]
    
    boxes = post_processing(img=img_in, conf_thresh=0.8, nms_thresh=0.6, output=outputs)

    print(boxes)

    if len(boxes) > 0 and len(boxes[0]) > 0 and len(boxes[0][0]) > 0:
        print(boxes[0])
        # Update center coordinates
        center.x = int(boxes[0][0][7] * IN_IMAGE_W)
        center.y = int(boxes[0][0][8] * IN_IMAGE_H)
        center.z = int((boxes[0][0][2]*IN_IMAGE_W - boxes[0][0][0]*IN_IMAGE_W ) * (boxes[0][0][3]*IN_IMAGE_H- boxes[0][0][1]*IN_IMAGE_H))
        print(center)
        pub_center.publish(center)
    else:
        center.x = 999
        center.y = 999
        pub_center.publish(center)

    class_names = load_class_names(namesfile)
    final_img = plot_boxes_cv2_video(image_src, boxes[0], class_names=class_names, color=(255,0,0))

    return final_img, center


def compileModel():
    # Function for compiling model with OpenVino
    global compiled_model

    ie = Core()
    model = ie.read_model(model=find_file(parent_folder="models", filename="yolov4soccer.onnx"))
    compiled_model = ie.compile_model(model=model, device_name="MYRIAD")


def imageCallback(img_msg):
    global center, compiled_model
    print("Llega image callback")
    try:
        frame = bridge.imgmsg_to_cv2(img_msg, "passthrough")
    except CvBridgeError as e:
        rospy.logerr(f"CvBridge Error: {e}")
        return
    
    # Process image with model and publish
    
    namesfile = find_file(parent_folder="models", filename="_classes.txt")
    
    img, center = detect(session=compiled_model, image_src=frame, namesfile=namesfile)
    final_img = bridge.cv2_to_imgmsg(img, "rgb8")
    #Change img_msg to final_img
    pub_img.publish(final_img)
    


if __name__ == "__main__":
    bridge = CvBridge()
    slope = Float64()
    center = Point()

    center.x = 0
    center.y = 0

    compileModel()

    rospy.init_node('vision_node')
    robot_id = rospy.get_param('robot_id', 1)


    pub_img = rospy.Publisher(f'/robotis_{robot_id}/ImgFinal', Image, queue_size=1)
    pub_center = rospy.Publisher(f'/robotis_{robot_id}/ball_center', Point, queue_size=1)

    rospy.loginfo("Hello ROS")

    subimg = rospy.Subscriber("/usb_cam_node/image_raw", Image, imageCallback)

    rospy.spin()
