#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

# Initialize global publishers and bridge
pub_img = None
pub_ball = None
pub_goal = None
pub_robot = None
bridge = CvBridge()

# Ball detection treshold
ball_lost_counter = 0
current_ball = Point(999,999,0)
ball_lost_treshold = 5 

# Load YOLOv11 TensorRT model
model_path = "/home/blenders/catkin_ws/src/vision_pkg/models/"
model_name = "rhoban-v6.engine"
trt_model = YOLO(model_path + model_name)

def process_and_publish(frame):
    global ball_lost_treshold, ball_lost_counter, current_ball

    results = trt_model(frame)
    annotated_frame = results[0].plot()
    img_msg = bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
    pub_img.publish(img_msg)

    # Default fallback
    temp_ball = Point(999, 999, 0)
    goal_center = Point(999, 999, 0)
    robot_center = Point(999, 999, 0)

    # Confidence tracking
    best_ball_conf = 0
    best_robot_conf = 0
    best_goal_conf = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)  # bottom Y
            label = r.names[cls_id].lower()

            print("label:", label, "| conf:", conf)

            if "ball" in label and conf > best_ball_conf:
                best_ball_conf = conf
                temp_ball = Point(cx, cy, 0)

            elif "goal" in label and conf > best_goal_conf:
                best_goal_conf = conf
                goal_center.x = cx
                goal_center.y = cy

            elif "robot" in label and conf > best_robot_conf:
                best_robot_conf = conf
                robot_center.x = cx
                robot_center.y = cy

    #Flickering correction
    if temp_ball != Point(999,999,0):
        current_ball = temp_ball
        ball_lost_counter = 0
    elif (ball_lost_counter >= ball_lost_treshold):
            ball_lost_counter += 1
            current_ball = Point(999,999,0)
    else:
        ball_lost_counter += 1
 

    # Publish final positions
    pub_ball.publish(current_ball)
    pub_goal.publish(goal_center)
    pub_robot.publish(robot_center)


def main():
    global pub_img, pub_ball, pub_goal, pub_robot

    rospy.init_node('main_vision_node')
    robot_id = rospy.get_param('robot_id', 1)

    # Publishers
    pub_img = rospy.Publisher(f'/robotis_{robot_id}/ImgFinal', Image, queue_size=1)
    pub_ball = rospy.Publisher(f'/robotis_{robot_id}/ball_center', Point, queue_size=1)
    pub_goal = rospy.Publisher(f'/robotis_{robot_id}/goal_center', Point, queue_size=1)
    pub_robot = rospy.Publisher(f'/robotis_{robot_id}/robot_center', Point, queue_size=1)

    rospy.loginfo("YOLOv11 vision node started")

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        rospy.logerr("Error: Could not open camera.")
        return

    # Main loop
    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logerr("Error: Failed to capture frame.")
            continue

        process_and_publish(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
