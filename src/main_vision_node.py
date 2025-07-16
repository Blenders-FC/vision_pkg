#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from blenders_msgs.msg import RobotPose, PointArray
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2

# Global publishers and state
pub_img = None
pub_ball = None
pub_goal = None
pub_robot = None
bridge = CvBridge()
valid_pose_received = False
last_right_goal = Point(999, 999, 0)

def init_pose_callback(msg):
    global valid_pose_received
    valid_pose_received = msg.valid
    rospy.loginfo(f"[Pose] valid: {valid_pose_received}")

def process_and_publish(frame):
    global valid_pose_received, quadrant, last_right_goal

    results = trt_model(frame)
    annotated_frame = results[0].plot()
    img_msg = bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
    pub_img.publish(img_msg)

    ball_center = Point(999, 999, 0)
    robot_center = Point(999, 999, 0)
    goal_candidates = []

    best_ball_conf = 0
    best_robot_conf = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            cx = int((x1 + x2) / 2)
            cy = int(y2)
            label = r.names[cls_id].lower()

            print("label:", label, "| conf:", conf)


            if "ball" in label and conf > best_ball_conf:
                best_ball_conf = conf
                ball_center = Point(cx, cy, 0)

            elif "goal" in label:
                goal_candidates.append((conf, Point(cx, cy, 0)))

            elif "robot" in label and conf > best_robot_conf:
                best_robot_conf = conf
                robot_center = Point(cx, cy, 0)

    # === Select 2 highest confidence goals ===
    left_goal = Point(999, 999, 0)
    right_goal = last_right_goal  # fallback to last known

    goal_candidates = sorted(goal_candidates, key=lambda x: x[0], reverse=True)
    if len(goal_candidates) >= 2:
        g1, g2 = goal_candidates[0][1], goal_candidates[1][1]
        if g1.x < g2.x:
            left_goal, right_goal = g1, g2
        else:
            left_goal, right_goal = g2, g1
        last_right_goal = right_goal
        rospy.loginfo("Two best goals used → left/right updated")
    elif len(goal_candidates) == 1:
        left_goal = goal_candidates[0][1]
        rospy.loginfo("Only one goal seen → using as left, keeping old right")
    else:
        rospy.loginfo("No goals seen → keeping old goal values")

    # === Publish robot and goal center ===
    pub_robot.publish(robot_center)
    goal_msg = PointArray(points=[left_goal, right_goal])
    pub_goal.publish(goal_msg)

    # === Decide what to publish to ball_center ===
    if not valid_pose_received:
        if len(goal_candidates) < 2:
            rospy.loginfo("Pre-init: < 2 goals → sending default to /ball_center")
            pub_ball.publish(Point(999, 999, 0))
        else:
            if quadrant in [1, 3]:
                rospy.loginfo("Pre-init: quadrant 1 or 3 → send LEFT goal as ball_center")
                pub_ball.publish(left_goal)
            elif quadrant in [2, 4]:
                rospy.loginfo("Pre-init: quadrant 2 or 4 → send RIGHT goal as ball_center")
                pub_ball.publish(right_goal)
    else:
        rospy.loginfo("Post-init: sending actual BALL as ball_center")
        pub_ball.publish(ball_center)

# === Model Load ===
model_path = "/home/blenders/catkin_ws/src/vision_pkg/models/"
model_name = "rhoban-v6.engine"
trt_model = YOLO(model_path + model_name)

def main():
    global pub_img, pub_ball, pub_goal, pub_robot, quadrant

    rospy.init_node('main_vision_node')
    robot_id = rospy.get_param('~robot_id', 1)
    quadrant = rospy.get_param('~quadrant', 3)

    # Subscribers
    rospy.Subscriber(f'/robotis_{robot_id}/robot_pose/init_pose', RobotPose, init_pose_callback)

    # Publishers
    pub_img = rospy.Publisher(f'/robotis_{robot_id}/ImgFinal', Image, queue_size=1)
    pub_ball = rospy.Publisher(f'/robotis_{robot_id}/ball_center', Point, queue_size=1)
    pub_goal = rospy.Publisher(f'/robotis_{robot_id}/goal_centers', PointArray, queue_size=1)
    pub_robot = rospy.Publisher(f'/robotis_{robot_id}/robot_center', Point, queue_size=1)

    rospy.loginfo("YOLOv11 vision node started")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        rospy.logerr("Error: Could not open camera.")
        return

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
