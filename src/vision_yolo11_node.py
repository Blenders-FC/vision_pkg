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

# Load YOLOv11 TensorRT model
trt_model = YOLO("best-nv2-62.engine")

def process_and_publish(frame):
    # Run inference
    results = trt_model(frame)

    # Get annotated image for visualization
    annotated_frame = results[0].plot()

    # Publish annotated image
    img_msg = bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
    pub_img.publish(img_msg)

    # Initialize centers
    ball_center = Point(999, 999, 0)
    goal_center = Point(999, 999, 0)
    robot_center = Point(999, 999, 0)

    # Process detections
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            xyxy = box.xyxy[0].cpu().numpy()
            cx = int((xyxy[0] + xyxy[2]) / 2)
            cy = int((xyxy[1] + xyxy[3]) / 2)

            label = r.names[cls_id].lower()

            print("label: ", label)

            # TODO: Check label names when migrating model

            if "ball" in label:
                ball_center.x = cx
                ball_center.y = cy

            elif "goal" in label:
                goal_center.x = cx
                goal_center.y = cy

            elif "robot" in label:
                robot_center.x = cx
                robot_center.y = cy

    # Publish centers
    pub_ball.publish(ball_center)
    pub_goal.publish(goal_center)
    pub_robot.publish(robot_center)

def main():
    global pub_img, pub_ball, pub_goal, pub_robot

    rospy.init_node('vision_yolo11_node')
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
