#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class FaceDetector:
    def __init__(self):
        rospy.init_node('face_detector', anonymous=True)
        
        # Load the cascade
        self.face_cascade = cv2.CascadeClassifier(rospy.get_param('~haar_path', 'data/haarcascade_frontalface_default.xml'))
        
        # For converting between ROS and OpenCV images
        self.bridge = CvBridge()
        
        # Publisher for the processed image
        self.image_pub = rospy.Publisher("~output_image", Image, queue_size=1)
        
        # Subscriber to camera feed
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)
        
        rospy.loginfo("Face detector node initialized")

    def image_callback(self, data):
        try:
            # Convert ROS image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(cv_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Convert back to ROS image and publish
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    try:
        detector = FaceDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
