#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class LaptopCamFaceDetector:
    def __init__(self):
        rospy.init_node('laptop_face_detector')
        
        # Initialize OpenCV camera capture
        self.cap = cv2.VideoCapture(0)  # 0 for default laptop camera
        if not self.cap.isOpened():
            rospy.logerr("Cannot open laptop camera!")
            exit(1)
            
        # Load face detector
        self.face_cascade = cv2.CascadeClassifier(
            rospy.get_param('~haar_path',
            '/opt/ros/noetic/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'))
        
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/laptop_cam/faces", Image, queue_size=1)
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def run(self):
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("Camera frame read failed")
                continue
                
            # Face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Draw detections
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            
            # Publish and display locally
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
            cv2.imshow('Laptop Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            rate.sleep()
            
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        detector = LaptopCamFaceDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass
