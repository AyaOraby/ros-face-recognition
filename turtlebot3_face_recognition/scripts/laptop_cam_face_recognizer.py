#!/usr/bin/env python
import rospy
import cv2
import os
import glob
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

class LaptopCamFaceRecognizer:
    def __init__(self):
        rospy.init_node('laptop_cam_face_recognizer')

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            rospy.logerr("Cannot open laptop camera!")
            exit(1)

        # Load Haarcascade
        haar_path = rospy.get_param('~haar_path', 'data/haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(haar_path)

        # Load known faces
        self.known_faces = []  # list of (name, face_roi)
        self.load_known_faces()

        # Publishers
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/laptop_cam/faces", Image, queue_size=1)
        self.name_pub = rospy.Publisher("/face_detected", String, queue_size=1)

        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def load_known_faces(self):
        folder_path = rospy.get_param('~image_folder', 'data/')
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.jfif')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))

        for img_path in image_files:
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                name = os.path.splitext(os.path.basename(img_path))[0]
                self.known_faces.append((name, cv2.resize(face_roi, (100,100))))  # Resize for comparison

    def recognize_face(self, face_roi):
        face_resized = cv2.resize(face_roi, (100,100))

        for name, known_face in self.known_faces:
            diff = cv2.absdiff(known_face, face_resized)
            score = np.sum(diff) / (100*100)

            if score < 25:  # Threshold to tune
                return name
        return None

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                name = self.recognize_face(face_roi)

                if name:
                    rospy.loginfo(f"Detected: {name}")
                    self.name_pub.publish(name)
                    cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)

            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
            cv2.imshow('Laptop Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            rate.sleep()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        recognizer = LaptopCamFaceRecognizer()
        recognizer.run()
    except rospy.ROSInterruptException:
        pass
