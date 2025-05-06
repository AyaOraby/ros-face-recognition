#!/usr/bin/env python
import rospy
import cv2
import os
import glob
import subprocess
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

class ImageFaceDetectorFolder:
    def __init__(self):
        rospy.init_node('image_face_detector', anonymous=True)
        
        # Load the cascade
        self.face_cascade = cv2.CascadeClassifier(rospy.get_param('~haar_path', 'data/haarcascade_frontalface_default.xml'))
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("~output_image", Image, queue_size=1)
        self.name_pub = rospy.Publisher("/facedetected", String, queue_size=10)   # NEW publisher
        
        # Get image folder path
        self.image_folder = rospy.get_param('~image_folder', 'data/')
        self.debug_folder = "/tmp/face_detection_debug/"
        os.makedirs(self.debug_folder, exist_ok=True)
        
        # Verify haar cascade file
        if not os.path.isfile(rospy.get_param('~haar_path')):
            rospy.logerr("Haar cascade file not found")
            return

        # Process all images
        self.process_images()

    def process_images(self):
        """Process all images in the folder"""
        image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.jfif')
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.image_folder, ext)))

        if not image_files:
            rospy.logerr(f"No images found in {self.image_folder}")
            return

        for img_path in image_files:
            rospy.loginfo(f"Processing image: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                rospy.logerr(f"Could not read image file: {img_path}")
                continue

            # Extract name from filename (remove extension)
            base_name = os.path.basename(img_path)
            name_without_ext = os.path.splitext(base_name)[0]

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                rospy.loginfo(f"Face detected: {name_without_ext}")
                self.name_pub.publish(name_without_ext)   # Publish the name to /facedetected

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, name_without_ext, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 1, cv2.LINE_AA)  # smaller font size

            # Save debug image
            debug_img_path = os.path.join(self.debug_folder, base_name)
            cv2.imwrite(debug_img_path, img)
            rospy.loginfo(f"Saved debug image: {debug_img_path}")

            # Publish the image
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
            except Exception as e:
                rospy.logerr(f"Error publishing image: {str(e)}")

            # Open debug image
            self.open_debug_image(debug_img_path)

            # Optional: wait a little to allow image_view to refresh
            rospy.sleep(1.0)

    def open_debug_image(self, img_path):
        """Open a debug image"""
        try:
            subprocess.Popen(['xdg-open', img_path])
            rospy.loginfo(f"Opened debug image: {img_path}")
        except Exception as e:
            rospy.logerr(f"Failed to open debug image: {str(e)}")

if __name__ == '__main__':
    try:
        detector = ImageFaceDetectorFolder()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

