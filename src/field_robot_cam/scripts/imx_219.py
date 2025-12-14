#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def gstreamer_pipeline(
        sensor_id=0,
        capture_width=3280,
        capture_height=2464,
        display_width=640,
        display_height=480,
        framerate=30,
        flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=%d, height=%d, "
        "format=NV12, framerate=%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=%d, height=%d, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink"
        % (sensor_id, capture_width, capture_height, framerate,
           flip_method, display_width, display_height)
    )

def main():
    rospy.init_node('imx219_publisher', anonymous=True)
    pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)

    bridge = CvBridge()

    pipeline = gstreamer_pipeline()
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        rospy.logerr("❌ Không mở được camera IMX219!")
        return

    rate = rospy.Rate(30)   # 30 FPS

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logwarn("⚠️ Không đọc được khung hình")
            continue

        msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        pub.publish(msg)

        rate.sleep()

    cap.release()

if __name__ == '__main__':
    main()

