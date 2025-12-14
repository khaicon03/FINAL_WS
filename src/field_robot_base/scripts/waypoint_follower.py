#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
from std_msgs.msg import Int32, Float32
from nav_msgs.msg import Odometry

class EncoderOdomNode(object):
    def __init__(self):
        # Tham số (bạn chỉnh trong launch cho đúng xe)
        self.wheel_radius   = rospy.get_param("~wheel_radius", 0.0325)       # m
        self.ticks_per_rev  = rospy.get_param("~ticks_per_rev", 333*4)     # CPR * 4
        self.encoder_sign   = rospy.get_param("~encoder_sign", 1.0)        # +1 đi tới vx dương
        self.odom_frame     = rospy.get_param("~odom_frame", "odom")
        self.base_frame     = rospy.get_param("~base_frame", "base_link")

        self.last_ticks = None
        self.last_time  = None
        self.wheel_speed = 0.0

        self.pub_speed = rospy.Publisher("speed", Float32, queue_size=10)
        self.pub_odom  = rospy.Publisher("odom",  Odometry, queue_size=10)

        rospy.Subscriber("encoder_ticks", Int32, self.encoder_cb, queue_size=50)

        rospy.loginfo("encoder_odom_node started")

    def encoder_cb(self, msg):
        now = rospy.Time.now()

        if self.last_ticks is None:
            # lần đầu: chỉ lưu lại để so sánh lần sau
            self.last_ticks = msg.data
            self.last_time  = now
            return

        dt = (now - self.last_time).to_sec()
        if dt <= 0.0:
            return

        tick_diff = msg.data - self.last_ticks
        self.last_ticks = msg.data
        self.last_time  = now

        # ticks -> vòng -> quãng đường
        rev = float(tick_diff) / float(self.ticks_per_rev)
        s   = 2.0 * math.pi * self.wheel_radius * rev   # mét
        vx  = self.encoder_sign * (s / dt)              # m/s

        self.wheel_speed = vx

        # ==== publish wheel_speed (debug) ====
        spd_msg = Float32()
        spd_msg.data = vx
        self.pub_speed.publish(spd_msg)

        # ==== publish /wheel_odom (twist only, pose = 0, cov lớn) ====
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id  = self.base_frame

        # pose = 0, không dùng, nên covariance rất lớn
        odom.pose.pose.orientation.w = 1.0
        odom.pose.covariance = [0.0]*36
        odom.pose.covariance[0]  = 1e6
        odom.pose.covariance[7]  = 1e6
        odom.pose.covariance[14] = 1e6
        odom.pose.covariance[21] = 1e6
        odom.pose.covariance[28] = 1e6
        odom.pose.covariance[35] = 1e6

        # twist: chỉ vx từ encoder, wz = 0 (EKF lấy wz từ IMU)
        odom.twist.twist.linear.x  = vx
        odom.twist.twist.linear.y  = 0.0
        odom.twist.twist.linear.z  = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = 0.0

        odom.twist.covariance = [0.0]*36
        odom.twist.covariance[0]  = 0.05  # vx
        odom.twist.covariance[35] = 0.05  # wz (dù đang =0, nhưng EKF vẫn cần số)

        self.pub_odom.publish(odom)

def main():
    rospy.init_node("encoder_odom_node")
    node = EncoderOdomNode()
    rospy.spin()

if __name__ == "__main__":
    main()

