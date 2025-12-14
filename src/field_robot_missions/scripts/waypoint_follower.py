#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import math
from std_msgs.msg import Int32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion
import tf


class EncoderImuOdomNode(object):
    def __init__(self):
        # ===== Tham số (chỉnh trong launch cho đúng xe) =====
        self.wheel_radius  = rospy.get_param("~wheel_radius", 0.0325)   # m
        self.ticks_per_rev = rospy.get_param("~ticks_per_rev", 333*4)   # CPR * 4 nếu dùng X4
        self.encoder_sign  = rospy.get_param("~encoder_sign", -1.0)      # +1 đi tới tick tăng, -1 nếu ngược
        self.yaw_sign      = rospy.get_param("~yaw_sign", 1.0)          # +1 giữ nguyên gyro.z, -1 nếu ngược
        self.odom_frame    = rospy.get_param("~odom_frame", "odom")
        self.base_frame    = rospy.get_param("~base_frame", "base_link")
        self.encoder_topic = rospy.get_param("~encoder_topic", "encoder_ticks")
        self.imu_topic     = rospy.get_param("~imu_topic", "imu/data")
        self.publish_tf    = rospy.get_param("~publish_tf", True)

        # ===== Trạng thái encoder & IMU =====
        self.last_ticks    = None
        self.last_enc_time = None

        self.last_imu_time = None

        # Trạng thái pose
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0        # rad

        # Vận tốc hiện tại (m/s, rad/s)
        self.vx = 0.0
        self.wz = 0.0

        # Publisher
        self.odom_pub = rospy.Publisher("odom", Odometry, queue_size=10)
        self.tf_br    = tf.TransformBroadcaster()

        # Subscriber
        rospy.Subscriber(self.encoder_topic, Int32, self.encoder_cb, queue_size=50)
        rospy.Subscriber(self.imu_topic, Imu, self.imu_cb, queue_size=50)

        rospy.loginfo("encoder_imu_odom node started")

    # ===== IMU: tích phân gyro.z -> yaw =====
    def imu_cb(self, msg):
        # Dùng time trong header, nếu rỗng thì lấy now
        now = msg.header.stamp
        if now == rospy.Time():
            now = rospy.Time.now()

        if self.last_imu_time is None:
            self.last_imu_time = now
            return

        dt = (now - self.last_imu_time).to_sec()
        if dt <= 0.0:
            self.last_imu_time = now
            return

        # gyro quanh trục z (rad/s)
        gz = msg.angular_velocity.z * self.yaw_sign
        self.wz = gz

        # cập nhật yaw
        self.yaw += gz * dt

        # chuẩn hoá yaw về [-pi, pi]
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))

        self.last_imu_time = now

    # ===== Encoder: tính quãng đường & cập nhật x,y =====
    def encoder_cb(self, msg):
        now = rospy.Time.now()

        if self.last_ticks is None:
            self.last_ticks    = msg.data
            self.last_enc_time = now
            return

        dt = (now - self.last_enc_time).to_sec()
        if dt <= 0.0:
            return

        tick_diff = msg.data - self.last_ticks
        self.last_ticks    = msg.data
        self.last_enc_time = now

        # ticks -> vòng -> quãng đường
        rev = float(tick_diff) / float(self.ticks_per_rev)
        s   = 2.0 * math.pi * self.wheel_radius * rev      # m
        s   = self.encoder_sign * s

        self.vx = s / dt

        # cập nhật x, y dùng yaw hiện tại
        dx = s * math.cos(self.yaw)
        dy = s * math.sin(self.yaw)

        self.x += dx
        self.y += dy

        # publish odom + tf
        self.publish_odom(now)

    # ===== Publish Odometry + TF =====
    def publish_odom(self, stamp):
        # quaternion từ yaw
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, self.yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id  = self.base_frame

        # Pose
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = Quaternion(*q)

        # Twist
        odom.twist.twist.linear.x  = self.vx
        odom.twist.twist.linear.y  = 0.0
        odom.twist.twist.linear.z  = 0.0
        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = self.wz

        # (covariance để mặc định 0 hoặc bạn tự set thêm nếu muốn)
        self.odom_pub.publish(odom)

        # TF: odom -> base_link
        if self.publish_tf:
            self.tf_br.sendTransform(
                (self.x, self.y, 0.0),
                q,
                stamp,
                self.base_frame,
                self.odom_frame
            )


def main():
    rospy.init_node("encoder_imu_odom")
    node = EncoderImuOdomNode()
    rospy.spin()


if __name__ == "__main__":
    main()

