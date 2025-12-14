#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist

class VelocityMux:
    def __init__(self):
        # Khởi tạo node
        rospy.init_node('cmd_vel_mux', anonymous=True)

        # --- CẤU HÌNH ---
        self.cmd_nav_topic = '/cmd_vel_nav'  # Topic ưu tiên 1 (Navigation)
        self.cmd_row_topic = '/cmd_vel_row'  # Topic ưu tiên 2 (Đi theo luống)
        self.cmd_out_topic = '/cmd_vel'      # Topic đầu ra (xuống STM32)
        
        self.timeout = 0.5  # (Giây) Nếu quá thời gian này không nhận được lệnh thì coi như mất tín hiệu
        self.rate = 20      # Tần số update đầu ra (Hz)

        # --- KHỞI TẠO BIẾN ---
        self.nav_msg = Twist()
        self.row_msg = Twist()
        
        self.last_nav_time = rospy.Time(0)
        self.last_row_time = rospy.Time(0)
        
        self.has_nav = False
        self.has_row = False

        # --- SUBSCRIBERS ---
        rospy.Subscriber(self.cmd_nav_topic, Twist, self.nav_callback)
        rospy.Subscriber(self.cmd_row_topic, Twist, self.row_callback)

        # --- PUBLISHER ---
        self.pub = rospy.Publisher(self.cmd_out_topic, Twist, queue_size=1)

        # --- TIMER ---
        # Tạo vòng lặp kiểm tra và gửi dữ liệu liên tục
        rospy.Timer(rospy.Duration(1.0/self.rate), self.control_loop)

        rospy.loginfo("Vel Mux Node Started.")
        rospy.loginfo("Priority: 1. %s | 2. %s" % (self.cmd_nav_topic, self.cmd_row_topic))

    def nav_callback(self, msg):
        self.nav_msg = msg
        self.last_nav_time = rospy.Time.now()
        self.has_nav = True

    def row_callback(self, msg):
        self.row_msg = msg
        self.last_row_time = rospy.Time.now()
        self.has_row = True

    def control_loop(self, event):
        current_time = rospy.Time.now()
        output_msg = Twist() # Mặc định là v=0, w=0 (Dừng)
        mode = "IDLE"

        # 1. Kiểm tra ưu tiên CAO NHẤT (Navigation)
        # Logic: Có tin nhắn VÀ tin nhắn đó mới nhận cách đây chưa quá timeout
        if self.has_nav and (current_time - self.last_nav_time).to_sec() < self.timeout:
            output_msg = self.nav_msg
            mode = "NAV"
        
        # 2. Kiểm tra ưu tiên THẤP HƠN (Row Following)
        # Logic: Không dùng Nav VÀ Row có tin nhắn mới
        elif self.has_row and (current_time - self.last_row_time).to_sec() < self.timeout:
            output_msg = self.row_msg
            mode = "ROW"
            
        # 3. Trường hợp IDLE (Cả 2 đều không gửi hoặc mất kết nối)
        else:
            # output_msg vẫn giữ nguyên là 0 (Stop)
            mode = "IDLE (Stop)"

        # Publish lệnh cuối cùng
        self.pub.publish(output_msg)
        
        # Log trạng thái (dùng throttle để không spam terminal)
        # Chỉ in log mỗi 2 giây/lần
        rospy.loginfo_throttle(2, "Mux Mode: %s | v=%.2f, w=%.2f" % (mode, output_msg.linear.x, output_msg.angular.z))

if __name__ == '__main__':
    try:
        mux = VelocityMux()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
