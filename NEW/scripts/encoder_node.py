#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import rospy
from std_msgs.msg import Int32, Bool

class EncoderStopNode(object):
    def __init__(self):
        # Topics
        self.encoder_topic = rospy.get_param("~encoder_topic", "encoder_ticks")
        self.stop_flag_topic = rospy.get_param("~stop_flag_topic", "/encoder_stop")

        # Convert ticks -> meters
        self.wheel_radius = float(rospy.get_param("~wheel_radius", 0.0325))     # m
        self.ticks_per_rev = float(rospy.get_param("~ticks_per_rev", 333 * 4)) # ticks / rev
        self.encoder_sign = float(rospy.get_param("~encoder_sign", -1.0))      # +1 hoặc -1

        # Stop condition
        self.target = float(rospy.get_param("~encoder_stop_distance", 0.20))   # m
        self.use_abs = bool(rospy.get_param("~encoder_stop_abs", True))        # cộng dồn |ds| (kể cả lùi)

        # State
        self.last_ticks = None
        self.dist = 0.0
        self.latched = False

        # Publisher (latched để keepstop chạy lên sau vẫn nhận True)
        self.pub_flag = rospy.Publisher(self.stop_flag_topic, Bool, queue_size=1, latch=True)

        # Subscriber
        rospy.Subscriber(self.encoder_topic, Int32, self.cb, queue_size=50)

        # init flag
        self.pub_flag.publish(False)

        rospy.loginfo("encoder_stop_node: encoder_topic=%s target=%.3fm flag_topic=%s",
                      self.encoder_topic, self.target, self.stop_flag_topic)

    def cb(self, msg):
        if self.latched:
            return

        ticks = msg.data
        if self.last_ticks is None:
            self.last_ticks = ticks
            return

        tick_diff = ticks - self.last_ticks
        self.last_ticks = ticks

        # ticks -> rev -> meters
        rev = float(tick_diff) / float(self.ticks_per_rev)
        ds = 2.0 * math.pi * float(self.wheel_radius) * rev
        ds = float(self.encoder_sign) * ds

        self.dist += abs(ds) if self.use_abs else ds

        if self.dist >= self.target:
            self.latched = True
            self.pub_flag.publish(True)
            rospy.logwarn("ENCODER STOP FLAG: reached %.3fm (target %.3fm) -> flag=True",
                          self.dist, self.target)

if __name__ == "__main__":
    rospy.init_node("encoder_stop_node", anonymous=False)
    EncoderStopNode()
    rospy.spin()
