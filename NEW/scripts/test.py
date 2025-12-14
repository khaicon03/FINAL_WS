#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random
import numpy as np

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


def wrap_pi(a):
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def scan_to_xy(scan):
    angles = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
    r = np.array(scan.ranges, dtype=np.float32)

    mask = np.isfinite(r)
    r = r[mask]
    angles = angles[mask]

    x = r * np.cos(angles)
    y = r * np.sin(angles)
    pts = np.stack([x, y], axis=1)
    return pts


def apply_tf_2d(pts, tx, ty, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    # Python2: không có toán tử "@"
    out = np.dot(pts, R.T)
    out[:, 0] += tx
    out[:, 1] += ty
    return out


def tls_line(pts):
    # ax + by + c = 0, ||(a,b)|| = 1
    if pts.shape[0] < 2:
        return None
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    n = float(np.linalg.norm(normal))
    if n < 1e-6:
        return None
    normal /= n
    a, b = float(normal[0]), float(normal[1])
    c = -(a * float(mean[0]) + b * float(mean[1]))
    return a, b, c


def ransac_line(pts, thresh, iters, min_inliers):
    n = pts.shape[0]
    if n < 2:
        return None

    best_inliers = None
    best_cnt = 0

    for _ in range(iters):
        i1, i2 = random.sample(range(n), 2)
        p1, p2 = pts[i1], pts[i2]
        v = p2 - p1
        vn = float(np.linalg.norm(v))
        if vn < 1e-6:
            continue
        v /= vn
        normal = np.array([-v[1], v[0]], dtype=np.float32)  # unit
        a, b = float(normal[0]), float(normal[1])
        c = -(a * float(p1[0]) + b * float(p1[1]))

        d = np.abs(a * pts[:, 0] + b * pts[:, 1] + c)  # since (a,b) unit
        inliers = d < thresh
        cnt = int(inliers.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inliers = inliers

    if best_inliers is None or best_cnt < min_inliers:
        return None

    refined = tls_line(pts[best_inliers])
    if refined is None:
        return None
    return refined  # (a,b,c)


def line_heading(a, b):
    # direction along wall: t = (-b, a)
    tx, ty = -b, a
    ang = math.atan2(ty, tx)
    # force "forward" (positive x) to avoid flips when averaging
    if tx < 0:
        ang = wrap_pi(ang + math.pi)
    return wrap_pi(ang)


class CorridorCenterCmdVel(object):
    def __init__(self):
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")

        # Driver của bạn: linear.x là PWM [-1..1]
        self.pwm_fwd = float(rospy.get_param("~pwm_fwd", 0.20))
        self.pwm_min = float(rospy.get_param("~pwm_min", 0.05))

        # Steering command in rad (firmware clamp ±30deg)
        self.delta_max = float(rospy.get_param("~delta_max", math.radians(30.0)))
        self.steer_sign = float(rospy.get_param("~steer_sign", 1.0))  # đảo -1 nếu lái ngược

        # ROI
        self.r_min = float(rospy.get_param("~r_min", 0.20))
        self.r_max = float(rospy.get_param("~r_max", 6.00))
        self.left_deg_min = float(rospy.get_param("~left_deg_min", 40.0))
        self.left_deg_max = float(rospy.get_param("~left_deg_max", 140.0))
        self.right_deg_min = float(rospy.get_param("~right_deg_min", -140.0))
        self.right_deg_max = float(rospy.get_param("~right_deg_max", -40.0))

        # Mounting offset (nếu lidar không nằm đúng tâm/không đúng hướng)
        self.laser_x = float(rospy.get_param("~laser_x", 0.0))
        self.laser_y = float(rospy.get_param("~laser_y", 0.0))
        self.laser_yaw = float(rospy.get_param("~laser_yaw", 0.0))

        # Control point (nếu bạn đo được “tâm trục sau nằm sau lidar bao nhiêu”)
        self.x_ref = float(rospy.get_param("~x_ref", 0.0))
        self.y_ref = float(rospy.get_param("~y_ref", 0.0))

        # RANSAC
        self.thresh = float(rospy.get_param("~ransac_thresh", 0.03))
        self.iters = int(rospy.get_param("~ransac_iters", 80))
        self.min_in = int(rospy.get_param("~min_inliers", 40))

        # Gains
        self.k_y = float(rospy.get_param("~k_y", 1.5))
        self.k_psi = float(rospy.get_param("~k_psi", 2.0))
        self.v0 = float(rospy.get_param("~v0", 0.2))  # chỉ để làm mượt term atan

        # Filter
        self.alpha = float(rospy.get_param("~ema_alpha", 0.55))
        self.ey_f = 0.0
        self.epsi_f = 0.0

        self.pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=10)
        self.sub = rospy.Subscriber(self.scan_topic, LaserScan, self.cb, queue_size=1)
        rospy.loginfo("corridor_center_cmdvel started (python2).")

    def stop(self):
        cmd = Twist()
        self.pub.publish(cmd)

    def cb(self, scan):
        pts = scan_to_xy(scan)

        r = np.linalg.norm(pts, axis=1)
        pts = pts[(r > self.r_min) & (r < self.r_max)]
        if pts.shape[0] < 80:
            self.stop()
            return

        pts = apply_tf_2d(pts, self.laser_x, self.laser_y, self.laser_yaw)

        ang_deg = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
        left_pts = pts[(ang_deg >= self.left_deg_min) & (ang_deg <= self.left_deg_max)]
        right_pts = pts[(ang_deg >= self.right_deg_min) & (ang_deg <= self.right_deg_max)]

        left = ransac_line(left_pts, self.thresh, self.iters, self.min_in) if left_pts.shape[0] >= 2 else None
        right = ransac_line(right_pts, self.thresh, self.iters, self.min_in) if right_pts.shape[0] >= 2 else None

        if left is None and right is None:
            self.stop()
            return

        def dist_at_ref(a, b, c):
            return a * self.x_ref + b * self.y_ref + c

        dL = None
        dR = None
        psi_list = []

        if left is not None:
            a, b, c = left
            dL = abs(dist_at_ref(a, b, c))
            psi_list.append(line_heading(a, b))

        if right is not None:
            a, b, c = right
            dR = abs(dist_at_ref(a, b, c))
            psi_list.append(line_heading(a, b))

        if len(psi_list) == 1:
            epsi = psi_list[0]
        else:
            s = sum(math.sin(p) for p in psi_list)
            c = sum(math.cos(p) for p in psi_list)
            epsi = wrap_pi(math.atan2(s, c))

        if dL is not None and dR is not None:
            ey = 0.5 * (dL - dR)
        elif dL is not None:
            target = float(rospy.get_param("~single_wall_target", dL))
            ey = (dL - target)
        else:
            target = float(rospy.get_param("~single_wall_target", dR))
            ey = -(dR - target)

        self.ey_f = self.alpha * ey + (1.0 - self.alpha) * self.ey_f
        self.epsi_f = self.alpha * epsi + (1.0 - self.alpha) * self.epsi_f

        v = max(self.pwm_fwd, self.pwm_min)
        delta = self.k_psi * self.epsi_f + math.atan2(self.k_y * self.ey_f, (v + self.v0))
        delta = max(-self.delta_max, min(self.delta_max, delta))
        delta *= self.steer_sign

        cmd = Twist()
        cmd.linear.x = self.pwm_fwd
        cmd.angular.z = delta  # driver STM32 của bạn hiểu đây là góc lái rad
        self.pub.publish(cmd)


if __name__ == "__main__":
    rospy.init_node("corridor_center_cmdvel")
    CorridorCenterCmdVel()
    rospy.spin()
