#!/usr/bin/env python3
import math
import random
import numpy as np

import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def scan_to_points(scan: LaserScan):
    """Return Nx2 points (x,y) in the LiDAR frame."""
    angles = scan.angle_min + np.arange(len(scan.ranges)) * scan.angle_increment
    r = np.array(scan.ranges, dtype=np.float32)

    mask = np.isfinite(r)
    r = r[mask]
    angles = angles[mask]

    x = r * np.cos(angles)
    y = r * np.sin(angles)
    pts = np.stack([x, y], axis=1)
    return pts, angles, r


def apply_rigid_transform(pts: np.ndarray, tx: float, ty: float, yaw: float):
    """Transform points by R(yaw)*p + t."""
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    out = (pts @ R.T)
    out[:, 0] += tx
    out[:, 1] += ty
    return out


def total_least_squares_line(pts: np.ndarray):
    """
    Fit a 2D line ax + by + c = 0 with ||(a,b)||=1 using TLS (SVD).
    Returns (a,b,c) or None if not enough points.
    """
    if pts.shape[0] < 2:
        return None

    mean = pts.mean(axis=0)
    centered = pts - mean

    # direction = first principal component
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]  # unit-ish
    # normal is perpendicular to direction
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-6:
        return None
    normal /= nrm
    a, b = float(normal[0]), float(normal[1])
    c = -(a * float(mean[0]) + b * float(mean[1]))
    return a, b, c


def ransac_line(pts: np.ndarray, thresh: float, iters: int, min_inliers: int):
    """
    RANSAC to get inliers, then refine with TLS.
    Returns (a,b,c,inlier_count,inlier_ratio) or None.
    """
    n = pts.shape[0]
    if n < 2:
        return None

    best_inliers = None
    best_count = 0

    for _ in range(iters):
        i1, i2 = random.sample(range(n), 2)
        p1, p2 = pts[i1], pts[i2]
        v = p2 - p1
        vn = float(np.linalg.norm(v))
        if vn < 1e-6:
            continue
        v = v / vn
        normal = np.array([-v[1], v[0]], dtype=np.float32)  # unit
        a, b = float(normal[0]), float(normal[1])
        c = -(a * float(p1[0]) + b * float(p1[1]))

        # distance to line
        d = np.abs(a * pts[:, 0] + b * pts[:, 1] + c)  # since (a,b) unit
        inliers = d < thresh
        cnt = int(inliers.sum())
        if cnt > best_count:
            best_count = cnt
            best_inliers = inliers

    if best_inliers is None or best_count < min_inliers:
        return None

    inlier_pts = pts[best_inliers]
    refined = total_least_squares_line(inlier_pts)
    if refined is None:
        return None

    a, b, c = refined
    ratio = best_count / float(n)
    return a, b, c, best_count, ratio


def line_direction_angle(a: float, b: float):
    """
    Line normal (a,b). A direction vector along the line is t = (-b, a).
    Return angle of t in radians, normalized to [-pi,pi], and t vector.
    """
    tx, ty = -b, a
    ang = math.atan2(ty, tx)
    ang = wrap_pi(ang)
    return ang, np.array([tx, ty], dtype=np.float32)


class CorridorCenteringNode:
    def __init__(self):
        # Params
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.cmd_topic = rospy.get_param("~cmd_topic", "/cmd_vel")

        self.wheelbase = float(rospy.get_param("~wheelbase", 0.24))
        self.speed = float(rospy.get_param("~speed", 0.3))
        self.v0 = float(rospy.get_param("~v0", 0.2))

        # LiDAR mounting offset to control frame (optional)
        self.laser_x = float(rospy.get_param("~laser_x", 0.0))
        self.laser_y = float(rospy.get_param("~laser_y", 0.0))
        self.laser_yaw = float(rospy.get_param("~laser_yaw", 0.0))

        # control point relative to the control frame (e.g., rear axle center)
        # If you know rear axle is behind LiDAR by X meters in the same frame, set x_ref = -X
        self.x_ref = float(rospy.get_param("~x_ref", 0.0))
        self.y_ref = float(rospy.get_param("~y_ref", 0.0))

        # ROI
        self.r_min = float(rospy.get_param("~r_min", 0.2))
        self.r_max = float(rospy.get_param("~r_max", 6.0))
        self.left_deg_min = float(rospy.get_param("~left_deg_min", 40.0))
        self.left_deg_max = float(rospy.get_param("~left_deg_max", 140.0))
        self.right_deg_min = float(rospy.get_param("~right_deg_min", -140.0))
        self.right_deg_max = float(rospy.get_param("~right_deg_max", -40.0))

        # RANSAC
        self.ransac_thresh = float(rospy.get_param("~ransac_thresh", 0.03))
        self.ransac_iters = int(rospy.get_param("~ransac_iters", 80))
        self.min_inliers = int(rospy.get_param("~min_inliers", 40))

        # Gains
        self.ky = float(rospy.get_param("~k_y", 1.5))
        self.kpsi = float(rospy.get_param("~k_psi", 2.0))

        # Limits
        self.delta_max = float(rospy.get_param("~delta_max", 0.45))  # rad (~26 deg)
        self.omega_max = float(rospy.get_param("~omega_max", 2.0))   # rad/s
        self.speed_min = float(rospy.get_param("~speed_min", 0.05))

        # Filtering
        self.alpha = float(rospy.get_param("~ema_alpha", 0.5))
        self.ey_f = 0.0
        self.epsi_f = 0.0

        self.pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=10)
        self.sub = rospy.Subscriber(self.scan_topic, LaserScan, self.on_scan, queue_size=1)

        rospy.loginfo("corridor_centering node started.")

    def on_scan(self, scan: LaserScan):
        pts, angles, r = scan_to_points(scan)

        # filter range
        mask = (r > self.r_min) & (r < self.r_max)
        pts = pts[mask]
        angles = angles[mask]

        if pts.shape[0] < 50:
            self.publish_stop()
            return

        # apply fixed mounting transform if needed (laser -> control frame)
        pts_cf = apply_rigid_transform(pts, self.laser_x, self.laser_y, self.laser_yaw)

        # ROI split by angle (angles are in original scan frame, but yaw offset is small; for safety,
        # recompute angle from transformed points)
        ang_cf = np.arctan2(pts_cf[:, 1], pts_cf[:, 0])  # [-pi,pi]
        ang_deg = np.degrees(ang_cf)

        left_mask = (ang_deg >= self.left_deg_min) & (ang_deg <= self.left_deg_max)
        right_mask = (ang_deg >= self.right_deg_min) & (ang_deg <= self.right_deg_max)

        left_pts = pts_cf[left_mask]
        right_pts = pts_cf[right_mask]

        left_line = ransac_line(left_pts, self.ransac_thresh, self.ransac_iters, self.min_inliers) if left_pts.shape[0] >= 2 else None
        right_line = ransac_line(right_pts, self.ransac_thresh, self.ransac_iters, self.min_inliers) if right_pts.shape[0] >= 2 else None

        if left_line is None and right_line is None:
            # nothing reliable
            self.publish_stop()
            return

        # compute e_y and e_psi with fallbacks
        ey, epsi = 0.0, 0.0
        have_heading = False

        # distance from control point (x_ref,y_ref) to line ax+by+c=0
        def dist_at_ref(a,b,c):
            return (a*self.x_ref + b*self.y_ref + c)  # signed, since ||(a,b)||=1

        dL = None
        if left_line is not None:
            aL, bL, cL, cntL, ratioL = left_line
            dL = abs(dist_at_ref(aL,bL,cL))
            angL, tL = line_direction_angle(aL,bL)
            # make direction point "forward" (positive x) to avoid averaging opposite angles
            if tL[0] < 0:
                angL = wrap_pi(angL + math.pi)
            epsi = angL
            have_heading = True

        dR = None
        if right_line is not None:
            aR, bR, cR, cntR, ratioR = right_line
            dR = abs(dist_at_ref(aR,bR,cR))
            angR, tR = line_direction_angle(aR,bR)
            if tR[0] < 0:
                angR = wrap_pi(angR + math.pi)

            if have_heading:
                # average angles carefully
                epsi = wrap_pi(math.atan2(math.sin(epsi) + math.sin(angR),
                                          math.cos(epsi) + math.cos(angR)))
            else:
                epsi = angR
                have_heading = True

        # lateral centering
        if dL is not None and dR is not None:
            ey = 0.5 * (dL - dR)  # >0 means more space on left -> steer left
        elif dL is not None:
            # keep a target distance to left wall (set target as current to avoid sudden jump)
            # You can set ~single_wall_target if you want.
            target = float(rospy.get_param("~single_wall_target", dL))
            ey = (dL - target)
        elif dR is not None:
            target = float(rospy.get_param("~single_wall_target", dR))
            ey = -(dR - target)

        # EMA filter
        self.ey_f = self.alpha * ey + (1.0 - self.alpha) * self.ey_f
        self.epsi_f = self.alpha * epsi + (1.0 - self.alpha) * self.epsi_f

        # control law -> steering angle delta
        v = max(self.speed, self.speed_min)
        delta = self.kpsi * self.epsi_f + math.atan2(self.ky * self.ey_f, (v + self.v0))
        delta = max(-self.delta_max, min(self.delta_max, delta))

        # convert to cmd_vel (yaw rate) if your driver expects yaw-rate:
        omega = (v / self.wheelbase) * math.tan(delta)
        omega = max(-self.omega_max, min(self.omega_max, omega))

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = omega
        self.pub.publish(cmd)

    def publish_stop(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.pub.publish(cmd)


if __name__ == "__main__":
    rospy.init_node("corridor_centering")
    node = CorridorCenteringNode()
    rospy.spin()
