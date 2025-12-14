#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# YOLOv8
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class DualRowFollowerROI:
    def __init__(self):
        rospy.loginfo("Dual Row Follower (ROI-based + YOLO) started.")

        self.bridge = CvBridge()

        # xuất lệnh ra /cmd_vel_row
        self.cmd_pub = rospy.Publisher("/cmd_vel_row", Twist, queue_size=10)

        # ========= PARAMS từ launch =========
        self.image_topic = rospy.get_param("~image_topic", "/csi_cam_0/image_raw")
        self.show_debug  = rospy.get_param("~show_debug", True)

        # tốc độ tiến
        self.forward_speed = rospy.get_param("~forward_speed", 0.25)

        # target theo 10cm
        self.left_target_ratio  = rospy.get_param("~left_target_ratio", 0.30)
        self.right_target_ratio = rospy.get_param("~right_target_ratio", 0.70)
        self.center_ratio       = rospy.get_param("~center_ratio", 0.50)

        # vùng trái/phải loại line giả
        self.left_min_x_ratio  = rospy.get_param("~left_min_x_ratio", 0.05)
        self.left_max_x_ratio  = rospy.get_param("~left_max_x_ratio", 0.45)
        self.right_min_x_ratio = rospy.get_param("~right_min_x_ratio", 0.55)
        self.right_max_x_ratio = rospy.get_param("~right_max_x_ratio", 0.95)

        # gain điều khiển
        self.k_ang   = rospy.get_param("~k_ang", 2.0)
        self.max_ang = rospy.get_param("~max_ang", 1.0)

        # Canny + Hough params
        self.canny1          = rospy.get_param("~canny1", 50)
        self.canny2          = rospy.get_param("~canny2", 150)
        self.hough_threshold = rospy.get_param("~hough_threshold", 40)

        # NGƯỠNG GÓC: chỉ giữ các line có góc so với trục NẰM (x) >= min_angle_deg
        self.min_angle_deg = max(35.0, rospy.get_param("~min_angle_deg", 35.0))

        # ================== PARAM YOLO ==================
        self.use_yolo        = rospy.get_param("~use_yolo", False)
        self.yolo_model_path = rospy.get_param("~yolo_model_path", "")
        self.yolo_conf       = rospy.get_param("~yolo_conf", 0.5)
        self.yolo_iou        = rospy.get_param("~yolo_iou", 0.45)

        # Khung dọc từ 30% đến 60% chiều ngang để check vật cản (full chiều cao)
        self.obstacle_x_min_ratio = rospy.get_param("~obstacle_x_min_ratio", 0.30)
        self.obstacle_x_max_ratio = rospy.get_param("~obstacle_x_max_ratio", 0.60)

        # Các class id được coi là CÂY. Mọi class KHÔNG thuộc list này
        # nếu nằm trong khung 30–60% sẽ bị coi là vật cản.
        self.plant_class_ids = rospy.get_param("~plant_class_ids", [])

        self.yolo_model = None
        if self.use_yolo:
            if YOLO is None:
                rospy.logwarn("use_yolo=True nhưng không import được ultralytics.YOLO, tắt YOLO.")
                self.use_yolo = False
            elif not self.yolo_model_path:
                rospy.logwarn("use_yolo=True nhưng không có yolo_model_path, tắt YOLO.")
                self.use_yolo = False
            else:
                try:
                    rospy.loginfo("Loading YOLOv8 model from: %s", self.yolo_model_path)
                    self.yolo_model = YOLO(self.yolo_model_path)
                except Exception as e:
                    rospy.logerr("Không load được YOLOv8 model: %s", e)
                    self.use_yolo = False

        # ================== COLOR CHECK TRONG BBOX CÂY ==================
        self.use_color_check = rospy.get_param("~use_color_check", True)

        # lower / upper HSV (bạn override trong launch)
        lower_default = [35, 50, 50]   # ví dụ
        upper_default = [85, 255, 255]

        self.color_lower = np.array(
            rospy.get_param("~color_lower", lower_default),
            dtype=np.uint8
        )
        self.color_upper = np.array(
            rospy.get_param("~color_upper", upper_default),
            dtype=np.uint8
        )

        # % pixel trong bbox thuộc dải màu này để coi là "match"
        self.color_min_ratio = rospy.get_param("~color_min_ratio", 0.2)  # 20%

        # ========== STATE: DỪNG KHI GẶP CÂY ĐỂ KIỂM TRA MÀU ==========
        self.stop_on_plant = rospy.get_param("~stop_on_plant", True)
        self.inspect_pause_frames = rospy.get_param("~inspect_pause_frames", 10)
        self.inspect_counter = 0  # >0 nghĩa là vừa check xong cây, chưa dừng lại nữa

        # subscriber camera
        rospy.Subscriber(self.image_topic, Image, self.image_callback)

    # =====================================================================
    # PHÁT HIỆN LUỐNG BẰNG HOUGHLINES + ROI NGANG (NỬA DƯỚI)
    # =====================================================================
    def detect_rows(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray_blur, self.canny1, self.canny2)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.hough_threshold,
            minLineLength=int(roi.shape[0] * 0.5),
            maxLineGap=25
        )

        left_xs = []
        right_xs = []

        h, w = roi.shape[:2]

        left_min  = int(w * self.left_min_x_ratio)
        left_max  = int(w * self.left_max_x_ratio)
        right_min = int(w * self.right_min_x_ratio)
        right_max = int(w * self.right_max_x_ratio)

        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]

                dx = x2 - x1
                dy = y2 - y1

                angle = np.degrees(np.arctan2(dy, dx))
                angle = abs(angle)
                if angle > 90:
                    angle = 180 - angle

                if angle < self.min_angle_deg:
                    continue

                x = x1 if y1 > y2 else x2

                if left_min <= x <= left_max:
                    left_xs.append(x)
                elif right_min <= x <= right_max:
                    right_xs.append(x)

        return left_xs, right_xs, edges, lines

    # =====================================================================
    # CHECK MÀU TRONG BBOX CÂY (HSV)
    # =====================================================================
    def check_plant_color(self, frame, bbox):
        """
        bbox: (x1, y1, x2, y2) theo frame gốc (BGR).
        Trả về: (is_match, ratio)
        is_match = True nếu tỉ lệ pixel thuộc dải [color_lower, color_upper] >= color_min_ratio
        """
        if not self.use_color_check:
            return False, 0.0

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w,     x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h,     y2))

        if x2 <= x1 or y2 <= y1:
            return False, 0.0

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False, 0.0

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)

        match_pixels = float(np.count_nonzero(mask))
        total_pixels = float(mask.size)
        if total_pixels <= 0:
            return False, 0.0

        ratio = match_pixels / total_pixels
        is_match = (ratio >= self.color_min_ratio)
        return is_match, ratio

    # =====================================================================
    # CALLBACK: xử lý từng frame
    # =====================================================================
    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            return

        h, w, _ = frame.shape

        # giảm counter nếu đang ở pha "đi tiếp sau khi kiểm tra cây"
        if self.inspect_counter > 0:
            self.inspect_counter -= 1

        # =====================================================
        # 1) YOLO: OBSTACLE ROI 30–60% NGANG + CÂY TOÀN KHUNG
        # =====================================================
        obstacle_on_path = False
        obstacle_boxes = []   # (x1,y1,x2,y2,cls_id,conf)
        plant_detections = [] # (x1,y1,x2,y2,cls_id,conf,is_match,ratio)

        if self.use_yolo and self.yolo_model is not None:
            try:
                results = self.yolo_model(
                    frame,
                    conf=self.yolo_conf,
                    iou=self.yolo_iou,
                    verbose=False
                )
            except Exception as e:
                rospy.logwarn("YOLO inference failed: %s", e)
                results = []

            if results:
                r = results[0]
                if hasattr(r, "boxes") and r.boxes is not None:
                    path_x_min = int(self.obstacle_x_min_ratio * w)
                    path_x_max = int(self.obstacle_x_max_ratio * w)

                    for box in r.boxes:
                        x1b, y1b, x2b, y2b = box.xyxy[0].cpu().numpy()
                        cls_id = int(box.cls[0].cpu().numpy())
                        conf   = float(box.conf[0].cpu().numpy())

                        x1b = int(x1b)
                        y1b = int(y1b)
                        x2b = int(x2b)
                        y2b = int(y2b)

                        cx = (x1b + x2b) // 2

                        # ---- A. Check vật cản trong khung 30–60% (full dọc) ----
                        if path_x_min <= cx <= path_x_max:
                            # Nếu có plant_class_ids:
                            #   - class KHÔNG thuộc plant_class_ids => vật cản
                            # Nếu KHÔNG set plant_class_ids => mọi class đều coi là vật cản
                            if (self.plant_class_ids and cls_id not in self.plant_class_ids) or \
                               (not self.plant_class_ids):
                                obstacle_on_path = True
                                obstacle_boxes.append((x1b, y1b, x2b, y2b, cls_id, conf))

                        # ---- B. Detect cây trên toàn khung hình ----
                        if self.plant_class_ids and cls_id in self.plant_class_ids:
                            is_match, ratio = self.check_plant_color(
                                frame, (x1b, y1b, x2b, y2b)
                            )
                            plant_detections.append(
                                (x1b, y1b, x2b, y2b, cls_id, conf, is_match, ratio)
                            )

        # Nếu có vật cản trong khung 30–60% -> dừng và (nếu debug) vẽ khung rồi RETURN
        if obstacle_on_path:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)

            if self.show_debug:
                frame_obs = frame.copy()
                path_x_min = int(self.obstacle_x_min_ratio * w)
                path_x_max = int(self.obstacle_x_max_ratio * w)

                cv2.rectangle(frame_obs,
                              (path_x_min, 0),
                              (path_x_max, h - 1),
                              (0, 255, 255), 2)

                for (x1b, y1b, x2b, y2b, cls_id, conf) in obstacle_boxes:
                    cv2.rectangle(frame_obs, (x1b, y1b), (x2b, y2b), (0, 0, 255), 2)
                    txt = "OBS id=%d conf=%.2f" % (cls_id, conf)
                    cv2.putText(frame_obs, txt, (x1b, max(0, y1b - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1)

                cv2.imshow("Obstacle_ROI_30_60", frame_obs)
                cv2.waitKey(1)

            return  # frame này không bám luống

        # =====================================================
        # 1.5) STOP KHI GẶP CÂY ĐỂ KIỂM TRA MÀU, SAU ĐÓ MỚI ĐI TIẾP
        # =====================================================
        stop_for_plant = False
        plant_to_inspect = None

        if self.stop_on_plant and self.inspect_counter <= 0:
            if len(plant_detections) > 0:
                # Lấy cây gần giữa ngang nhất (thay vì lấy bừa cây đầu tiên)
                center_x = w // 2
                best_idx = 0
                best_dist = 1e9
                for i, (x1b, y1b, x2b, y2b, cls_id, conf, is_match, ratio) in enumerate(plant_detections):
                    cx = (x1b + x2b) // 2
                    dist = abs(cx - center_x)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i

                plant_to_inspect = plant_detections[best_idx]
                stop_for_plant = True

        if stop_for_plant and plant_to_inspect is not None:
            # dừng robot
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)

            x1b, y1b, x2b, y2b, cls_id, conf, is_match_prev, ratio_prev = plant_to_inspect

            # kiểm tra màu lá lần nữa (nếu muốn chắc chắn)
            is_match, ratio = self.check_plant_color(frame, (x1b, y1b, x2b, y2b))

            if is_match:
                rospy.loginfo("PLANT COLOR MATCH at bbox (%d,%d,%d,%d), ratio=%.2f",
                              x1b, y1b, x2b, y2b, ratio)
            else:
                rospy.loginfo("PLANT COLOR NOT MATCH at bbox (%d,%d,%d,%d), ratio=%.2f",
                              x1b, y1b, x2b, y2b, ratio)

            # đặt thời gian "đi tiếp" không dừng lại liền
            self.inspect_counter = self.inspect_pause_frames

            if self.show_debug:
                inspect_vis = frame.copy()
                color = (0, 0, 255) if is_match else (0, 255, 0)
                txt = "MATCH" if is_match else "NORMAL"
                cv2.rectangle(inspect_vis, (x1b, y1b), (x2b, y2b), color, 2)
                cv2.putText(inspect_vis, txt, (x1b, max(0, y1b - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.imshow("Plant Inspection", inspect_vis)
                cv2.waitKey(1)

            # rất quan trọng: RETURN → frame này không bám luống
            return

        # =====================================================
        # 2) KHÔNG VẬT CẢN, KHÔNG ĐANG KIỂM TRA CÂY: BÁM LUỐNG
        # =====================================================
        roi = frame[int(h * 0.5):h, :]

        left_xs, right_xs, edges, lines = self.detect_rows(roi)

        # ======== DEBUG HIỂN THỊ ========
        if self.show_debug:
            roi_vis = roi.copy()
            roi_h, roi_w, _ = roi_vis.shape

            left_min  = int(roi_w * self.left_min_x_ratio)
            left_max  = int(roi_w * self.left_max_x_ratio)
            right_min = int(roi_w * self.right_min_x_ratio)
            right_max = int(roi_w * self.right_max_x_ratio)

            cv2.rectangle(roi_vis, (left_min, 0), (left_max, roi_h - 1), (255, 0, 0), 2)
            cv2.rectangle(roi_vis, (right_min, 0), (right_max, roi_h - 1), (0, 255, 0), 2)

            cx_roi = int(self.center_ratio * roi_w)
            cv2.line(roi_vis, (cx_roi, 0), (cx_roi, roi_h - 1), (255, 0, 255), 1)

            if lines is not None:
                for l in lines:
                    x1_l, y1_l, x2_l, y2_l = l[0]
                    dx = x2_l - x1_l
                    dy = y2_l - y1_l

                    angle = np.degrees(np.arctan2(dy, dx))
                    angle = abs(angle)
                    if angle > 90.0:
                        angle = 180.0 - angle

                    if angle < self.min_angle_deg:
                        continue

                    cv2.line(roi_vis, (x1_l, y1_l), (x2_l, y2_l), (0, 0, 255), 2)

            for x in left_xs:
                cv2.line(roi_vis, (x, 0), (x, roi_h - 1), (255, 255, 0), 1)
            for x in right_xs:
                cv2.line(roi_vis, (x, 0), (x, roi_h - 1), (0, 255, 255), 1)

            frame_vis = frame.copy()

            # vẽ khung 30–60% cho dễ hình dung
            path_x_min = int(self.obstacle_x_min_ratio * w)
            path_x_max = int(self.obstacle_x_max_ratio * w)
            cv2.rectangle(frame_vis,
                          (path_x_min, 0),
                          (path_x_max, h - 1),
                          (0, 255, 255), 1)

            # vẽ bbox cây + trạng thái màu (từ plant_detections)
            for (x1b, y1b, x2b, y2b, cls_id, conf, is_match, ratio) in plant_detections:
                color = (0, 0, 255) if is_match else (0, 255, 0)
                label = "MATCH" if is_match else "NORMAL"
                cv2.rectangle(frame_vis, (x1b, y1b), (x2b, y2b), color, 2)
                txt = "%s(%.2f)" % (label, conf)
                cv2.putText(frame_vis, txt, (x1b, max(0, y1b - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            x1 = 0
            y1 = int(h * 0.5)
            x2 = w
            y2 = h

            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cx_full = int(self.center_ratio * w)
            cv2.line(frame_vis, (cx_full, y1), (cx_full, y2), (255, 0, 0), 1)

            for x in left_xs:
                cv2.line(frame_vis, (x, y1), (x, y2), (255, 255, 0), 1)
            for x in right_xs:
                cv2.line(frame_vis, (x, y1), (x, y2), (0, 255, 255), 1)

            cv2.imshow("Frame with ROI + rows + plants", frame_vis)
            cv2.imshow("ROI_with_lines", roi_vis)
            cv2.imshow("Edges", edges)
            cv2.waitKey(1)
        # =========== Hết DEBUG ===========

        # ------------------ ĐIỀU KHIỂN BÁM LUỐNG ------------------
        cmd = Twist()
        cmd.linear.x = self.forward_speed

        if len(left_xs) > 0 and len(right_xs) > 0:
            x_left  = max(left_xs)
            x_right = min(right_xs)
            center = (x_left + x_right) / 2.0
            desired = self.center_ratio * w
            error = (desired - center) / float(w)

        elif len(left_xs) > 0:
            x_left = max(left_xs)
            desired = self.left_target_ratio * w
            error = (desired - x_left) / float(w)

        elif len(right_xs) > 0:
            x_right = min(right_xs)
            desired = self.right_target_ratio * w
            error = (desired - x_right) / float(w)

        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return

        ang = self.k_ang * error
        ang = max(-self.max_ang, min(self.max_ang, ang))
        cmd.angular.z = ang

        self.cmd_pub.publish(cmd)


if __name__ == "__main__":
    rospy.init_node("row_follow_dual_roi")
    node = DualRowFollowerROI()
    rospy.spin()
