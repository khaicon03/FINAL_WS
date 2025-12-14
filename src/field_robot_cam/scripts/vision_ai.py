#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
import os
from sensor_msgs.msg import LaserScan

# ========== C?U HÌNH MÀU S?C (HSV) CHO YOLO ==========
# (B?n gi? nguyên thông s? màu ?ã tune c?a b?n)
LOWER_HEALTHY = np.array([35, 80, 80], dtype=np.uint8)
UPPER_HEALTHY = np.array([85, 255, 255], dtype=np.uint8)

class AgriVisionBridge:
    def __init__(self):
        rospy.init_node("agri_vision_bridge")
        rospy.loginfo("--- AGRI VISION BRIDGE: CAM -> LASERSCAN STARTING ---")

        # --- Publisher quan tr?ng nh?t cho Move Base ---
        self.scan_pub = rospy.Publisher("/vision/scan", LaserScan, queue_size=1)

        # ========= THAM S? C?U HÌNH (C?N ?O ??C TH?C T?) =========
        self.cam_fov_h = 62.2       # Góc m? ngang c?a IMX219 (??)
        self.cam_height = 0.08      # Chi?u cao camera tính t? m?t ??t (mét) - VÍ D?: 30cm
        self.cam_tilt = 0        # Góc cúi c?a camera (??). 0 là nhìn th?ng, 90 là nhìn xu?ng ??t.
        
        # Gi?i h?n t?m nhìn (mét)
        self.scan_range_min = 0.1
        self.scan_range_max = 4.0

        # ========= YOLO CONFIG (Gi? nguyên c?a b?n) =========
        self.init_yolo()

        # ========= CAMERA CONFIG (GStreamer cho Jetson Nano) =========
        self.gst_str = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=640, height=360, format=BGRx ! " 
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink"
        )
        self.cap = cv2.VideoCapture(self.gst_str, cv2.CAP_GSTREAMER)

        # Bi?n ??m frame ?? gi?m t?i YOLO
        self.frame_count = 0
        self.skip_yolo_frames = 2 
        self.last_detections = []   
        self.yolo_input_size = (320, 320)

    def init_yolo(self):
        # T? ??ng tìm ???ng d?n file
        script_dir = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(script_dir, "custom_yolov4_tiny_best.weights")
        cfg_path = os.path.join(script_dir, "custom_yolov4_tiny.cfg")
        names_path = os.path.join(script_dir, "coco.names") 

        rospy.loginfo(f"Loading YOLO from: {weights_path}")
        try:
            self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            try:
                with open(names_path, "r") as f: self.classes = [l.strip() for l in f.readlines()]
            except: self.classes = ["unknown"]

            self.layer_names = self.net.getLayerNames()
            out_layers_indices = self.net.getUnconnectedOutLayers()
            if len(out_layers_indices.shape) > 1: out_layers_indices = out_layers_indices.flatten()
            self.output_layers = [self.layer_names[i - 1] for i in out_layers_indices]
            self.yolo_ready = True
        except Exception as e:
            rospy.logerr(f"YOLO Error: {e}")
            self.yolo_ready = False

    def detect_objects(self, frame):
        # (Hàm c? c?a b?n - Gi? nguyên logic)
        if not self.yolo_ready: return []
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, self.yolo_input_size, (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.4:
                    center_x, center_y = int(detection[0]*width), int(detection[1]*height)
                    w, h = int(detection[2]*width), int(detection[3]*height)
                    x, y = int(center_x - w/2), int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
        results = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                label = self.classes[class_ids[i]] if class_ids[i] < len(self.classes) else "unknown"
                results.append({"box": boxes[i], "label": label})
        return results

    def pixel_to_distance(self, pixel_y, image_height):
        """
        Chuy?n ??i to? ?? pixel Y sang kho?ng cách th?c t? (mét)
        S? d?ng mô hình Pinhole Camera ??n gi?n.
        pixel_y: to? ?? dòng (t? 0 ? trên cùng ??n image_height ? d??i ?áy)
        """
        # Chuy?n h? to? ??: G?c ? tâm ?nh
        # v d??ng h??ng xu?ng d??i
        v = pixel_y - (image_height / 2.0)
        
        # Góc nhìn t??ng ?ng v?i pixel này (theo chi?u d?c)
        # Gi? s? FOV d?c t? l? v?i FOV ngang theo aspect ratio (??n gi?n hoá)
        # Ho?c dùng s? c? ??nh: IMX219 Vertical FOV kho?ng 48.8 ??
        v_fov_rad = math.radians(48.8)
        
        # Góc l?ch so v?i tr?c quang h?c
        angle_pixel = (v / (image_height / 2.0)) * (v_fov_rad / 2.0)
        
        # T?ng góc so v?i ph??ng ngang = Góc nghiêng camera + Góc pixel
        total_angle = math.radians(self.cam_tilt) + angle_pixel
        
        # N?u góc <= 0 (nhìn lên tr?i) -> Vô c?c
        if total_angle <= 0.05: # Tránh chia cho 0 ho?c s? âm nh?
            return float('inf')
        
        # Công th?c: d = h / tan(theta)
        distance = self.cam_height / math.tan(total_angle)
        return distance

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret: 
                rate.sleep()
                continue
            
            self.frame_count += 1
            h, w = frame.shape[:2]

            # --- 1. YOLO DETECTION ---
            if self.frame_count % (self.skip_yolo_frames + 1) == 0:
                self.last_detections = self.detect_objects(frame)
            
            # --- 2. X? LÝ ?NH (Hough Line) ---
            # C?t ROI (ch? l?y n?a d??i ?nh ?? ?? nhi?u n?n tr?i)
            roi_start = int(h * 0.4) 
            roi = frame[roi_start:h, 0:w]
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            edges = cv2.Canny(blur, 50, 150)
            
            # Tìm ???ng th?ng
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, minLineLength=80, maxLineGap=40)

            # --- 3. T?O LASERSCAN GI? L?P ---
            scan = LaserScan()
            scan.header.stamp = rospy.Time.now()
            scan.header.frame_id = "camera_link" # QUAN TR?NG: Ph?i kh?p v?i TF Tree
            
            # C?u hình góc quét (t??ng ?ng v?i chi?u ngang ?nh)
            fov_rad = math.radians(self.cam_fov_h)
            scan.angle_min = -fov_rad / 2.0
            scan.angle_max = fov_rad / 2.0
            scan.angle_increment = fov_rad / w  # M?i pixel c?t là 1 tia laser
            scan.time_increment = 0.0
            scan.range_min = self.scan_range_min
            scan.range_max = self.scan_range_max
            
            # M?c ??nh toàn b? là 'inf' (Tr?ng)
            ranges = np.full(w, float('inf'), dtype=np.float32)

            # A. ??a d? li?u Line vào LaserScan
            # T?o m?t mask ?en ?? v? line lên ?ó
            line_mask = np.zeros((h - roi_start, w), dtype=np.uint8)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    # V? line lên mask (dày 2px ?? ch?c ch?n b?t ???c)
                    cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
            
            # Quét t?ng c?t c?a mask ?? tìm ?i?m "g?n nh?t" (y l?n nh?t)
            # (Cách t?i ?u: dùng numpy argmax)
            # L?t ng??c mask theo tr?c d?c ?? tìm ?i?m ??u tiên g?p t? d??i lên
            flipped_mask = line_mask[::-1, :] 
            
            # Tìm các c?t có ?i?m tr?ng
            has_obstacle = np.any(flipped_mask > 0, axis=0)
            
            # L?y index c?a ?i?m tr?ng ??u tiên (t? d??i lên)
            obstacle_indices = np.argmax(flipped_mask > 0, axis=0)
            
            # Tính toán kho?ng cách cho các c?t có v?t c?n
            for x in range(w):
                if has_obstacle[x]:
                    # To? ?? y trong ?nh g?c = (h - roi_start) - index_??o_ng??c + roi_start
                    # ??n gi?n hoá: y trong ROI (ch?a ??o) = (h_roi) - 1 - index_??o
                    y_in_roi = (h - roi_start) - 1 - obstacle_indices[x]
                    y_global = y_in_roi + roi_start
                    
                    dist = self.pixel_to_distance(y_global, h)
                    
                    # Gi?i h?n range
                    if self.scan_range_min < dist < self.scan_range_max:
                        ranges[x] = dist

            # B. ??a d? li?u YOLO (V?t c?n nguy hi?m) vào LaserScan
            for obj in self.last_detections:
                label = obj['label']
                # N?u là v?t c?n (không ph?i cây, ho?c cây b? b?nh tu? logic c?a b?n)
                if label != "plant": 
                    bx, by, bw, bh = obj['box']
                    
                    # Tính kho?ng cách ??c l??ng ??n chân v?t th? (?áy box)
                    dist_obj = self.pixel_to_distance(by + bh, h)
                    
                    # N?u v?t ? quá xa (>4m) ho?c quá g?n (<0.1m) thì b? qua
                    if dist_obj > self.scan_range_max or dist_obj < self.scan_range_min:
                        continue

                    # Gán giá tr? kho?ng cách này cho toàn b? góc nhìn (b? ngang) mà v?t th? chi?m
                    start_x = max(0, bx)
                    end_x = min(w, bx + bw)
                    
                    # ??t "t??ng ?o" ? v? trí v?t th?
                    ranges[start_x:end_x] = np.minimum(ranges[start_x:end_x], dist_obj)
                    
                    # Visual debug
                    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0,0,255), 2)

            # Convert numpy array sang list cho LaserScan message
            scan.ranges = ranges.tolist()
            self.scan_pub.publish(scan)

            # --- DEBUG VIEW ---
            # V? các ???ng Laser tìm ???c lên hình ?? ki?m tra
            # (Ch? v? vài tia ??i di?n ?? ?? lag)
            for i in range(0, w, 20):
                r = ranges[i]
                if r < self.scan_range_max:
                    # Tính ng??c l?i to? ?? pixel ?? v? (ch? mang tính minh ho?)
                    # ?ây là debug tr?c quan xem tia laser ?ang "ch?m" vào ?âu
                    pass 

            cv2.imshow("AgriVision Bridge", frame)
            # cv2.imshow("Line Mask", line_mask) # B?t cái này n?u mu?n xem mask tr?ng ?en
            
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        node = AgriVisionBridge()
        node.run()
    except rospy.ROSInterruptException:
        pass
