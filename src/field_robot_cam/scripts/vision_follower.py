#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import os
import time
import csv
from datetime import datetime
from geometry_msgs.msg import Twist

# ========== C?U HÌNH YOLO ==========
# D?i màu xanh lá ?? xác ??nh cây kh?e (H: 35-85 là vùng màu xanh lá trong OpenCV)
LOWER_HEALTHY = np.array([35, 40, 40], dtype=np.uint8) 
UPPER_HEALTHY = np.array([85, 255, 255], dtype=np.uint8)

# ========== CÁC HÀM H? TR? ==========
steering_history = []
HISTORY_LENGTH = 10

def smooth_steering(new_value):
    global steering_history
    steering_history.append(new_value)
    if len(steering_history) > HISTORY_LENGTH:
        steering_history.pop(0)
    return sum(steering_history) / len(steering_history)

def bird_eye_view(img):
    """
    Chuy?n ??i ?nh sang góc nhìn t? trên xu?ng (Bird's Eye View)
    """
    h, w = img.shape[:2]
    # === C?U HÌNH ?I?M C?T (CALIBRATION) ===
    src_points = np.float32([
    	[w * 0.25, h * 0.65],  # Tăng từ 0.55 lên 0.65 (Nhìn gần hơn, cắt bớt phần xa)
    	[w * 0.75, h * 0.65],  
    	[0, h * 0.9],         
    	[w, h * 0.9]          
    ])
    
    dst_points = np.float32([
        [w * 0.2, 0],        # Top-Left
        [w * 0.8, 0],        # Top-Right
        [w * 0.2, h],        # Bottom-Left
        [w * 0.8, h]         # Bottom-Right
    ])
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped, M

class AgriRobotController:
    def __init__(self):
        rospy.init_node("agri_lane_follower_bev")
        rospy.loginfo("--- AGRI ROBOT: STARTED ---")

        self.cmd_pub = rospy.Publisher("/cmd_vel_row", Twist, queue_size=1) 

        # ========= TUNING PARAMS =========
        self.max_speed = rospy.get_param("~max_speed", 0.2)
        self.base_speed = rospy.get_param("~base_speed", 0.15)
        self.kp = 0.001  
        self.kd = 0.001 
        self.last_error = 0

        # ========= QU?N LÝ TR?NG THÁI & LOGGING =========
        # --- CH?NH S?A ???NG D?N T?I ?ÂY ---
        # L?u file vào /home/robot/Desktop/plant_data.csv
        desktop_path = "/home/robot/Desktop"
        
        # Ki?m tra xem th? m?c Desktop có t?n t?i không ?? tránh l?i
        if not os.path.exists(desktop_path):
            rospy.logwarn(f"Thu muc {desktop_path} khong ton tai! Se luu tai thu muc hien tai.")
            self.csv_file = "plant_data.csv"
        else:
            self.csv_file = os.path.join(desktop_path, "plant_data.csv")

        rospy.loginfo(f"Data se duoc luu tai: {self.csv_file}")

        # T?o file CSV và ghi header n?u ch?a có
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Cay_STT", "Thoi_gian", "Suc_khoe"])
        
        self.plant_count = 0        # ??m s? cây ?ã g?p
        self.is_inspecting = False  # Tr?ng thái ?ang ki?m tra cây
        self.inspect_start_time = 0 # Th?i gian b?t ??u d?ng l?i
        self.inspect_duration = 3.0 # D?ng 3 giây ?? ki?m tra
        
        self.last_plant_time = 0    # ?? tránh log trùng l?p (cooldown)
        self.plant_cooldown = 5.0   # Sau 5 giây m?i ???c log cây ti?p theo

        # ========= YOLO CONFIG =========
        script_dir = os.path.dirname(os.path.realpath(__file__))
        weights_path = os.path.join(script_dir, "yolov4-tiny-merged_best.weights")
        cfg_path = os.path.join(script_dir, "yolov4-tiny-merged.cfg")
        names_path = os.path.join(script_dir, "coco.names") 

        self.frame_count = 0
        self.skip_yolo_frames = 3 
        self.last_detections = []   
        self.yolo_input_size = (320, 320)
        self.prev_time = 0

        rospy.loginfo(f"Load YOLO: {weights_path}")
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

        # ========= CAMERA CONFIG =========
        self.gst_str = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=640, height=360, format=BGRx ! " 
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink"
        )
        self.cap = cv2.VideoCapture(self.gst_str, cv2.CAP_GSTREAMER)

    # ================= HEALTH CHECK & LOGGING =================
    def check_plant_health(self, frame, box):
        """
        C?t vùng ?nh ch?a cây, ki?m tra màu s?c ?? xác ??nh s?c kh?e.
        """
        x, y, w, h = box
        # ??m b?o box n?m trong khung hình
        x = max(0, x); y = max(0, y)
        w = min(w, frame.shape[1] - x); h = min(h, frame.shape[0] - y)
        
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0: return "Unknown"

        # Chuy?n sang HSV ?? l?c màu xanh
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, LOWER_HEALTHY, UPPER_HEALTHY)
        
        # Tính t? l? pixel màu xanh / t?ng pixel trong box
        green_pixel_count = cv2.countNonZero(mask)
        total_pixels = w * h
        ratio = green_pixel_count / total_pixels

        # Ng??ng: N?u h?n 20% là màu xanh lá chu?n -> Kh?e, ng??c l?i -> B?nh
        if ratio > 0.2:
            return "Khoe" # Healthy
        else:
            return "Benh" # Sick/Diseased

    def log_data(self, health_status):
        self.plant_count += 1
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Format ghi: Cây(STT) + Th?i gian + S?c kh?e
        row_data = [f"Cay({self.plant_count})", timestamp, health_status]
        
        try:
            # M? file self.csv_file ?ã ???c set ???ng d?n tuy?t ??i
            with open(self.csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(row_data)
            rospy.loginfo(f"LOGGED to {self.csv_file}: {row_data}")
        except Exception as e:
            rospy.logerr(f"L?i ghi file CSV: {e}")

    # ================= YOLO FUNCTION =================
    def detect_objects(self, frame):
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
                if confidence > 0.5: 
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
                results.append({"box": boxes[i], "label": label, "conf": confidences[i]})
        return results

    # ================= MAIN LOOP =================
    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret: 
                rate.sleep()
                continue

            curr_time = time.time()
            fps = 1.0 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
            self.prev_time = curr_time
            self.frame_count += 1
            h, w = frame.shape[:2]

            # --- 1. YOLO DETECTION ---
            if self.frame_count % (self.skip_yolo_frames + 1) == 0:
                self.last_detections = self.detect_objects(frame)
            
            cmd = Twist()
            target_plant_box = None
            
            # Ki?m tra xem có cây nào ? g?n ?? inspect không
            # Ch? inspect n?u KHÔNG trong th?i gian cooldown
            can_inspect = (curr_time - self.last_plant_time) > self.plant_cooldown

            for obj in self.last_detections:
                x, y, bw, bh = obj['box']
                label = obj['label']
                
                # V? box YOLO (Visual)
                color = (0, 0, 255)
                if label == "plant": 
                    color = (0, 255, 0)
                    # N?u cây ?? to (g?n) và robot ???c phép inspect
                    if bh > h * 0.25 and can_inspect and not self.is_inspecting:
                        target_plant_box = obj['box']
                
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), color, 2)
                cv2.putText(frame, f"{label}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- 2. X? LÝ LOGIC INSPECT vs MOVE ---
            
            if self.is_inspecting:
                # TR?NG THÁI: ?ANG KI?M TRA
                # 1. D?ng robot
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                state_text = "INSPECTING..."

                # Ki?m tra th?i gian d?ng
                if curr_time - self.inspect_start_time > self.inspect_duration:
                    # ?ã d?ng ?? lâu -> ?i ti?p
                    self.is_inspecting = False
                    self.last_plant_time = curr_time # B?t ??u tính cooldown t? lúc ?i ti?p
                
                # Khi ?ang inspect thì KHÔNG v? line, KHÔNG tính toán lái
            
            elif target_plant_box is not None:
                # TR?NG THÁI: V?A PHÁT HI?N CÂY -> B?T ??U INSPECT
                self.is_inspecting = True
                self.inspect_start_time = curr_time
                
                # Th?c hi?n logic s?c kh?e & Log ngay lúc d?ng
                health_status = self.check_plant_health(frame, target_plant_box)
                self.log_data(health_status)
                
                # D?ng ngay l?p t?c
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                state_text = f"FOUND PLANT: {health_status}"

            else:
                # TR?NG THÁI: DI CHUY?N & BÁM LÀN (BÌNH TH??NG)
                # Ch? v? Line và tính toán lái khi KHÔNG inspect
                
                bev_img, _ = bird_eye_view(frame)
                gray = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (7, 7), 0)
                edges = cv2.Canny(blur, 30, 100)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 40, minLineLength=40, maxLineGap=80)
                
                left_lines = []
                right_lines = []
                mid_x = w // 2

                if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        if abs(x2 - x1) < abs(y2 - y1) * 0.5: 
                            avg_x = (x1 + x2) / 2
                            if avg_x < mid_x:
                                left_lines.append(avg_x)
                                cv2.line(bev_img, (x1, y1), (x2, y2), (255, 0, 0), 2) 
                            else:
                                right_lines.append(avg_x)
                                cv2.line(bev_img, (x1, y1), (x2, y2), (0, 0, 255), 2) 

                l_pos = np.mean(left_lines) if len(left_lines) > 0 else 0
                r_pos = np.mean(right_lines) if len(right_lines) > 0 else w
                
                current_lane_center = mid_x 
                has_lane = False

                if len(left_lines) > 0 and len(right_lines) > 0:
                    current_lane_center = (l_pos + r_pos) / 2
                    state_text = "Moving: Dual Lane"
                    has_lane = True
                elif len(left_lines) > 0:
                    current_lane_center = l_pos + (w * 0.3) 
                    state_text = "Moving: Left Only"
                    has_lane = True
                elif len(right_lines) > 0:
                    current_lane_center = r_pos - (w * 0.3)
                    state_text = "Moving: Right Only"
                    has_lane = True
                else:
                    state_text = "Moving: Search Lane"
                    has_lane = False

                if has_lane:
                    error = mid_x - current_lane_center
                    steering = (self.kp * error) + (self.kd * (error - self.last_error))
                    self.last_error = error
                    cmd.angular.z = smooth_steering(steering)
                    cmd.linear.x = self.base_speed
                    
                    # Visual BEV
                    cv2.circle(bev_img, (int(current_lane_center), h//2), 10, (0, 255, 255), -1)
                else:
                    cmd.linear.x = 0.05 # ?i ch?m dò ???ng
                    cmd.angular.z = 0.0

                # Hi?n th? BEV nh? khi ?ang di chuy?n
                bev_small = cv2.resize(bev_img, (200, 150))
                frame[0:150, 0:200] = bev_small 
                cv2.rectangle(frame, (0,0), (200,150), (255,255,0), 2)

            # Publish l?nh
            self.cmd_pub.publish(cmd)

            # --- HI?N TH? THÔNG TIN ---
            cv2.putText(frame, f"FPS: {fps:.1f}", (210, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Mode: {state_text}", (210, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if self.is_inspecting:
                # ??m ng??c th?i gian d?ng
                remaining = self.inspect_duration - (curr_time - self.inspect_start_time)
                cv2.putText(frame, f"Wait: {remaining:.1f}s", (210, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.imshow("AgriRobot Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        node = AgriRobotController()
        node.run()
    except rospy.ROSInterruptException:
        pass
