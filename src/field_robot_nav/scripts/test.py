#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy as np
import math
import yaml
import os
import heapq
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from tf.transformations import euler_from_quaternion

# ==========================================
# 1. CLASS: A* PATH PLANNER
# ==========================================
class PathPlanner:
    def __init__(self, yaml_path, robot_radius=0.3):
        self.yaml_path = yaml_path
        self.robot_radius = robot_radius
        self.map_grid = None
        self.resolution = 0.05
        self.origin = [0.0, 0.0, 0.0]
        self.height = 0
        self.width = 0
        self.load_map()

    def load_map(self):
        if not os.path.exists(self.yaml_path):
            rospy.logerr("Lỗi: Không tìm thấy file map!")
            return

        with open(self.yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.resolution = config['resolution']
        self.origin = config['origin']
        img_path = os.path.join(os.path.dirname(self.yaml_path), config['image'])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Inflation (Phình vật cản)
        radius_px = int(math.ceil(self.robot_radius / self.resolution))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius_px*2, radius_px*2))
        inflated_img = cv2.erode(img, kernel, iterations=1)
        
        self.height, self.width = inflated_img.shape
        self.map_grid = np.zeros_like(inflated_img)
        self.map_grid[inflated_img < 250] = 1 # 1 là vật cản

    def world_to_map(self, wx, wy):
        mx = int((wx - self.origin[0]) / self.resolution)
        my = int((wy - self.origin[1]) / self.resolution)
        my = self.height - 1 - my 
        return mx, my

    def map_to_world(self, mx, my):
        inv_my = self.height - 1 - my
        wx = (mx * self.resolution) + self.origin[0]
        wy = (inv_my * self.resolution) + self.origin[1]
        return wx, wy

    def get_neighbors(self, node):
        x, y = node
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1), 
                     (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)]
        valid = []
        for nx, ny in neighbors:
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.map_grid[ny][nx] == 0:
                    cost = math.sqrt((nx-x)**2 + (ny-y)**2)
                    valid.append(((nx, ny), cost))
        return valid

    def run_a_star(self, start_world, goal_world):
        start_node = self.world_to_map(*start_world)
        goal_node = self.world_to_map(*goal_world)
        print("DEBUG: Map Size = {} x {}".format(self.width, self.height))
        print("DEBUG: Robot Pixel = {}".format(start_node))
        # Kiểm tra điểm trong vật cản
        if not (0 <= start_node[0] < self.width and 0 <= start_node[1] < self.height):
            rospy.logwarn("Vị trí xe nằm ngoài bản đồ!")
            return None
            
        if self.map_grid[goal_node[1]][goal_node[0]] == 1:
            rospy.logwarn("Lỗi: Điểm đích nằm trong tường!")
            return None

        frontier = []
        heapq.heappush(frontier, (0, start_node))
        came_from = {start_node: None}
        cost_so_far = {start_node: 0}
        
        path_found = False
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal_node:
                path_found = True
                break
            
            for next_node, cost in self.get_neighbors(current):
                new_cost = cost_so_far[current] + cost
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + math.sqrt((goal_node[0]-next_node[0])**2 + (goal_node[1]-next_node[1])**2)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current
        
        if not path_found:
            rospy.logwarn("Không tìm thấy đường đi!")
            return None

        path = []
        curr = goal_node
        while curr != start_node:
            path.append(self.map_to_world(*curr))
            curr = came_from[curr]
        path.append(self.map_to_world(*start_node))
        path.reverse()
        return path

# ==========================================
# 2. CLASS: ROS NAVIGATOR (Kết nối RViz & AMCL)
# ==========================================
class RvizNavigator:
    def __init__(self, planner):
        rospy.init_node('rviz_navigator_node')
        
        self.planner = planner
        self.is_moving = False
        
        # Vị trí hiện tại (từ AMCL)
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.has_pose = False

        # Publisher điều khiển xe
        self.pub_vel = rospy.Publisher('/cmd_vel_nav', Twist, queue_size=10)
        
        # 1. Lắng nghe vị trí từ AMCL
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_callback)
        
        # 2. Lắng nghe điểm đích từ RViz (Nút 2D Nav Goal)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)

        rospy.loginfo("--- HE THONG SAN SANG ---")
        rospy.loginfo("Hay dung RViz, chon nut '2D Nav Goal' de ra lenh!")

    def amcl_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        rot = msg.pose.pose.orientation
        (_, _, self.current_theta) = euler_from_quaternion([rot.x, rot.y, rot.z, rot.w])
        self.has_pose = True

    def goal_callback(self, msg):
        """Hàm này chạy khi bạn click chuột trên RViz"""
        if not self.has_pose:
            rospy.logwarn("Chưa có định vị (AMCL), không thể chạy!")
            return
            
        if self.is_moving:
            rospy.logwarn("Xe đang chạy, nhận lệnh mới -> Dừng lệnh cũ, tính toán lại...")
            self.is_moving = False # Dừng thread cũ nếu có (ở code đơn giản này ta chấp nhận chạy đè)

        target_x = msg.pose.position.x
        target_y = msg.pose.position.y
        
        rospy.loginfo("Nhan lenh tu RViz: Den toa do x={:.2f}, y={:.2f}".format(target_x, target_y))
        
        # Tính toán A*
        start_pos = (self.current_x, self.current_y)
        goal_pos = (target_x, target_y)
        
        path = self.planner.run_a_star(start_pos, goal_pos)
        
        if path:
            rospy.loginfo("Da tim thay duong di: {} diem. Bat dau chay!".format(len(path)))
            self.execute_path(path)

    def execute_path(self, path):
        self.is_moving = True
        rate = rospy.Rate(10)
        
        # PID Parameters
        kp_linear = 0.5
        kp_angular = 1.2
        max_speed = 0.25

        for point in path:
            target_x, target_y = point
            
            while not rospy.is_shutdown():
                # Check nếu có lệnh mới đè vào (Logic nâng cao, ở đây chạy đơn giản)
                # ...
                
                # Cập nhật khoảng cách
                dist = math.sqrt((target_x - self.current_x)**2 + (target_y - self.current_y)**2)
                
                if dist < 0.15: # Đến gần điểm waypoint
                    break
                
                # Tính góc lái
                angle_to_goal = math.atan2(target_y - self.current_y, target_x - self.current_x)
                angle_err = angle_to_goal - self.current_theta
                
                # Chuẩn hóa góc
                while angle_err > math.pi: angle_err -= 2*math.pi
                while angle_err < -math.pi: angle_err += 2*math.pi
                
                twist = Twist()
                
                # Logic lái: Nếu lệch nhiều thì quay tại chỗ trước
                if abs(angle_err) > 0.4:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.5 * angle_err
                    # Limit tốc độ quay
                    twist.angular.z = max(min(twist.angular.z, 0.5), -0.5)
                else:
                    twist.linear.x = min(kp_linear * dist, max_speed)
                    twist.angular.z = kp_angular * angle_err
                
                self.pub_vel.publish(twist)
                rate.sleep()

        # Đến đích cuối cùng
        self.is_moving = False
        self.pub_vel.publish(Twist()) # Dừng xe
        rospy.loginfo("DA DEN DICH! Cho lenh tiep theo...")

# ==========================================
# 3. MAIN
# ==========================================
if __name__ == "__main__":
    # Thay duong dan map cua ban
    yaml_file = "/home/chunne/test/src/field_robot_slam/maps/farm_map.yaml"
    
    try:
        planner = PathPlanner(yaml_file, robot_radius=0.4)
        navigator = RvizNavigator(planner)
        rospy.spin() # Giữ chương trình chạy mãi để lắng nghe RViz
    except rospy.ROSInterruptException:
        pass
