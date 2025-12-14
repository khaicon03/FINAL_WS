#!/usr/bin/env python
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion
import math
import argparse
import sys

def send_goal(x, y, yaw=0.0):
    rospy.init_node('field_robot_send_goal')
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y

    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    goal.target_pose.pose.orientation = Quaternion(0.0, 0.0, qz, qw)

    client.send_goal(goal)
    client.wait_for_result()
    rospy.loginfo("Goal result: %s", client.get_state())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Send a move_base goal (map frame)")
    parser.add_argument("--x", type=float, default=2.0, help="Goal X in map frame (m)")
    parser.add_argument("--y", type=float, default=1.0, help="Goal Y in map frame (m)")
    parser.add_argument(
        "--yaw", type=float, default=0.0, help="Yaw in radians (counter-clockwise)"
    )
    args, _ = parser.parse_known_args(sys.argv[1:])

    try:
        send_goal(args.x, args.y, args.yaw)
    except rospy.ROSInterruptException:
        pass

