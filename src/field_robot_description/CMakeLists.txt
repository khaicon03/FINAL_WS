cmake_minimum_required(VERSION 3.0.2)
project(field_robot_description)

find_package(catkin REQUIRED COMPONENTS roscpp xacro urdf)
catkin_package()
include_directories(${catkin_INCLUDE_DIRS})
