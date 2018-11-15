cmake_minimum_required(VERSION 2.8.3)
project(ros_face_recognition)

find_package(catkin REQUIRED COMPONENTS
   rospy
   std_msgs
   sensor_msgs
   message_generation
)

add_message_files(
  FILES
  Box.msg
)

add_service_files(
  FILES
  Face.srv
  Name.srv
  Detect.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package()