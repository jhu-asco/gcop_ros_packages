cmake_minimum_required(VERSION 2.8.3)
project(gcop_ros_bullet)

add_definitions(-g -std=c++0x)#DEBUG
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  tf
	tf2_ros
  geometry_msgs
  roscpp
  rospy
  std_msgs
	visualization_msgs
	sensor_msgs
  gcop_comm
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

find_package(GCOP REQUIRED algos bulletsystems systems)
include_directories(${GCOP_INCLUDE_DIRS})
set(LIBS ${LIBS} ${GCOP_LIBRARIES})

find_package(TinyXML REQUIRED)
include_directories(${TINYXML_INCLUDE_DIR})
set(LIBS ${LIBS} ${TINYXML_LIBRARIES})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Bullet REQUIRED)
include_directories(${BULLET_INCLUDE_DIR})
set(LIBS ${LIBS} ${BULLET_LIBRARIES})

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBT_USE_DOUBLE_PRECISION")#Compiling Bullet with double precision turned on

find_package(OpenMP)

if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

message("GCOP LIBS: ${GCOP_LIBRARIES}")
## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   geometry_msgs#   std_msgs
# )

generate_dynamic_reconfigure_options(
  cfg/paramedit.cfg
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES gcop_ros_bullet
#  CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Declare a cpp library
# add_library(gcop_ros_bullet
#   src/${PROJECT_NAME}/gcop_ros_bullet.cpp
# )

## Declare a cpp executable
# add_executable(gcop_ros_bullet_node src/gcop_ros_bullet_node.cpp)
add_executable(cebulletcartest src/cebulletcartest.cc)
target_link_libraries(cebulletcartest ${LIBS} ${catkin_LIBRARIES})
add_dependencies(cebulletcartest ${PROJECT_NAME}_gencfg gcop_comm_generate_messages_cpp)

#add_executable(cebulletcartest1 src/cebulletcartest1.cc)
#target_link_libraries(cebulletcartest1 ${LIBS} ${catkin_LIBRARIES})
#add_dependencies(cebulletcartest1 ${PROJECT_NAME}_gencfg gcop_comm_generate_messages_cpp)

add_executable(gnbulletcartest src/gnbulletcartest.cc)
target_link_libraries(gnbulletcartest ${LIBS} ${catkin_LIBRARIES})
add_dependencies(gnbulletcartest ${PROJECT_NAME}_gencfg gcop_comm_generate_messages_cpp)

#add_executable(gnbulletcartest1 src/gnbulletcartest1.cc)
#target_link_libraries(gnbulletcartest1 ${LIBS} ${catkin_LIBRARIES})
#add_dependencies(gnbulletcartest1 ${PROJECT_NAME}_gencfg gcop_comm_generate_messages_cpp)

add_executable(spsabulletcartest src/spsabulletcartest.cc)
target_link_libraries(spsabulletcartest ${LIBS} ${catkin_LIBRARIES})
add_dependencies(spsabulletcartest ${PROJECT_NAME}_gencfg gcop_comm_generate_messages_cpp)

add_executable(hybrid_ceandgn_bulletcartest src/hybrid_ceandgn_bulletcartest.cc)
target_link_libraries(hybrid_ceandgn_bulletcartest ${LIBS} ${catkin_LIBRARIES})
add_dependencies(hybrid_ceandgn_bulletcartest ${PROJECT_NAME}_gencfg gcop_comm_generate_messages_cpp)

#add_executable(hybrid_ceandgn_bulletcartest1 src/hybrid_ceandgn_bulletcartest1.cc)
#target_link_libraries(hybrid_ceandgn_bulletcartest1 ${LIBS} ${catkin_LIBRARIES})
#add_dependencies(hybrid_ceandgn_bulletcartest1 ${PROJECT_NAME}_gencfg gcop_comm_generate_messages_cpp)

add_executable(hybrid_ceandgn_bulletcartest2 src/hybrid_ceandgn_bulletcartest2.cc)
target_link_libraries(hybrid_ceandgn_bulletcartest2 ${LIBS} ${catkin_LIBRARIES})
add_dependencies(hybrid_ceandgn_bulletcartest2 ${PROJECT_NAME}_gencfg gcop_comm_generate_messages_cpp)

add_executable(hybrid_ceandgn_feedbacktest src/hybrid_ceandgn_feedbacktest.cc)
target_link_libraries(hybrid_ceandgn_feedbacktest ${LIBS} ${catkin_LIBRARIES})
add_dependencies(hybrid_ceandgn_feedbacktest ${PROJECT_NAME}_gencfg gcop_comm_generate_messages_cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(gcop_ros_bullet_node gcop_ros_bullet_generate_messages_cpp)

## Specify libraries to link a library or executable target against
# target_link_libraries(gcop_ros_bullet_node
#   ${catkin_LIBRARIES}
# )

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS gcop_ros_bullet gcop_ros_bullet_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_gcop_ros_bullet.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
