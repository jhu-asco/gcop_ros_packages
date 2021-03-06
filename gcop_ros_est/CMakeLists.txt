cmake_minimum_required(VERSION 2.8.3)
project(gcop_ros_est)

set(CMAKE_BUILD_TYPE Release)
add_definitions(-g -std=c++0x -pthread )

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
camera_calibration_parsers 
cv_bridge 
image_transport 
nodelet 
roscpp 
std_msgs
sensor_msgs 
geometry_msgs
nav_msgs
tf
gps_common
dynamic_reconfigure
visualization_msgs
message_generation
enu
gcop_ros_utils
resource_retriever
)


generate_dynamic_reconfigure_options(
  cfg/Insekf.cfg
  cfg/SimSens.cfg
  cfg/Calibrator.cfg
  )
## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(GCOP REQUIRED)
find_package(TinyXML REQUIRED)
set(GCOP_LIBRARIES ${GCOP_LIBRARIES} ${TINYXML_LIBRARIES})
find_package(OpenCV REQUIRED core highgui imgproc calib3d)
find_package(Eigen3 REQUIRED)
#find_package(GLU REQUIRED)
#find_package(GLUT REQUIRED)
find_library(SWIFTNAV swiftnav)

pkg_check_modules(YAML_CPP yaml-cpp)
if(${YAML_CPP_VERSION} VERSION_GREATER 0.5)
    add_definitions(-DHAVE_NEW_YAMLCPP)
endif()
link_directories(${YAML_CPP_LIBRARY_DIRS})
include_directories(${YAML_CPP_INCLUDE_DIRS})

find_package(OpenCV REQUIRED COMPONENTS highgui imgproc )
include(${OpenCV_INCLUDE_DIRS})
set(LIBS ${LIBS} ${OpenCV_LIBRARIES})

add_message_files(
    FILES
    InsekfDiag.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
 )
 
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
#   std_msgs  # Or other packages containing msgs
# )

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
#  LIBRARIES gcop_est
#  CATKIN_DEPENDS camera_calibration_parsers image_transport nodelet roscpp tf
#  DEPENDS system_lib
CATKIN_DEPENDS message_runtime
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)

include_directories(
include 
${catkin_INCLUDE_DIRS} 
${TINYXML_INCLUDE_DIR} 
${GCOP_INCLUDE_DIRS} 
${OpenCV_INCLUDE_DIRS} 
${EIGEN3_INCLUDE_DIR} 
${CV_BRIDGE_INCLUDE_DIRS}
${Boost_INCLUDE_DIRS}
)

## Declare a cpp library
# add_library(gcop_est
#   src/${PROJECT_NAME}/gcop_est.cpp
# )
#add_library(cam_est src/nodelets/estimator1.cpp)
#target_link_libraries(cam_est ${GCOP_LIBRARIES} ${OpenCV_LIBS} ${catkin_LIBRARIES})

## Declare a cpp executable
# add_executable(gcop_est_node src/gcop_est_node.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(gcop_est_node gcop_est_generate_messages_cpp)

## Specify libraries to link a library or executable target against
# target_link_libraries(gcop_est_node
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
# install(TARGETS gcop_est gcop_est_node
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
# catkin_add_gtest(${PROJECT_NAME}-test test/test_gcop_est.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)

#add_executable(obstacle_detection_lidar src/obstacle_detection_lidar.cpp)
#target_link_libraries(obstacle_detection_lidar ${catkin_LIBRARIES} ${Boost_LIBRARIES})

add_executable(satmap_server src/satmap_server.cpp)
target_link_libraries(satmap_server ${catkin_LIBRARIES} ${Boost_LIBRARIES})

add_executable(simsens src/simsens.cpp)
target_link_libraries(simsens 
		              ${GCOP_LIBRARIES}
	                  ${TINYXML_LIBRARIES}
	                  ${catkin_LIBRARIES}
	      	          ${SWIFTNAV})
add_dependencies(simsens ${PROJECT_NAME}_gencfg )

add_executable(insekf src/insekf.cpp)
target_link_libraries(insekf 
		              ${GCOP_LIBRARIES}
	                  ${TINYXML_LIBRARIES}
	                  ${catkin_LIBRARIES}
	                  ${SWIFTNAV}
	                  ${YAML_CPP_LIBRARIES} )
add_dependencies(insekf ${PROJECT_NAME}_gencfg )

add_executable(insekf_artags src/insekf_artags.cpp)
target_link_libraries(insekf_artags 
		              ${GCOP_LIBRARIES}
	                  ${TINYXML_LIBRARIES}
	                  ${catkin_LIBRARIES}
	                  ${SWIFTNAV}
	                  ${YAML_CPP_LIBRARIES} )
add_dependencies(insekf_artags ${PROJECT_NAME}_gencfg )

add_executable(calibrator src/calibrator.cpp)
target_link_libraries(calibrator 
		      ${GCOP_LIBRARIES}
		      ${OpenCV_LIBRARIES}
	      ${TINYXML_LIBRARIES}
	      ${catkin_LIBRARIES})
add_dependencies(calibrator ${PROJECT_NAME}_gencfg )

add_executable(mbsforceest src/mbsforceest.cc)
target_link_libraries(mbsforceest 
                      ${GCOP_LIBRARIES}
                      ${TINYXML_LIBRARIES}
                      ${catkin_LIBRARIES})
add_dependencies(mbsforceest gcop_comm_generate_messages_cpp)
