/*
 * test.cpp
 *
 *  Created on: Dec 12, 2015
 *      Author: subhransu
 */
//ROS
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

// ROS standard messages
#include <std_msgs/String.h>
#include <visualization_msgs/Marker.h>

//ROS dynamic reconfigure
#include <dynamic_reconfigure/server.h>
//#include <test_node/TestNodeConfig.h>

//Other includes
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <iterator>

//yaml
#include <yaml-cpp/yaml.h>

//local includes
#include <gcop_ros_utils/eigen_ros_conv.h>
#include <gcop_ros_utils/yaml_eig_conv.h>

// Eigen Includes
#include <Eigen/Dense>
#include <Eigen/Geometry>

//-------------------------------------------------------------------------
//-----------------------NAME SPACES ---------------------------------
//-------------------------------------------------------------------------
using namespace std;
using namespace Eigen;


//-------------------------------------------------------------------------
//-----------------------GLOBAL VARIABLES ---------------------------------
//-------------------------------------------------------------------------
sig_atomic_t g_shutdown_requested=0;


//------------------------------------------------------------------------
//-----------------------FUNCTION DEFINITIONS ----------------------------
//------------------------------------------------------------------------


void mySigIntHandler(int signal){
  g_shutdown_requested=1;
}

void timer_start(struct timeval *time){
  gettimeofday(time,(struct timezone*)0);
}

long timer_us(struct timeval *time){
  struct timeval now;
  gettimeofday(&now,(struct timezone*)0);
  return 1000000*(now.tv_sec - time->tv_sec) + now.tv_usec - time->tv_usec;
}

template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}
typedef Transform<double,2,Affine> Transform2d;

//------------------------------------------------------------------------
//--------------------------------MAIN -----------------------------------
//------------------------------------------------------------------------

int main(int argc, char** argv){

  Affine3d igrid_to_mgrid = Scaling(Vector3d(1/0.1, 1/0.2, 1/0.5))* Translation3d(Vector3d(1, 1, 1));
  Vector3d p = igrid_to_mgrid* Vector3d(1,1,0);

  cout<<"p:\n"<<p.transpose()<<endl;
  return 0;
}



