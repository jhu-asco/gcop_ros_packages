/*
 * gcar_sim.cpp
 *
 *  Created on: Dec 8, 2015
 *      Author: subhransu
 */


//ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

// ROS standard messages
#include <std_msgs/String.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>

//gcop_comm msgs
#include <gcop_comm/State.h>
#include <gcop_comm/CtrlTraj.h>
#include <gcop_comm/Trajectory_req.h>

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

//gcop include
#include <gcop/so3.h>
#include <gcop/lqcost.h>
#include <gcop/gcar.h>
#include <gcop/utils.h>
#include <gcop/se2.h>
#include <gcop/ddp.h>

//yaml
#include <yaml-cpp/yaml.h>

//local includes
#include <gcop_ros_utils/eigen_ros_conv.h>
#include <gcop_ros_utils/eig_splinterp.h>
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


void mySigIntHandler(int signal)
{
  g_shutdown_requested=1;
}

void timer_start(struct timeval *time)
{
  gettimeofday(time,(struct timezone*)0);
}

long timer_us(struct timeval *time)
{
  struct timeval now;
  gettimeofday(&now,(struct timezone*)0);
  return 1000000*(now.tv_sec - time->tv_sec) + now.tv_usec - time->tv_usec;
}

template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

//------------------------------------------------------------------------
//-----------------------------CALLBACK CLASS ----------------------------
//------------------------------------------------------------------------

class CallBackGcarSim
{
public:
  typedef Transform<double,2,Affine> Transform2d;

public:
  CallBackGcarSim();
  ~CallBackGcarSim();

  void setupFromYaml(void);

public:
  ros::Rate loop_rate_main_;

private:
  ros::NodeHandle nh_, nh_p_;
  YAML::Node yaml_node_;

  bool debug_on_;

  string strtop_odom_,strtop_ctrl_;

  ros::Publisher sub_odom_;
  ros::Subscriber pub_ctrl_;
  ros::Timer timer_vis_;

  tf::TransformBroadcaster tf_br_;
  tf::TransformListener tf_lr_;

  Gcar sys_gcar_;

private:

  void setupTopicsAndNames(void);
  void initSubsPubsAndTimers(void);

};

CallBackGcarSim::CallBackGcarSim():
                        nh_p_("~"),
                        loop_rate_main_(1000),
                        sys_gcar_()
{
  cout<<"**************************************************************************"<<endl;
  cout<<"**********************GCAR SIMULATOR FOR DSL-DDP-PLANNER******************"<<endl;
  cout<<"*Entering constructor of cbc"<<endl;

  //Setup YAML reading and parsing
  string strfile_params;nh_p_.getParam("strfile_params",strfile_params);
  cout<<"loading yaml param file into yaml_node"<<endl;
  yaml_node_ = YAML::LoadFile(strfile_params);

  // Setup general settings from yaml
  setupFromYaml();

  // Setup topic names
  setupTopicsAndNames();
  cout<<"Setup topic names from yaml file done"<<endl;

  //Setup Subscriber, publishers and Timers
  initSubsPubsAndTimers();
  cout<<"Initialized publishers, subscriber and timers"<<endl;
}


CallBackGcarSim::~CallBackGcarSim()
{

}

void
CallBackGcarSim::setupFromYaml(void)
{
  debug_on_ = yaml_node_["debug_on"].as<bool>();
}

void
CallBackGcarSim::setupTopicsAndNames(void)
{
  if(debug_on_)
    cout<<"setting up topic names"<<endl;

  // Input Topics


  // output Topics


  if(debug_on_)
  {
    cout<<"Topics are(put here):"<<endl;
  }
}

void
CallBackGcarSim::initSubsPubsAndTimers(void)
{
  //Setup subscribers

  //Setup Publishers

  //Setup timers

}


//------------------------------------------------------------------------
//--------------------------------MAIN -----------------------------------
//------------------------------------------------------------------------

int main(int argc, char** argv)
{
  ros::init(argc,argv,"gcar_sim",ros::init_options::NoSigintHandler);
  signal(SIGINT,mySigIntHandler);

  ros::NodeHandle nh;

  CallBackGcarSim cbc;

  while(!g_shutdown_requested)
  {
    ros::spinOnce();
    cbc.loop_rate_main_.sleep();
  }
  return 0;
}




