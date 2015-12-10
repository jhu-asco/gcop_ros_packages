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

//ROS dynamic reconfigure
#include <dynamic_reconfigure/server.h>
#include <gcop_ctrl/GcarSimConfig.h>

// ROS standard messages
#include <std_msgs/String.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>

//gcop_comm msgs
#include <gcop_comm/GcarCtrl.h>

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
  gcop_ctrl::GcarSimConfig config_;
  dynamic_reconfigure::Server<gcop_ctrl::GcarSimConfig> dyn_server_;

  bool debug_on_;

  string strtop_odom_,strtop_ctrl_;

  ros::Publisher pub_odom_;
  ros::Subscriber sub_ctrl_;
  ros::Timer timer_dynamics_;

  tf::TransformBroadcaster tf_br_;
  tf::TransformListener tf_lr_;

  double state_x_,state_y_,state_a_;
  double ctrl_vel_,ctrl_phi_;

  gcop_comm::GcarCtrl msg_ctrl_;
  ros::Duration durn_execution_delay_;
  bool new_traj_, any_traj_;

  int i_;

public:
  void sendCtrl(void);
private:
  void cbReconfig(gcop_ctrl::GcarSimConfig &config, uint32_t level);

  void setupTopicsAndNames(void);
  void initSubsPubsAndTimers(void);

  void cbTimerDynamics(const ros::TimerEvent& event);
  void cbCtrl(const gcop_comm::GcarCtrlConstPtr& msg_ctrl);

};

CallBackGcarSim::CallBackGcarSim():
                        nh_p_("~"),
                        loop_rate_main_(1000),
                        durn_execution_delay_(0),
                        new_traj_(false),any_traj_(false),
                        ctrl_phi_(0),ctrl_vel_(0)
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

  //setup dynamic reconfigure
  dynamic_reconfigure::Server<gcop_ctrl::GcarSimConfig>::CallbackType dyn_cb_f;
  dyn_cb_f = boost::bind(&CallBackGcarSim::cbReconfig, this, _1, _2);
  dyn_server_.setCallback(dyn_cb_f);

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
CallBackGcarSim::cbReconfig(gcop_ctrl::GcarSimConfig &config, uint32_t level)
{
  static bool first_time=true;

  if(!first_time)
  {
    //loop rate setting
    if(config_.dyn_loop_rate_main != config.dyn_loop_rate_main)
      loop_rate_main_= ros::Rate(config.dyn_loop_rate_main);

    if(config.dyn_update_state)
    {
      config.dyn_update_state = false;
      state_x_=config.dyn_state_x;
      state_y_=config.dyn_state_y;
      state_a_=config.dyn_state_a;
    }
  }
  else
  {
    cout<<"First time in reconfig. Setting config from yaml"<<endl;

    //general settings
    config.dyn_debug_on           = yaml_node_["debug_on"].as<bool>();
    config.dyn_loop_rate_main     = yaml_node_["loop_rate_main"].as<double>();
    loop_rate_main_= ros::Rate(config.dyn_loop_rate_main);

    config.dyn_state_x            = state_x_;
    config.dyn_state_y            = state_y_;
    config.dyn_state_a            = state_a_;

    first_time = false;
  }
  config_ = config;
}

void
CallBackGcarSim::setupFromYaml(void)
{
  debug_on_ = yaml_node_["debug_on"].as<bool>();
  Vector3d pose_3d = yaml_node_["pose2d"].as<Vector3d>();
  state_x_=pose_3d(0);
  state_y_=pose_3d(1);
  state_a_=pose_3d(2);
}

void
CallBackGcarSim::setupTopicsAndNames(void)
{
  if(debug_on_)
    cout<<"setting up topic names"<<endl;

  // Input Topics
  strtop_ctrl_ = yaml_node_["strtop_ctrl"].as<string>();

  // output Topics
  strtop_odom_ = yaml_node_["strtop_odom"].as<string>();

  if(debug_on_)
  {
    cout<<"Topics are(put here):"<<endl;
    cout<<"strtop_ctrl:"<<strtop_ctrl_<<endl;
    cout<<"strtop_odom:"<<strtop_odom_<<endl;
  }
}

void
CallBackGcarSim::initSubsPubsAndTimers(void)
{
  //Setup subscribers
  sub_ctrl_ = nh_.subscribe(strtop_ctrl_,1, &CallBackGcarSim::cbCtrl,this);

  //Setup Publishers
  pub_odom_ = nh_.advertise<nav_msgs::Odometry>( strtop_odom_, 0 );

  //Setup timers
  timer_dynamics_ = nh_.createTimer(ros::Duration(0.01), &CallBackGcarSim::cbTimerDynamics, this);
  timer_dynamics_.start();
}

void
CallBackGcarSim::cbTimerDynamics(const ros::TimerEvent& event)
{
  double x=state_x_;
  double y=state_y_;
  double a=state_a_;

  //Update state with input controls
  double dt = 0.01;
  double l=0.5;
  double u = ctrl_vel_;
  double w = u*tan(ctrl_phi_)/l;

  if(w>1e-10)
  {
    x = x + sin(a+w*dt)*u/w - sin(a)*u/w;
    y = y - cos(a+w*dt)*u/w + cos(a)*u/w;
  }
  else
  {
    x = x + cos(a)*u*dt;
    y = y + sin(a)*u*dt;
  }
  a = a + w*dt;

  state_x_=x;
  state_y_=y;
  state_a_=a;

  //send the updated state back as odom message
  nav_msgs::Odometry msg_odom;
  msg_odom.header.stamp=ros::Time::now();
  msg_odom.pose.pose.position.x = x;
  msg_odom.pose.pose.position.y = y;
  msg_odom.pose.pose.position.z = 0;
  msg_odom.pose.pose.orientation.w = cos(a/2);
  msg_odom.pose.pose.orientation.x = 0;
  msg_odom.pose.pose.orientation.y = 0;
  msg_odom.pose.pose.orientation.z = sin(a/2);
  msg_odom.twist.twist.linear.x = u;
  msg_odom.child_frame_id="/rampage";
  msg_odom.header.frame_id="/world";

  pub_odom_.publish(msg_odom);

}
void
CallBackGcarSim::cbCtrl(const gcop_comm::GcarCtrlConstPtr& p_msg_ctrl)
{
  msg_ctrl_ = *p_msg_ctrl;
  if(msg_ctrl_.ts_eph.size()>1)
  {
    new_traj_ = true;
    any_traj_ = true;
  }
}

void
CallBackGcarSim::sendCtrl(void)
{
  vector<ros::Time>::iterator it_lb;
  ros::Time t_now;
  ros::Time t_execution;
  if(any_traj_)
  {
    ros::Time t_now = ros::Time::now();
    ros::Time t_execution = t_now + durn_execution_delay_;
    if(new_traj_)
    {
      it_lb = lower_bound(msg_ctrl_.ts_eph.begin(),msg_ctrl_.ts_eph.end(),t_execution);
      i_ =it_lb - msg_ctrl_.ts_eph.begin();
      if(i_==msg_ctrl_.ts_eph.size())
      {
        cout<<"The execution time is past every element of ctrl traj. Increase time horizon"<<endl;
        any_traj_ = false; //because there is no valid traj
        //rampageCmdMotorsSI(*g_p_pub_uav_cmds,0,0);
        ctrl_phi_=0;
        ctrl_vel_=0;
      }
      new_traj_ =false;
    }

    if(t_execution>=msg_ctrl_.ts_eph[i_])
    {
      //rampageCmdMotorsSI(*g_p_pub_uav_cmds,g_msg_ctrl.us_vel[i],g_msg_ctrl.us_phi[i]);
      ctrl_phi_=msg_ctrl_.us_phi[i_];
      ctrl_vel_=msg_ctrl_.us_vel[i_];
      i_++;
      if(i_==msg_ctrl_.ts_eph.size())//reached end of traj
      {
        cout<<"All ctrl have been sent waiting for new ctrls"<<endl;
        any_traj_ = false; //because there is no valid traj anymore
        //rampageCmdMotorsSI(*g_p_pub_uav_cmds,0,0);
        ctrl_phi_=0;
        ctrl_vel_=0;
      }
    }
  }
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
    cbc.sendCtrl();
    ros::spinOnce();
    cbc.loop_rate_main_.sleep();
  }
  return 0;
}




