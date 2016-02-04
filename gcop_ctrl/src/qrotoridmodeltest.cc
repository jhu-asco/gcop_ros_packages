/** This is an example on how to control a qrotor model using GCOP Library.
 * Author: Gowtham Garimella
 */
#include "ros/ros.h"
#include <iomanip>
#include <iostream>
#include <dynamic_reconfigure/server.h>
#include "gcop_comm/CtrlTraj.h"//msg for publishing ctrl trajectory
#include "gcop_ctrl/QRotorIDModelInterfaceConfig.h"
#include <gcop_ctrl/qrotoridmodelcontrol.h>
#include <gcop_comm/gcop_trajectory_visualizer.h>


using namespace std;

//ros messages
gcop_comm::CtrlTraj trajectory; ///< Trajectory message for publishing the optimized trajectory

//ros Subscriber for final goal:
ros::Subscriber goal_subscriber_;///< Subscribes to goal pose from rviz

//ros Publisher for ctrltraj:
ros::Publisher traj_publisher_;

//QRotorIDModel Control:
QRotorIDModelControl model_control;///< Quadrotor ddp model control

//Visualization of Gcop Trajectory
GcopTrajectoryVisualizer *visualizer_;///< Visualizes CtrlTrajectories

//Params:
int Nit = 30;
double goal_z;

/** Iterates through the optimization algorithm. Is called by a ros timer
 */
void iterateCallback(const ros::TimerEvent & event)
{
  //	ros::Time startime = ros::Time::now();
  struct timeval timer;
  timer_start(timer);
  model_control.iterate(Nit);
  long te = timer_us(timer);
  cout << "Time taken " << te << " us." << endl;

  //Publish the optimized trajectory
  model_control.getCtrlTrajectory(trajectory);
  visualizer_->publishTrajectory(trajectory);
  traj_publisher_.publish(trajectory);
}

void paramreqCallback(gcop_ctrl::QRotorIDModelInterfaceConfig &config, uint32_t level)
{
    Nit = config.Nit;
    goal_z = config.zf;
}

void goalreqCallback(const geometry_msgs::PoseStamped &goal_pose)
{
    ROS_INFO("Received Goal Iterating");
    geometry_msgs::Pose goal_pose_ = goal_pose.pose;
    goal_pose_.position.z = goal_z;
    model_control.setGoal(goal_pose_);
    //Iterate
    ros::TimerEvent event;
    iterateCallback(event);
}

/** Reconfiguration interface for configuring the optimization problem
 */
int main(int argc, char** argv)
{
  ros::init(argc, argv, "rccarctrl");
  ros::NodeHandle rosddp("/ddp");
  visualizer_ = new GcopTrajectoryVisualizer(rosddp);
  //Initialize subscriber
  goal_subscriber_ = rosddp.subscribe("/move_base_simple/goal",1,goalreqCallback);
  traj_publisher_ = rosddp.advertise<gcop_comm::CtrlTraj>("ctrltraj",1);

  //Trajectory message initialization
  /*trajectory.N = N;
  trajectory.statemsg.resize(N+1);
  trajectory.ctrl.resize(N);
  trajectory.time = ts;
  trajectory.finalgoal.statevector.resize(4);
  */
  //trajectory.time.resize(N);
  //Dynamic Reconfigure setup Callback ! immediately gets called with default values
  dynamic_reconfigure::Server<gcop_ctrl::QRotorIDModelInterfaceConfig> server;
  dynamic_reconfigure::Server<gcop_ctrl::QRotorIDModelInterfaceConfig>::CallbackType f;
  f = boost::bind(&paramreqCallback, _1, _2);
  server.setCallback(f);

  //	ros::TimerEvent event;

  //  iterateCallback(event);
  //create timer for iteration
  ros::spin();
  return 0;
}
