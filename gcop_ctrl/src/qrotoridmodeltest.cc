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


using namespace std;

//ros messages

//ros Subscriber for final goal:
//ros::Subscriber goal_subscriber_;///< Subscribes to goal pose from rviz


//QRotorIDModel Control:
QRotorIDModelControl *model_control;///< Quadrotor ddp model control

//Obstacle Info
//geometry_msgs::Vector3 obs_posn, obs_axis, obs_posn1, obs_axis1;
//double obs_radius, obs_radius1;

//Params:
//int Nit = 30;
//double goal_z;

/** Iterates through the optimization algorithm. Is called by a ros timer
 */
void Iterate()
{
  //	ros::Time startime = ros::Time::now();
  model_control->iterate();
  geometry_msgs::Vector3 localpos;
  localpos.x = 0; localpos.y = 0; localpos.z = 1;
  geometry_msgs::Vector3 rpy;
  rpy.x = 0; rpy.y = 0; rpy.z = 1;
  model_control->publishTrajectory(localpos,rpy);

  //Publish the optimized trajectory
}

/*void paramreqCallback(gcop_ctrl::QRotorIDModelInterfaceConfig &config, uint32_t level)
{
    if(config.iterate)
        Iterate();
}
*/

/*void goalreqCallback(const geometry_msgs::PoseStamped &goal_pose)
{
    ROS_INFO("Received Goal Iterating");
    geometry_msgs::Pose goal_pose_ = goal_pose.pose;
    goal_pose_.position.z = goal_z;
    model_control->setGoal(goal_pose_);
    //Iterate
    ros::TimerEvent event;
    iterateCallback(event);
}
*/

/** Reconfiguration interface for configuring the optimization problem
 */
int main(int argc, char** argv)
{
  ros::init(argc, argv, "rccarctrl");
  ros::NodeHandle rosddp("/ddp");
  model_control = new QRotorIDModelControl(rosddp);
  Iterate();

  //Initialize subscriber
  //goal_subscriber_ = rosddp.subscribe("/move_base_simple/goal",1,goalreqCallback);

  //Dynamic Reconfigure setup Callback ! immediately gets called with default values
  //dynamic_reconfigure::Server<gcop_ctrl::QRotorIDModelInterfaceConfig> server;
  //dynamic_reconfigure::Server<gcop_ctrl::QRotorIDModelInterfaceConfig>::CallbackType f;
  //f = boost::bind(&paramreqCallback, _1, _2);
  //server.setCallback(f);

  ros::spin();
  return 0;
}
