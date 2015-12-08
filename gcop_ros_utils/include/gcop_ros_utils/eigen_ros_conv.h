/*
 * eigen_ros_conv.h
 *
 *  Created on: Aug 18, 2015
 *      Author: subhransu
 */

#ifndef EIGEN_ROS_CONV_H_
#define EIGEN_ROS_CONV_H_

#include <ros/ros.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Twist.h>
#include <Eigen/Dense>
#include <gcop/so3.h>

using namespace Eigen;
using namespace gcop;

void eig2PoseMsg(geometry_msgs::Pose& pose, const Matrix3d& rot, const Vector3d& pos)
{
  Quaterniond quat(rot);
  pose.orientation.w = quat.w();
  pose.orientation.x = quat.x();
  pose.orientation.y = quat.y();
  pose.orientation.z = quat.z();

  pose.position.x = pos(0);
  pose.position.y = pos(1);
  pose.position.z = pos(2);
}

void eig2PoseMsg(geometry_msgs::Pose& pose_ros, const Affine3d& pose_eig)
{
  Quaterniond quat(pose_eig.rotation());
  pose_ros.orientation.w = quat.w();
  pose_ros.orientation.x = quat.x();
  pose_ros.orientation.y = quat.y();
  pose_ros.orientation.z = quat.z();

  pose_ros.position.x = pose_eig.translation()(0);
  pose_ros.position.y = pose_eig.translation()(1);
  pose_ros.position.z = pose_eig.translation()(2);
}

void eig2TwistMsg(geometry_msgs::Twist& twist, const Vector3d& w, const Vector3d& v)
{
  twist.angular.x = w(0);
  twist.angular.y = w(1);
  twist.angular.z = w(2);

  twist.linear.x  = v(0);
  twist.linear.y  = v(1);
  twist.linear.z  = v(2);
}

void poseMsg2Eig(Matrix3d& rot, Vector3d& pos, const geometry_msgs::Pose& pose)
{
  Quaterniond quat(pose.orientation.w , pose.orientation.x , pose.orientation.y , pose.orientation.z);
  rot=quat.matrix();
  pos << pose.position.x , pose.position.y , pose.position.z;
}

void poseMsg2Eig(Affine3d& pose_eig, const geometry_msgs::Pose& pose_ros)
{
  pose_eig = Translation3d(pose_ros.position.x,pose_ros.position.y,pose_ros.position.z)
            *Quaterniond(pose_ros.orientation.w , pose_ros.orientation.x , pose_ros.orientation.y , pose_ros.orientation.z);
}


#endif /* EIGEN_ROS_CONV_H_ */
