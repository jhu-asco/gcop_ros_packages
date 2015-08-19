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
  Vector4d wxyz;
  SO3::Instance().g2quat(wxyz,rot);
  pose.orientation.w = wxyz(0);
  pose.orientation.x = wxyz(1);
  pose.orientation.y = wxyz(2);
  pose.orientation.z = wxyz(3);

  pose.position.x = pos(0);
  pose.position.y = pos(1);
  pose.position.z = pos(2);
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

#endif /* EIGEN_ROS_CONV_H_ */
