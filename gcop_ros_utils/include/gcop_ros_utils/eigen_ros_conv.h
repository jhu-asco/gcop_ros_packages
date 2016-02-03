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
#include <geometry_msgs/Transform.h>
#include <Eigen/Dense>
#include <gcop/so3.h>

using namespace Eigen;
using namespace gcop;

void eig2PoseMsg(geometry_msgs::Pose& msg_pose, const Matrix3d& orientation, const Vector3d& position)
{
  Quaterniond quat(orientation);
  msg_pose.orientation.w = quat.w();
  msg_pose.orientation.x = quat.x();
  msg_pose.orientation.y = quat.y();
  msg_pose.orientation.z = quat.z();

  msg_pose.position.x = position(0);
  msg_pose.position.y = position(1);
  msg_pose.position.z = position(2);
}

void eig2PoseMsg(geometry_msgs::Pose& msg_pose, const Affine3d& eig_pose)
{
  Quaterniond quat(eig_pose.rotation());
  msg_pose.orientation.w = quat.w();
  msg_pose.orientation.x = quat.x();
  msg_pose.orientation.y = quat.y();
  msg_pose.orientation.z = quat.z();

  msg_pose.position.x = eig_pose.translation()(0);
  msg_pose.position.y = eig_pose.translation()(1);
  msg_pose.position.z = eig_pose.translation()(2);
}

void eig2TransformMsg(geometry_msgs::Transform& msg_transform, const Matrix3d& rotation, const Vector3d& translation)
{
  Quaterniond quat(rotation);
  msg_transform.rotation.w = quat.w();
  msg_transform.rotation.x = quat.x();
  msg_transform.rotation.y = quat.y();
  msg_transform.rotation.z = quat.z();

  msg_transform.translation.x = translation(0);
  msg_transform.translation.y = translation(1);
  msg_transform.translation.z = translation(2);
}

void eig2TransformMsg(geometry_msgs::Transform& msg_transform, const Affine3d& pose_eig)
{
  Quaterniond quat(pose_eig.rotation());
  msg_transform.rotation.w = quat.w();
  msg_transform.rotation.x = quat.x();
  msg_transform.rotation.y = quat.y();
  msg_transform.rotation.z = quat.z();

  msg_transform.translation.x = pose_eig.translation()(0);
  msg_transform.translation.y = pose_eig.translation()(1);
  msg_transform.translation.z = pose_eig.translation()(2);
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

void poseMsg2Eig(Matrix3d& orientation, Vector3d& position, const geometry_msgs::Pose& msg_pose)
{
  Quaterniond quat(msg_pose.orientation.w , msg_pose.orientation.x , msg_pose.orientation.y , msg_pose.orientation.z);
  orientation=quat.matrix();
  position << msg_pose.position.x , msg_pose.position.y , msg_pose.position.z;
}

void poseMsg2Eig(Affine3d& eig_pose, const geometry_msgs::Pose& msg_pose)
{
  eig_pose = Translation3d(msg_pose.position.x,msg_pose.position.y,msg_pose.position.z)
            *Quaterniond(msg_pose.orientation.w , msg_pose.orientation.x , msg_pose.orientation.y , msg_pose.orientation.z);
}

void transformMsg2Eig(Matrix3d& rotation, Vector3d& translation, const geometry_msgs::Transform& msg_transform)
{
  Quaterniond quat(msg_transform.rotation.w , msg_transform.rotation.x , msg_transform.rotation.y , msg_transform.rotation.z);
  rotation=quat.matrix();
  translation << msg_transform.translation.x , msg_transform.translation.y , msg_transform.translation.z;
}

void transformMsg2Eig(Affine3d& eig_transform, const geometry_msgs::Transform& msg_transform)
{
  eig_transform = Translation3d(msg_transform.translation.x,msg_transform.translation.y,msg_transform.translation.z)
            *Quaterniond(msg_transform.rotation.w , msg_transform.rotation.x , msg_transform.rotation.y , msg_transform.rotation.z);
}

#endif /* EIGEN_ROS_CONV_H_ */
