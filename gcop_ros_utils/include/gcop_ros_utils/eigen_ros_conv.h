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
#include <gcop_ros_utils/lie_utils/se3.h>

using namespace Eigen;

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

void toAffine2d(Affine2d& pose2d, const Affine3d& pose3d){
  pose2d.linear() = pose3d.linear().topLeftCorner<2,2>();
  pose2d.translation() = pose3d.translation().head<2>();
}

Affine2d toAffine2d(const Affine3d& pose3d){
  Affine2d pose2d;
  toAffine2d(pose2d, pose3d);
  return pose2d;
}

void toAffine2d(Affine2d& pose2d, const Matrix3d& se2){
pose2d.linear().setIdentity();
pose2d.affine().topLeftCorner<2,3>() = se2.topLeftCorner<2,3>();
}

Affine2d toAffine2d(const Matrix3d& se2){
  Affine2d pose2d;
  toAffine2d(pose2d, se2);
  return pose2d;
}

void toAffine3d(Affine3d& pose3d, const Affine2d& pose2d){
  pose3d.translation()<< pose2d.translation(), 0;
  pose3d.linear().setZero();
  pose3d.linear().topLeftCorner<2,2>()= pose2d.linear();
}

Affine3d toAffine3d(const Affine2d& pose2d){
  Affine3d pose3d;
  toAffine3d(pose3d, pose2d);
  return pose3d;
}


Affine3d axyToAffine3d(double a, double x, double y){
return  Translation3d(x,y,0)
       *AngleAxisd(a, Vector3d::UnitZ());
}

Affine3d axyToAffine3d(const Vector3d& axy){
return  Translation3d(axy(1),axy(2),0)
       *AngleAxisd(axy(0), Vector3d::UnitZ());
}

Vector3d affine3dToAxy(const Affine3d& pose){
  Vector3d rpy; gcop_ros_utils::SO3::Instance().g2q(rpy,pose.rotation().matrix());
  return Vector3d(rpy(2),pose.translation()(0),pose.translation()(1));
}

Affine2d axyToAffine2d(double a, double x, double y){
return  Translation2d(x,y)*Rotation2Dd(a);
}

Affine2d axyToAffine2d(const Vector3d& axy){
return  Translation2d(axy(1),axy(2))*Rotation2Dd(axy(0));
}

Vector3d affine2dToAxy(const Affine2d& pose){
  return affine3dToAxy(toAffine3d(pose));
}

#endif /* EIGEN_ROS_CONV_H_ */
