/*
 * pose_twist.h
 *
 *  Created on: Feb 27, 2016
 *      Author: subhransu
 */

#ifndef POSE_TWIST_H
#define POSE_TWIST_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ros/ros.h>
#include <gcop_ros_utils/lie_utils/se3.h>

using namespace Eigen;

typedef Matrix<double, 6,1> Vector6d;

class PoseTwist{
public:
  Affine3d pose_;
  Vector6d twist_;
  bool spatial_linear_;  //! If true then the linear velocity is in spatial coordinates else in body
  bool spatial_angular_; //! If true then the angular velocity is in spatial coordinates else in body
  ros::Time stamp_;          //! Time when pose was received
public:
  PoseTwist(double* arr, bool spatial_linear=true, bool spatial_angular=true);
  PoseTwist(bool spatial_linear=true, bool spatial_angular=true);
  void arr2PoseTwist(const double* arr);
  void arr2PoseTwist(const double* arr,Affine3d& pose, Vector6d& twist)const;
  void poseTwist2arr(double* arr) const;
  void poseTwist2arr(double* arr,const Affine3d& pose_binw, const Vector6d& twist_binb)const;
  static PoseTwist& Instance(void);
};




#endif /* POSE_TWIST_H */
