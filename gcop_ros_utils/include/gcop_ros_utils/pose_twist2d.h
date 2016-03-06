/*
 * pose_twist2d.h
 *
 *  Created on: Feb 27, 2016
 *      Author: subhransu
 */

#ifndef POSE_TWIST2D_H
#define POSE_TWIST2D_H
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <gcop_ros_utils/pose_twist.h>

using Eigen::Affine2d;
using Eigen::Vector3d;

class PoseTwist2d{
public:
  Affine2d pose_;
  Vector3d twist_;

  PoseTwist2d();
  ~PoseTwist2d();

  PoseTwist toPoseTwist3d()const ;

  void fromPoseTwist3d();

};

#endif /* POSE_TWIST2D_H */
