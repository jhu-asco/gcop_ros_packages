#ifndef _OVS_UTIL_H_
#define _OVS_UTIL_H_

#include <vector>
#include <Eigen/Dense>
#include <gcop/body3dmanifold.h>

Eigen::Matrix<double, 2, 4> image_jacobian_flat(Eigen::Vector2d pt, double fx, double fy, double cx, 
  double cy, double z);

void find_stable_final_pose_ransac(const std::vector<Eigen::Vector3d>& pts3d,
  const std::vector<Eigen::Vector2d>& pts2d,
  const Eigen::Matrix3d& K, const Eigen::Matrix4d& cam_transform,
  const Eigen::Matrix3d& att,
  gcop::Body3dState& x,
  int iterations, std::vector<int>& inliers);

void find_stable_final_pose(const std::vector<Eigen::Vector3d>& pts3d, 
  const std::vector<Eigen::Vector2d>& pts2d,
  const Eigen::Matrix3d& K, const Eigen::Matrix4d& cam_transform, const Eigen::Matrix3d& att, 
  gcop::Body3dState& x,  
  int max_iterations);

#endif
