#include "gcop_ctrl/utils.h"
#include <ctime>
#include <gcop/se3.h>

using namespace gcop;

Eigen::Matrix<double, 2, 4> image_jacobian_flat(Eigen::Vector2d pt, double fx, double fy, double cx, 
  double cy, double z)
{
  double u = pt(0) - cx;
  double v = pt(1) - cy;
  Eigen::Matrix<double, 2, 4> im_jac;
  //im_jac <<   -fx/z,      0, u/z,
  //                0, -fy/z, v/z;
  im_jac <<  -(u*u+fx*fx)/fx,  -fx/z,      0, u/z,
                     -u*v/fy,      0, -fy/z, v/z;
  return im_jac;
}

void find_stable_final_pose_ransac(const std::vector<Eigen::Vector3d>& pts3d,
  const std::vector<Eigen::Vector2d>& pts2d,
  const Eigen::Matrix3d& K, const Eigen::Matrix4d& cam_transform,
  const Eigen::Matrix3d& att,
  Body3dState& x,
  int iterations,
  std::vector<int>& inliers)
{
  const int m = 5;
  const double reproj_thresh = 10.0;

  inliers.clear();

  std::vector<int> ind;
  for(unsigned int i = 0; i < pts3d.size(); i++)
  {
    ind.push_back(i);
  }
 
  std::vector<int> best_inliers_idx;
  for(int i = 0; i < iterations; i++)
  {
    std::vector<Eigen::Vector3d> rand_pts3d;
    std::vector<Eigen::Vector2d> rand_pts2d;
    std::random_shuffle(ind.begin(), ind.end());
    std::vector<int> rand_ind(ind.begin(), ind.begin()+m);
    for(int j = 0; j < m; j++)
    {
      rand_pts3d.push_back(pts3d[rand_ind[j]]);
      rand_pts2d.push_back(pts2d[rand_ind[j]]);
    }

    Body3dState x_rand = x;
    find_stable_final_pose(rand_pts3d, rand_pts2d, K, cam_transform, att, x_rand, 10000);

    std::vector<int> inliers_idx;
    Eigen::Matrix4d state_mat;
    state_mat.setIdentity(); 
    state_mat.topLeftCorner<3,3>() = x_rand.R;
    state_mat.block<3,1>(0,3) = x_rand.p;
    Eigen::Matrix<double, 3, 4> P = K*(state_mat*cam_transform).inverse().block<3,4>(0,0);
    for(int j = 0; j < pts3d.size(); j++)
    {
      Eigen::Vector3d pt_proj = P*pts3d.at(j).homogeneous();
      double depth = pt_proj(2);
      Eigen::Vector2d pt2d = pt_proj.head<2>()/depth;
      double error = (pt2d - pts2d.at(j)).norm();
      if(error < reproj_thresh)
      {
        inliers_idx.push_back(j);
      }
    }
    if(inliers_idx.size() > best_inliers_idx.size())
    {
      best_inliers_idx = inliers_idx;
    }
  }
  std::vector<Eigen::Vector3d> inlier_pts3d;
  std::vector<Eigen::Vector2d> inlier_pts2d;
  for(int j = 0; j < best_inliers_idx.size(); j++)
  {
    inlier_pts3d.push_back(pts3d[best_inliers_idx[j]]);
    inlier_pts2d.push_back(pts2d[best_inliers_idx[j]]);
  }
  find_stable_final_pose(inlier_pts3d, inlier_pts2d, K, cam_transform, att, x, 100000); 
  inliers = best_inliers_idx;
}

void find_stable_final_pose(const std::vector<Eigen::Vector3d>& pts3d, 
  const std::vector<Eigen::Vector2d>& pts2d,
  const Eigen::Matrix3d& K, const Eigen::Matrix4d& cam_transform, 
  const Eigen::Matrix3d& att,
  Body3dState& x,  
  int iterations)
{
  const double h = 1e-1;

  Eigen::Matrix<double, 4, 4> cam_vel_transform;
  cam_vel_transform.setZero();
  cam_vel_transform.bottomRightCorner<3,3>() = cam_transform.topLeftCorner<3,3>();
  cam_vel_transform(0,0) = cam_transform(2,1);

  double fx = K(0,0);
  double fy = K(1,1);
  double cx = K(0,2);
  double cy = K(1,2);
  
  Eigen::Matrix4d state_mat;
  state_mat.setIdentity(); 
  state_mat.topLeftCorner<3,3>() = att;
  state_mat.block<3,1>(0,3) = x.p;

  Eigen::MatrixXd image_jac(2*pts3d.size(), 4);
  Eigen::VectorXd feature_errors(2*pts3d.size());
  //double feature_error = 0;
  for(int i = 0; i < iterations; i++)
  {
    std::clock_t start = std::clock();
    Eigen::Matrix<double, 3, 4> P = K*(state_mat*cam_transform).inverse().block<3,4>(0,0);
    for(int j = 0; j < pts3d.size(); j++)
    {
      Eigen::Vector3d pt_proj = P*pts3d.at(j).homogeneous();
      double depth = pt_proj(2);
      Eigen::Vector2d pt2d = pt_proj.head<2>()/depth;
      feature_errors.segment<2>(2*j) = pt2d - pts2d.at(j);
      image_jac.block<2,4>(2*j, 0) = image_jacobian_flat(pt2d, fx, fy, cx, cy, depth);
    }
    //feature_error = sqrt(feature_errors.squaredNorm()/pts3d.size());
    
    image_jac = image_jac*cam_vel_transform.transpose();

    //Calc GN step
    Eigen::Matrix<double, 6, 1> s;
    s.setZero();
    Eigen::MatrixXd image_jac_psdinv = 
      (image_jac.transpose()*image_jac).inverse()*image_jac.transpose();
    s.tail<4>() = -image_jac_psdinv*feature_errors;

    Eigen::Matrix4d m;    
    gcop::SE3::Instance().cay(m, h*s);
    state_mat = state_mat*m;

    //cout 
    //  << "Feature Errors " << i << " = " << std::endl << feature_errors.head<10>() << std::endl
    //  << "\t Time = " << 1000.*(std::clock()-start)/CLOCKS_PER_SEC
    //  << endl << "p = " << state_mat.block<3,1>(0,3).transpose() 
    //  << endl << "R = " << std::endl << state_mat.block<3,3>(0,0)
    //  << endl << "s = " << s.transpose() << std::endl;
  
    if(s.norm() < 1e-5)
      break;
  }
  //std::cout << "PnP reproj error = " << feature_error << std::endl;
  x.R = state_mat.topLeftCorner<3,3>();
  x.p = state_mat.block<3,1>(0,3);
}
