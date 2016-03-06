#ifndef GCOP_ROS_UTILS_SE3_H
#define GCOP_ROS_UTILS_SE3_H

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "so3.h"

namespace gcop_ros_utils {

  using namespace Eigen;

  typedef Matrix<double, 6, 1> Vector6d;
  typedef Matrix<double, 7, 1> Vector7d;
  typedef Matrix<double, 4, 4> Matrix4d;
  typedef Matrix<double, 6, 6> Matrix6d;

  class SE3{
  public:

    SE3();

    static SE3& Instance();
  
    void inv(Matrix4d &gi, const Matrix4d &g) const;

    void hat(Matrix4d &vh, const Vector6d &v) const;

    void hatinv(Vector6d &v, const Matrix4d &vh) const;

    void Tg(Matrix6d &M, const Matrix4d &g) const;
    
    void Ad(Matrix6d &M, const Matrix4d &g) const;

    void ad(Matrix6d &M, const Vector6d &v) const;

    void adt(Matrix6d &M, const Vector6d &mu) const;

    void adinv(Vector6d& v, const Matrix6d& m) const;

    void exp(Matrix4d &g, const Vector6d &v) const;

    void log(Vector6d& v, const Matrix4d& g) const;

    void log(Vector6d &v, const Affine3d &g) const;

    void cay(Matrix4d& g, const Vector6d &v) const;

    void dcay(Matrix6d& M, const Vector6d& v) const;
    
    void dcayinv(Matrix6d& M, const Vector6d& v) const;

    void dexp(Matrix6d& M, const Vector6d& v) const;
    
    void dexpinv(Matrix6d& M, const Vector6d& v) const;

    void tlnmu(Vector6d& mup, const Vector6d& v, const Vector6d &mu) const;

    void tln(Matrix6d &M, const Vector6d &v) const;

    void q2g(Matrix4d &g, const Vector6d &q) const;

    void quatxyz2g(Matrix4d &g, const Vector7d &wquatxyz) const;

    void g2quatxyz(Vector7d &wquatxyz, const Matrix4d &g) const;

    void g2q(Vector6d &q, const Matrix4d &g) const;

    void rpyxyz2g(Matrix4d &g, const Vector3d &rpy, const Vector3d &xyz) const;

    void g2rpyxyz(Vector3d &rpy, Vector3d &xyz, const Matrix4d &g) const;

    double tol;     ///< numerical tolerance

    SO3 &so3;       ///< convenience object for accessing SO3 operations    
    Matrix4d Id;
  };
}

#endif //GCOP_ROS_UTILS_SE3_H
