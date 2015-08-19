#ifndef _IMAGECOST_H
#define _IMAGECOST_H

#include <gcop/body3dcost.h>
#include <gcop/body3d.h>
#include <gcop/lqcost.h>
  
using namespace std;
using namespace Eigen;
using namespace gcop;
using namespace cv; 

template <int c>
class ImageCost : public LqCost<Body3dState, 12, c, Dynamic>{
  typedef Matrix<double, c, 1> Vectorcd;
  typedef Matrix<double, Dynamic, 1> Vectorgd;
  typedef Matrix<double, Dynamic, 1> Vectormd;
  typedef Matrix<double, Dynamic, 12> Matrixgxd;
  typedef Matrix<double, Dynamic, c> Matrixgud;
  typedef Matrix<double, Dynamic, Dynamic> Matrixgpd;
  typedef Matrix<double, c, c> Matrixcd;
  typedef Matrix<double, 6, c> Matrix6xcd;
  typedef Matrix<double, 12, c> Matrix12xcd;
  typedef Matrix<double, Dynamic, Dynamic> Matrixmd;
  typedef Matrix<double, 12, Dynamic> Matrixnmd;
  typedef Matrix<double, Dynamic, 12> Matrixmnd;
        

public:
    
  ImageCost(std::vector<Eigen::Vector3d> pts3d, std::vector<Eigen::Vector2d> pts2d, 
    Eigen::Matrix3d K, Body3d<c> &sys, double tf, const Body3dState &xf, 
    Eigen::Matrix4d cam_transform, bool diag = true);
    
  virtual bool Res(Vectorgd &g,
             double t, const Body3dState &x, const Vectorcd &u, double h,
             const Vectormd *p = 0,
             Matrixgxd *dgdx = 0, Matrixgud *dgdu = 0,
             Matrixgpd *dgdp = 0);
  
  virtual double L(double t, const Body3dState &x, const Vectorcd &u, double h,
            const Vectormd *p = 0,
            Vector12d *Lx = 0, Matrix12d *Lxx = 0,
            Vectorcd *Lu = 0, Matrixcd *Luu = 0,
            Matrix12xcd *Lxu = 0, Vectormd *Lp = 0, Matrixmd *Lpp = 0,
            Matrixmnd *Lpx = 0);

  double imageQ;
  double imageQf;

private:
  double GetAvgReprojectionError(std::vector<Eigen::Vector3d> pts3d, 
    std::vector<Eigen::Vector2d>& pts2d, 
    const Eigen::Matrix4d& tf, const Eigen::Matrix3d& K);

  Eigen::Matrix4d cam_transform;
  Eigen::Matrix3d K;
  std::vector<Eigen::Vector3d> pts3d;
  std::vector<Eigen::Vector2d> pts2d;
};  


template <int c> 
ImageCost<c>::ImageCost(std::vector<Eigen::Vector3d> pts3d, std::vector<Eigen::Vector2d> pts2d, Eigen::Matrix3d K, Body3d<c> &sys, double tf, const Body3dState &xf, 
  Eigen::Matrix4d cam_transform, bool diag) : 
  LqCost<Body3dState, 12, c>(sys, tf, xf, diag), 
  cam_transform(cam_transform),
  pts3d(pts3d),
  pts2d(pts2d),
  K(K)
{
  this->Q.setZero();
  this->imageQ = 0; 
  this->imageQf = 1; 
  this->Qf(0,0) = 0;
  this->Qf(1,1) = 0;
  this->Qf(2,2) = 0;
  this->Qf(3,3) = 5;
  this->Qf(4,4) = 5;
  this->Qf(5,5) = 5;
    
  this->Qf(6,6) = .1;
  this->Qf(7,7) = .1;
  this->Qf(8,8) = .1;
  this->Qf(9,9) = 1;
  this->Qf(10,10) = 1;
  this->Qf(11,11) = 1;
    
  this->R.diagonal() = Matrix<double, c, 1>::Constant(1.);
}
  
template <int c>
double ImageCost<c>::GetAvgReprojectionError(std::vector<Eigen::Vector3d> pts3d, std::vector<Eigen::Vector2d>& pts2d, const Eigen::Matrix4d& tf, const Eigen::Matrix3d& K)
{
  assert(pts3d.size() == pts2d.size());

  double error = 0;
  Eigen::Matrix<double, 3, 4> P = K*tf.block<3,4>(0,0);
  for(int i = 0; i < pts3d.size(); i++)
  {
    Eigen::Vector3d pt2d_reproj = P*Eigen::Vector4d(pts3d[i](0), pts3d[i](1), pts3d[i](2), 1);
    pt2d_reproj /= pt2d_reproj(2);

    error += (pt2d_reproj.head<2>()-pts2d[i]).norm();
  }
  error /= pts3d.size();
  return error;
}

template <int c> 
bool ImageCost<c>::Res(Vectorgd &g,
             double t, const Body3dState &x, const Vectorcd &u, double h,
             const Vectormd *p,
             Matrixgxd *dgdx, Matrixgud *dgdu,
             Matrixgpd *dgdp)
{
  g = Vectorgd::Zero(g.size());

  Eigen::Vector3d cay = Eigen::VectorXd::Zero(3);
  SO3::Instance().cayinv(cay, x.R);
  if (t > this->tf - 1e-10) 
  {
    Eigen::Matrix4d sys_world_transform;
    sys_world_transform.setIdentity();
    sys_world_transform.block<3,3>(0,0) = x.R;
    sys_world_transform.block<3,1>(0,3) = x.p;
    Eigen::Matrix4d cam_world_transform = sys_world_transform*cam_transform;

    double error = this->GetAvgReprojectionError(pts3d, pts2d, cam_world_transform.inverse(), K);
  
    g(3) = sqrt(imageQf)*error;//sqrt(error);
    g.segment<3>(0) = this->Qfsqrt.block(0,0,3,3)*cay;
    g.segment<3>(6) = this->Qfsqrt.block(6,6, 3, 3)*x.w;
    g.segment<3>(9) = this->Qfsqrt.block(9,9, 3, 3)*x.v;

    //std::cout << "Res: " << g.transpose() << std::endl;
    //std::cout << "State: " << x.second.transpose() << std::endl;
  }
  else 
  {
    double error = 0;
    if(imageQ > 0)
    {
      Eigen::Matrix4d sys_world_transform;
      sys_world_transform.setIdentity();
      sys_world_transform.block<3,3>(0,0) = x.R;
      sys_world_transform.block<3,1>(0,3) = x.p;
      Eigen::Matrix4d cam_world_transform = sys_world_transform*cam_transform;

      error = this->GetAvgReprojectionError(pts3d, pts2d, cam_world_transform.inverse(), K);
    }

    g(3) = h*sqrt(imageQ)*error;//sqrt(error);
    g.segment<3>(0) = h*this->Qsqrt.block(0,0,3,3)*cay;
    g.segment<3>(6) = h*this->Qsqrt.block(6,6, 3, 3)*x.w;
    g.segment<3>(9) = h*this->Qsqrt.block(9,9, 3, 3)*x.v;
  }
  
  this->du = h*u;
  if (this->diag)
    g.tail(c) = this->Rsqrt.diagonal().cwiseProduct(this->du);
  else
    g.tail(c) = this->Rsqrt*(this->du);
  
  //std::cout << "Controls: " << u.transpose()  << std::endl;
  //getchar();

  //std::cout << "# matches: " << num_matches << std::endl;

  return true;
}

template <int c> 
double ImageCost<c>::L(double t, const Body3dState &x, const Matrix<double, c, 1> &u,
                          double h, const Vectormd *p,
                          Vector12d *Lx, Matrix12d *Lxx,
                          Matrix<double, c, 1> *Lu, Matrix<double, c, c> *Luu,
                          Matrix<double, 12, c> *Lxu, Vectormd *Lp, Matrixmd *Lpp,
                          Matrixmnd *Lpx)
{
  Eigen::Vector3d cay = Eigen::VectorXd::Zero(3);
  SO3::Instance().cayinv(cay, x.R);
  Eigen::Vector3d lv = x.v;
  Eigen::Vector3d lw = x.w;

  if (t > this->tf - 1e-10) 
  {
    //std::cout << "state: " << x.second.transpose() << std::endl;
    Eigen::Matrix4d sys_world_transform;
    sys_world_transform.setIdentity();
    sys_world_transform.block<3,3>(0,0) = x.R;
    sys_world_transform.block<3,1>(0,3) = x.p;
    Eigen::Matrix4d cam_world_transform = sys_world_transform*cam_transform;

    double error = this->GetAvgReprojectionError(pts3d, pts2d, cam_world_transform.inverse(), K);
    //cout << "photometric error: " << error << endl;

    if(Lu)
      Lu->setZero();
    if (Luu)
      Luu->setZero();
    
    //std::cout << "error: " << error << std::endl;
    return imageQf*error + (x.w.dot(this->Qf.block(6,6,3,3)*x.w) 
      + x.v.dot(this->Qf.block(9,9,3,3)*x.v) 
      + cay.dot(this->Qf.block(0,0,3,3)*cay))/2.;
  }
  else
  {
    double error = 0;
    if(imageQ > 0)
    {
      Eigen::Matrix4d sys_world_transform;
      sys_world_transform.setIdentity();
      sys_world_transform.block<3,3>(0,0) = x.R;
      sys_world_transform.block<3,1>(0,3) = x.p;
      Eigen::Matrix4d cam_world_transform = sys_world_transform*cam_transform;
      error = imageQ*this->GetAvgReprojectionError(pts3d, pts2d, cam_world_transform.inverse(), K);
    }

    if (Lu)
      if (this->diag)
        *Lu = this->R.diagonal().cwiseProduct(h*u);
      else
        *Lu = this->R*(h*u);

    if(Luu)
      *Luu = h*this->R;  

    return h*(error + lw.dot(this->Q.block(6,6,3,3)*lw) + lv.dot(this->Q.block(9,9,3,3)*lv) 
      + cay.dot(this->Q.block(0,0,3,3)*cay)
      + u.dot(this->R*u))/2.;
  }
 
/*
  Vector12d dx;
  this->X.Lift(dx, this->xf, x);

  
  // check if final state
  if (t > this->tf - 1e-10) {
    if (Lx) {
      if (this->diag)
        *Lx = this->Qf.diagonal().cwiseProduct(dx);
      else
        *Lx = this->Qf*dx;
      
      // add dcayinv if this->Q(1:3,1:3) != a*Id
      //      (*Lx).head(3) = Rt*(*Lx).head<3>();
    }
    if (Lxx) {
      *Lxx = this->Qf;
      //      (*Lxx).topLeftCorner(3,3) = Rt*(*Lxx).topLeftCorner<3,3>()*R;
    }

    if (Lu)
      Lu->setZero();
    if (Luu)
      Luu->setZero();
    if (Lxu)
      Lxu->setZero();

    if (this->diag)
      return dx.dot(this->Qf.diagonal().cwiseProduct(dx))/2;
    else
      return dx.dot(this->Qf*dx)/2;
    
  } else {
    if (Lx) {
      if (this->diag)
        *Lx = this->Q.diagonal().cwiseProduct(dx);
      else
        *Lx = this->Q*dx;
      //      (*Lx).head<3>() = Rat*(*Lx).head<3>();
    }
    if (Lxx) {
      *Lxx = this->Q;
      //      (*Lxx).topLeftCorner<3,3>() = Rt*this->Q.topLeftCorner<3,3>()*R;
    }
    if (Lu)
      if (this->diag)
        *Lu = this->R.diagonal().cwiseProduct(u);
      else
        *Lu = this->R*u;

    if (Luu)
      *Luu = this->R;
      
    if (Lxu)
      Lxu->setZero();

    if (this->diag)
      return (dx.dot(this->Q.diagonal().cwiseProduct(dx)) + u.dot(this->R.diagonal().cwiseProduct(u)))/2;
    else
      return (dx.dot(this->Q*dx) + u.dot(this->R*u))/2;
  }
*/
}
  



#endif
