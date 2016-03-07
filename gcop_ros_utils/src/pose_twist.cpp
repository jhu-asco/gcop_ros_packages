#include <gcop_ros_utils/pose_twist.h>

PoseTwist::PoseTwist(double* arr, bool spatial_linear, bool spatial_angular)
:spatial_linear_(spatial_linear), spatial_angular_(spatial_angular){
  pose_.setIdentity();
  twist_.setZero();
}

PoseTwist::PoseTwist(bool spatial_linear, bool spatial_angular)
:spatial_linear_(spatial_linear), spatial_angular_(spatial_angular){
  pose_.setIdentity();
  twist_.setZero();
}

void PoseTwist::arr2PoseTwist(const double* arr){
  arr2PoseTwist(arr,pose_, twist_);
}

void PoseTwist::arr2PoseTwist(const double* arr,Affine3d& pose, Vector6d& twist)const{
  for(size_t i=0;i<6;i++)
    twist[i] = arr[i];
  Vector3d rpy; rpy<< arr[6], arr[7] , arr[8];

  Matrix3d rotn; gcop_ros_utils::SO3::Instance().q2g(rotn, rpy); pose.linear() = rotn;
  pose.translation() << arr[9] ,arr[10], arr[11];
}

/**
 * Converts the stored pose and twist into a compact representation of array of 12 numbers
 * @param arr [wx,wy,wz,vx,vy,vz,roll,pitch,yaw,x,y,z]
 */
void PoseTwist::poseTwist2arr(double* arr)const{
  poseTwist2arr(arr,pose_, twist_);
}

void PoseTwist::poseTwist2arr(double* arr,const Affine3d& pose, const Vector6d& twist)const{
  for(size_t i=0;i<6;i++)
    arr[i] = twist[i];
  Vector3d rpy; gcop_ros_utils::SO3::Instance().g2q(rpy,pose.rotation().matrix());
  arr[6] = rpy[0]; arr[7] = rpy[1]; arr[8] = rpy[2];
  arr[9] = pose.translation()(0); arr[10] = pose.translation()(1); arr[11] = pose.translation()(2);
}

PoseTwist& PoseTwist::Instance(void){
  static PoseTwist pt;
  return pt;
}


