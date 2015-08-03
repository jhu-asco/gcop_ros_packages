/*
 * sim_ins_mag_gps.cpp
 *
 *  Created on: Jul 20, 2015
 *      Author: subhransu
 */

// ROS relevant includes
#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <ros/package.h>
#include <tf_conversions/tf_eigen.h>
#include <dynamic_reconfigure/server.h>
#include <gcop_ros_est/CalibratorConfig.h>


// ROS standard messages
#include <std_msgs/String.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/MagneticField.h>

//Other includes
#include <iostream>
#include <fstream>
#include <signal.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <limits>
#include <algorithm>
#include <XmlRpcValue.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

//-------------------------------------------------------------------------
//-----------------------NAME SPACES ---------------------------------
//-------------------------------------------------------------------------
using namespace std;
using namespace Eigen;


//-------------------------------------------------------------------------
//-----------------------TYPEDEFS ---------------------------------
//-------------------------------------------------------------------------


//-------------------------------------------------------------------------
//-----------------------GLOBAL VARIABLES ---------------------------------
//-------------------------------------------------------------------------
sig_atomic_t g_shutdown_requested=0;


//------------------------------------------------------------------------
//-----------------------FUNCTION DEFINITIONS ----------------------------
//------------------------------------------------------------------------


void mySigIntHandler(int signal)
{
  g_shutdown_requested=1;
}

void timer_start(struct timeval *time)
{
  gettimeofday(time,(struct timezone*)0);
}

long timer_us(struct timeval *time)
{
  struct timeval now;
  gettimeofday(&now,(struct timezone*)0);
  return 1000000*(now.tv_sec - time->tv_sec) + now.tv_usec - time->tv_usec;
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

/**
 * converts XmlRpc::XmlRpcValue to Eigen::Matrix<double,r,c> type
 * @param mat: is an Eigen::Matrix(with static rows and cols)
 * @param my_list
 */
template<typename T>
void xml2Mat(T &mat, XmlRpc::XmlRpcValue &my_list)
{
  assert(my_list.getType() == XmlRpc::XmlRpcValue::TypeArray);
  assert(my_list.size() > 0);
  assert(mat.size()==my_list.size());

  for (int i = 0; i < mat.rows(); i++)
  {
    for(int j=0; j<mat.cols();j++)
    {
      int k = j+ i*mat.cols();
      assert(my_list[k].getType() == XmlRpc::XmlRpcValue::TypeDouble);
      mat(i,j) =  (double)(my_list[k]);
    }
  }
}

/**
 * Converts a XmlRpc::XmlRpcValue to a Eigen::Vector(dynamic type)
 * @param vec
 * @param my_list
 */
void xml2vec(VectorXd &vec, XmlRpc::XmlRpcValue &my_list)
{
  assert(my_list.getType() == XmlRpc::XmlRpcValue::TypeArray);
  assert(my_list.size() > 0);
  vec.resize(my_list.size());

  for (int32_t i = 0; i < my_list.size(); i++)
  {
    assert(my_list[i].getType() == XmlRpc::XmlRpcValue::TypeDouble);
    vec[i] =  (double)(my_list[i]);
  }
}

//------------------------------------------------------------------------
//------------------------MAIN CALLBACK CLASS ----------------------------
//------------------------------------------------------------------------


class CallBackCalibrator
{
public:
  typedef Matrix<double, 4, 4> Matrix4d;
  typedef Matrix<double, 4, 1> Vector4d;
  typedef Matrix<double, 7, 1> Vector7d;

public:
  CallBackCalibrator();
  ~CallBackCalibrator();

private:

  void cbSubImu(const sensor_msgs::Imu::ConstPtr& msg_imu);
  void cbSubMag(const sensor_msgs::MagneticField::ConstPtr& msg_mag);
  void cbSubAccV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg_acc_v3s);
  void cbSubMagV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg_mag_v3s);

  void cbReconfig(gcop_ros_est::CalibratorConfig &config, uint32_t level);

  void setupTopicsAndNames(void);

  void initRvizMarkers(void);
  void sendMarkersAndTF(Vector3d xyz_gps);

  void initSubsPubsAndTimers(void);
  template<typename T>
  void rosParam2Mat(T &mat, std::string param);

  void writeOneAcc(void);


private:
  //ROS relavant members
  ros::NodeHandle nh_, nh_p_;
  gcop_ros_est::CalibratorConfig config_;
  dynamic_reconfigure::Server<gcop_ros_est::CalibratorConfig> dyn_server_;
  ros::Timer timer_general_;
  ros::Publisher pub_viz_cov_;
  string strfrm_map_, strfrm_robot_, strfrm_gps_lcl_;

  ros::Subscriber sub_imu_   , sub_mag_;
  string    strtop_imu_, strtop_mag_;

  ros::Subscriber sub_acc_v3s_   , sub_mag_v3s_;
  string    strtop_acc_v3s_, strtop_mag_v3s_;

  Transform<double,3, Affine> magcal_trfm_;

  ofstream ofs_acc_, ofs_mag_;
  Vector3d acc_,mag_;
public:
  ros::Rate loop_rate_;
};

CallBackCalibrator::CallBackCalibrator():
    nh_p_("~"),
    loop_rate_(500)
{
  cout<<"*Entering constructor of cbc"<<endl;
  //setup dynamic reconfigure
  dynamic_reconfigure::Server<gcop_ros_est::CalibratorConfig>::CallbackType dyn_cb_f;
  dyn_cb_f = boost::bind(&CallBackCalibrator::cbReconfig, this, _1, _2);
  dyn_server_.setCallback(dyn_cb_f);

  // Setup topic names
  setupTopicsAndNames();

  //Setup publishers and Timers
  initSubsPubsAndTimers();

  //Setup rviz markers
  initRvizMarkers();

  //Setup output file stream
  string path_pkg = ros::package::getPath("gcop_ros_est");
  ofs_acc_.open(path_pkg+"/calib/acc_calib.dat", ofstream::out);
}

CallBackCalibrator::~CallBackCalibrator()
{

}

void
CallBackCalibrator::initRvizMarkers(void)
{
  if(config_.dyn_debug_on)
  {
    cout<<"*Initializing all rviz markers."<<endl;
  }
  int id=-1;
  //Marker for path
  id++;
  //Marker for displaying gps covariance
//  nh_p_.getParam("strfrm_gps_lcl", marker_cov_gps_lcl_.header.frame_id);
//  marker_cov_gps_lcl_.ns = "insekf";
//  marker_cov_gps_lcl_.id = id;
//  marker_cov_gps_lcl_.type = visualization_msgs::Marker::SPHERE;
//  marker_cov_gps_lcl_.action = visualization_msgs::Marker::ADD;
//  marker_cov_gps_lcl_.pose.position.x = 0;
//  marker_cov_gps_lcl_.pose.position.y = 0;
//  marker_cov_gps_lcl_.pose.position.z = 0;
//  marker_cov_gps_lcl_.pose.orientation.x = 0.0;
//  marker_cov_gps_lcl_.pose.orientation.y = 0.0;
//  marker_cov_gps_lcl_.pose.orientation.z = 0.0;
//  marker_cov_gps_lcl_.pose.orientation.w = 1.0;
//  marker_cov_gps_lcl_.color.r = 0.0;
//  marker_cov_gps_lcl_.color.g = 1.0;
//  marker_cov_gps_lcl_.color.b = 0.0;

}
void
CallBackCalibrator::sendMarkersAndTF(Vector3d xyz_gps)
{

}
void
CallBackCalibrator::cbSubImu(const sensor_msgs::Imu::ConstPtr& msg_imu)
{

}
void
CallBackCalibrator::cbSubMag(const sensor_msgs::MagneticField::ConstPtr& msg_mag)
{
  Vector3d mag_raw;
  mag_raw << msg_mag->magnetic_field.x, msg_mag->magnetic_field.y, msg_mag->magnetic_field.z;
  //mag_= magcal_trfm_*mag_raw;
}

void
CallBackCalibrator::cbSubAccV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg_acc_v3s)
{
  Vector3d acc_raw;
  acc_raw << msg_acc_v3s->vector.x,msg_acc_v3s->vector.y,msg_acc_v3s->vector.z;
  acc_=acc_raw;
  //acc_ = scale2si_acc_* (q_r2acc_ * acc_raw);
}
void
CallBackCalibrator::cbSubMagV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg_mag_v3s)
{
  Vector3d mag_raw;
  mag_raw << msg_mag_v3s->vector.x, msg_mag_v3s->vector.y, msg_mag_v3s->vector.z;
  //mag_= magcal_trfm_*mag_raw;
}

template<typename T>
void
CallBackCalibrator::rosParam2Mat(T &mat, std::string param)
{
  XmlRpc::XmlRpcValue mat_xml;
  nh_p_.getParam(param,mat_xml);
  xml2Mat(mat, mat_xml);
}

void
CallBackCalibrator::setupTopicsAndNames(void)
{
  if(config_.dyn_debug_on)
    cout<<"setting up topic names"<<endl;

  nh_p_.getParam("strtop_imu",strtop_imu_);
  nh_p_.getParam("strtop_mag",strtop_mag_);

  nh_p_.getParam("strtop_acc_v3s",strtop_acc_v3s_);
  nh_p_.getParam("strtop_mag_v3s",strtop_mag_v3s_);

  nh_p_.getParam("strfrm_map",strfrm_map_);
  nh_p_.getParam("strfrm_robot",strfrm_robot_);
  nh_p_.getParam("strfrm_gps_lcl",strfrm_gps_lcl_);

}


void
CallBackCalibrator::initSubsPubsAndTimers(void)
{
  //Publishers
  pub_viz_cov_ = nh_.advertise<visualization_msgs::Marker>( "visualization_marker", 0 );

  //Subscribers
  sub_imu_ = nh_.subscribe<sensor_msgs::Imu>(strtop_imu_,1000,&CallBackCalibrator::cbSubImu, this);
  sub_mag_ = nh_.subscribe<sensor_msgs::MagneticField>(strtop_mag_,1000,&CallBackCalibrator::cbSubMag, this);

  sub_acc_v3s_  = nh_.subscribe<geometry_msgs::Vector3Stamped>(strtop_acc_v3s_,1000,&CallBackCalibrator::cbSubAccV3S, this);
  sub_mag_v3s_  = nh_.subscribe<geometry_msgs::Vector3Stamped>(strtop_mag_v3s_,1000,&CallBackCalibrator::cbSubMagV3S, this);

  //Timers
}


void
CallBackCalibrator::cbReconfig(gcop_ros_est::CalibratorConfig &config, uint32_t level)
{
  cout<<"* Entering reconfig with level:"<<level<<endl;
  config_ = config;
  if(level==numeric_limits<uint32_t>::max())
  {
    cout<<"First time reconfig"<<endl;
  }
  else
  {
    if(config_.save_one_acc_reading)
    {
      writeOneAcc();
      config_.save_one_acc_reading=false;
    }
  }
  config = config_;
}

void
CallBackCalibrator::writeOneAcc(void)
{
  IOFormat full_precision(FullPrecision,0,",");
  ofs_acc_<<acc_.transpose().format(full_precision)<<endl;
}
//------------------------------------------------------------------------
//--------------------------------MAIN -----------------------------------
//------------------------------------------------------------------------

int main(int argc, char** argv)
{
  ros::init(argc,argv,"ekf_imu_mag_gps",ros::init_options::NoSigintHandler);
  signal(SIGINT,mySigIntHandler);

  CallBackCalibrator cbc;

  while(!g_shutdown_requested)
  {
    ros::spinOnce();
    cbc.loop_rate_.sleep();
  }
  return 0;

}









