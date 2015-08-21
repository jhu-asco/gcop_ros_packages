/*
 * sim_ins_mag_gps.cpp
 *
 *  Created on: Jul 20, 2015
 *      Author: subhransu
 */

// ROS relevant includes
#include <ros/ros.h>
#include <ros/package.h>
#include <tf_conversions/tf_eigen.h>
#include <dynamic_reconfigure/server.h>
#include <gcop_ros_est/SimSensConfig.h>


// ROS standard messages
#include <std_msgs/String.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/MagneticField.h>

//gcop_comm msgs
#include <gcop_comm/State.h>
#include <gcop_comm/CtrlTraj.h>
#include <gcop_comm/Trajectory_req.h>

//GCOP includes
#include <gcop/utils.h>
#include <gcop/so3.h>
#include <gcop/kalmanpredictor.h>
#include <gcop/kalmancorrector.h>
#include <gcop/ins.h>
#include <gcop/insimu.h>
#include <gcop/insgps.h>
#include <gcop/insmag.h>

//GIS
#include "llh_enu_cov.h"

//Other includes
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <limits>
#include <algorithm>

#include <Eigen/Dense>

//-------------------------------------------------------------------------
//-----------------------NAME SPACES ---------------------------------
//-------------------------------------------------------------------------
using namespace std;
using namespace gcop;
using namespace Eigen;

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


//------------------------------------------------------------------------
//-----------------------------CALLBACK CLASS ----------------------------
//------------------------------------------------------------------------

class CallBackSimSens
{
public:
  typedef Matrix<double, 4, 4> Matrix4d;
  typedef Matrix<double, 4, 1> Vector4d;

public:
  CallBackSimSens();
  ~CallBackSimSens();

private:
  void cbTimerGeneral(const ros::TimerEvent& event);
  void cbTimerStateEvol(const ros::TimerEvent& event);
  void cbTimerGps(const ros::TimerEvent& event);
  void cbTimerImu(const ros::TimerEvent& event);
  void cbTimerMag(const ros::TimerEvent& event);
  void cbTimerAccV3S(const ros::TimerEvent& event);
  void cbTimerMagV3S(const ros::TimerEvent& event);
  void cbTimerGyroV3S(const ros::TimerEvent& event);


  void cbReconfig(gcop_ros_est::SimSensConfig &config, uint32_t level);
  void setFromParamsConfig(void);

  void setupTopics(void);
  double maxRate(void);
  void initPubsAndTimers(void);
  void editPubsAndTimersByOutput(void);
  void editTimersByRate(void);

  void initSensParams(void);
  void updateNoiseParams(void);
  void initCtrls(void);
  void updateCtrls(void);
  void initInsState(void);



private:
  //ROS relavant members
  ros::NodeHandle nh_, nh_p_;
  gcop_ros_est::SimSensConfig config_;
  dynamic_reconfigure::Server<gcop_ros_est::SimSensConfig> dyn_server_;
  bool dyn_write_;      //save config_ to config
  ros::Timer timer_general_, timer_state_evol_;

  ros::Timer     timer_gps_;
  ros::Publisher pub_gps_;
  std::string    strtop_gps_;

  ros::Timer     timer_imu_ , timer_mag_;//for RosDefault message type
  ros::Publisher pub_imu_   , pub_mag_;
  std::string    strtop_imu_, strtop_mag_;

  ros::Timer     timer_gyro_v3s_ , timer_acc_v3s_ , timer_mag_v3s_;//for Vector3Stamped type
  ros::Publisher pub_gyro_v3s_   , pub_acc_v3s_   , pub_mag_v3s_;
  std::string    strtop_gyro_v3s_, strtop_acc_v3s_, strtop_mag_v3s_;

  //ins state evolution and sensor messages
  InsState x_;
  Ins ins_;
  double t_;
  ros::Time t_epoch_start_;

  InsImu<3> sens_imu_;
  InsImu<6> sens_imu_mag_;
  InsGps<>    sens_gps_;
  InsMag<>    sens_mag_;
  Vector3d  ctrl_w_true_;
  Vector3d  ctrl_a_true_;
  Vector3d  ctrl_w_drft_;
  Vector3d  ctrl_a_drft_;

public:
  ros::Rate loop_rate_;
};

CallBackSimSens::CallBackSimSens():
    nh_p_("~"),
    loop_rate_(200),
    t_(0)
{
  cout<<"*Entering constructor of cbc"<<endl;
  //setup dynamic reconfigure
  dynamic_reconfigure::Server<gcop_ros_est::SimSensConfig>::CallbackType dyn_cb_f;
  dyn_cb_f = boost::bind(&CallBackSimSens::cbReconfig, this, _1, _2);
  dyn_server_.setCallback(dyn_cb_f);

  // Setup topic names
  setupTopics();

  //Setup publishers and Timers
  initPubsAndTimers();

  //Setup ins state and evolution
  initSensParams();
  initCtrls();
  initInsState();
}

CallBackSimSens::~CallBackSimSens()
{

}

void
CallBackSimSens::cbTimerGeneral(const ros::TimerEvent& event)
{

}

void
CallBackSimSens::cbTimerStateEvol(const ros::TimerEvent& event)
{
  // generate true
  InsState x_b;
  if(!event.last_real.isZero()) //i.e. not the first time
  {
    double dt = (event.current_real - event.last_real).toSec();
    t_ +=dt;
    Vector6d ctrl_ut_drft;ctrl_ut_drft<<ctrl_w_drft_,ctrl_a_drft_;
    ins_.Step(x_b, t_, x_, ctrl_ut_drft, dt);
    x_ = x_b;
  }
  else
    t_epoch_start_ = event.current_real;

  updateCtrls();
}

void
CallBackSimSens::cbTimerGps(const ros::TimerEvent& event)
{
  static int seq=0;
  if(event.last_real.isZero())
    seq=0;
  seq++;
  sensor_msgs::NavSatFix msg_gps;
  msg_gps.header.frame_id = "simsens";
  msg_gps.header.seq = seq;
  msg_gps.header.stamp =t_epoch_start_ + ros::Duration(t_);

  msg_gps.altitude = x_.p(2) + config_.dyn_alt0_m;

  Vector3d llh0_ddm; llh0_ddm << config_.dyn_lat0_deg,config_.dyn_lon0_deg,0.0;
  Vector3d enu;      enu << x_.p(0),x_.p(1),0;
  Vector3d llh_ddm;  enuSI2llhDDM(llh_ddm, llh0_ddm, enu);

  msg_gps.latitude = llh_ddm(0);
  msg_gps.longitude= llh_ddm(1);

  pub_gps_.publish(msg_gps);
}
void
CallBackSimSens::cbTimerImu(const ros::TimerEvent& event)
{
  static int seq=0;
  if(event.last_real.isZero())
    seq=0;
  seq++;
  sensor_msgs::Imu msg_imu;
  msg_imu.header.frame_id = "simsens";
  msg_imu.header.seq = seq;
  msg_imu.header.stamp =t_epoch_start_ + ros::Duration(t_);

  Vector3d ctrl_a_drfty_noisy = ctrl_a_drft_ + sqrt(config_.dyn_cov_acc)*randn()*Vector3d::Ones();
  //TODO:make use of sensor class to produce this data and make sure it matches with your guess
  Vector3d ctrl_w_drfty_noisy = ctrl_w_drft_ + sqrt(config_.dyn_cov_gyro)*randn()*Vector3d::Ones();
  msg_imu.linear_acceleration.x = ctrl_a_drfty_noisy(0);
  msg_imu.linear_acceleration.y = ctrl_a_drfty_noisy(1);
  msg_imu.linear_acceleration.z = ctrl_a_drfty_noisy(2);
  msg_imu.angular_velocity.x = ctrl_w_drfty_noisy(0);
  msg_imu.angular_velocity.y = ctrl_w_drfty_noisy(1);
  msg_imu.angular_velocity.z = ctrl_w_drfty_noisy(2);

  pub_imu_.publish(msg_imu);
}
void
CallBackSimSens::cbTimerMag(const ros::TimerEvent& event)
{
  static int seq=0;
  if(event.last_real.isZero())
    seq=0;
  seq++;

  sensor_msgs::MagneticField msg_mag;
  msg_mag.header.frame_id = "simsens";
  msg_mag.header.seq = seq;
  msg_mag.header.stamp =t_epoch_start_ + ros::Duration(t_);

  Vector3d mag_noisy = x_.R.transpose()*sens_mag_.m0 + sqrt(config_.dyn_cov_mag)*randn()*Vector3d::Ones();
  msg_mag.magnetic_field.x = mag_noisy(0);
  msg_mag.magnetic_field.y = mag_noisy(1);
  msg_mag.magnetic_field.z = mag_noisy(2);
  pub_mag_.publish(msg_mag);

}
void
CallBackSimSens::cbTimerAccV3S(const ros::TimerEvent& event)
{
  //TODO:reset sequence back to zero when output type is changed
  static int seq=0;
  if(event.last_real.isZero())
    seq=0;
  seq++;
  geometry_msgs::Vector3Stamped msg_acc;
  msg_acc.header.frame_id = "simsens";
  msg_acc.header.seq = seq;
  msg_acc.header.stamp =t_epoch_start_ + ros::Duration(t_);

  //TODO:make use of sensor class to produce this data and make sure it matches with your guess
  Vector3d meas;
  Vector6d ctrl_u_drft; ctrl_u_drft << ctrl_w_drft_, ctrl_a_drft_;
  sens_imu_(meas, t_, x_, ctrl_u_drft);
  cout<<"meas:\n"<<meas<<endl;
  cout<<"ctrl_a_drft_:\n"<<ctrl_a_drft_<<endl;
  getchar();
  Vector3d ctrl_a_drfty_noisy = ctrl_a_drft_ + sqrt(config_.dyn_cov_acc)*randn()*Vector3d::Ones();
  msg_acc.vector.x = ctrl_a_drfty_noisy(0);
  msg_acc.vector.y = ctrl_a_drfty_noisy(1);
  msg_acc.vector.z = ctrl_a_drfty_noisy(2);
  pub_acc_v3s_.publish(msg_acc);
}
void
CallBackSimSens::cbTimerMagV3S(const ros::TimerEvent& event)
{
  //TODO:reset sequence back to zero when output type is changed
  //TODO: use the geographic library to provide the data
  static int seq=0;
  if(event.last_real.isZero())
    seq=0;
  seq++;
  geometry_msgs::Vector3Stamped msg_mag;
  msg_mag.header.frame_id = "simsens";
  msg_mag.header.seq = seq;
  msg_mag.header.stamp =t_epoch_start_ + ros::Duration(t_);

  Vector3d mag_noisy = x_.R.transpose()*sens_mag_.m0 + sqrt(config_.dyn_cov_mag)*randn()*Vector3d::Ones();
  msg_mag.vector.x = mag_noisy(0);
  msg_mag.vector.y = mag_noisy(1);
  msg_mag.vector.z = mag_noisy(2);
  pub_mag_v3s_.publish(msg_mag);
}
void
CallBackSimSens::cbTimerGyroV3S(const ros::TimerEvent& event)
{
  //TODO:reset sequence back to zero when output type is changed
  static int seq=0;
  if(event.last_real.isZero())
    seq=0;
  seq++;
  geometry_msgs::Vector3Stamped msg_gyro;
  msg_gyro.header.frame_id = "simsens";
  msg_gyro.header.seq = seq;
  msg_gyro.header.stamp =t_epoch_start_ + ros::Duration(t_);

  //TODO:make use of sensor class to produce this data and make sure it matches with your guess
  Vector3d ctrl_w_drfty_noisy = ctrl_w_drft_ + sqrt(config_.dyn_cov_gyro)*randn()*Vector3d::Ones();
  msg_gyro.vector.x = ctrl_w_drfty_noisy(0);
  msg_gyro.vector.y = ctrl_w_drfty_noisy(1);
  msg_gyro.vector.z = ctrl_w_drfty_noisy(2);
  pub_gyro_v3s_.publish(msg_gyro);
}

void
CallBackSimSens::setupTopics(void)
{
  nh_p_.getParam("strtop_gps",strtop_gps_);

  nh_p_.getParam("strtop_imu",strtop_imu_);
  nh_p_.getParam("strtop_mag",strtop_mag_);

  nh_p_.getParam("strtop_gyro_v3s",strtop_gyro_v3s_);
  nh_p_.getParam("strtop_acc_v3s",strtop_acc_v3s_);
  nh_p_.getParam("strtop_mag_v3s",strtop_mag_v3s_);
}

double
CallBackSimSens::maxRate(void)
{
  double rates[] = {config_.dyn_rate_gps, config_.dyn_rate_imu,
                    config_.dyn_rate_mag, config_.dyn_rate_acc, config_.dyn_rate_gyro};
  return *max_element(rates, rates+5);
}
void
CallBackSimSens::initPubsAndTimers(void)
{

  //Timers
  timer_general_ = nh_.createTimer(ros::Duration(0.05), &CallBackSimSens::cbTimerGeneral, this);
  timer_state_evol_ = nh_.createTimer(ros::Duration(1/(4*maxRate())), &CallBackSimSens::cbTimerStateEvol, this);
  timer_general_.start();
  timer_state_evol_.start();


  timer_gps_      = nh_.createTimer(ros::Duration(1/config_.dyn_rate_gps), &CallBackSimSens::cbTimerGps,    this);
  timer_imu_      = nh_.createTimer(ros::Duration(1/config_.dyn_rate_imu), &CallBackSimSens::cbTimerImu,    this);
  timer_mag_      = nh_.createTimer(ros::Duration(1/config_.dyn_rate_mag), &CallBackSimSens::cbTimerMag,    this);
  timer_acc_v3s_  = nh_.createTimer(ros::Duration(1/config_.dyn_rate_acc), &CallBackSimSens::cbTimerAccV3S, this);
  timer_mag_v3s_  = nh_.createTimer(ros::Duration(1/config_.dyn_rate_mag), &CallBackSimSens::cbTimerMagV3S, this);
  timer_gyro_v3s_ = nh_.createTimer(ros::Duration(1/config_.dyn_rate_gyro), &CallBackSimSens::cbTimerGyroV3S,this);

  //Publishers
  pub_gps_  = nh_.advertise<sensor_msgs::NavSatFix>(strtop_gps_,0);
  editPubsAndTimersByOutput();
}

void
CallBackSimSens::editPubsAndTimersByOutput(void)
{
  if(config_.dyn_debug_on)
    cout<<"Setting up publishers and editing timers based on output types"<<endl;
  switch(config_.dyn_type_sensor_msg)
  {
    case 0://V3S
      pub_imu_.shutdown();
      pub_mag_.shutdown();
      pub_acc_v3s_  = nh_.advertise<geometry_msgs::Vector3Stamped>( strtop_acc_v3s_, 0 );
      pub_gyro_v3s_ = nh_.advertise<geometry_msgs::Vector3Stamped>( strtop_gyro_v3s_, 0 );
      pub_mag_v3s_  = nh_.advertise<geometry_msgs::Vector3Stamped>( strtop_mag_v3s_, 0 );

      timer_imu_.stop();
      timer_mag_.stop();
      timer_acc_v3s_.start();
      timer_mag_v3s_.start();
      timer_gyro_v3s_.start();
      break;
    case 1://RosDefault
      pub_acc_v3s_.shutdown();
      pub_gyro_v3s_.shutdown();
      pub_mag_v3s_.shutdown();
      pub_imu_  = nh_.advertise<sensor_msgs::Imu>( strtop_imu_,0);
      pub_mag_  = nh_.advertise<sensor_msgs::MagneticField>(strtop_mag_,0);

      timer_acc_v3s_.stop();
      timer_mag_v3s_.stop();
      timer_gyro_v3s_.stop();
      timer_imu_.start();
      timer_mag_.start();
      break;
    default:
      assert(0);
      break;
  }
}

void
CallBackSimSens::editTimersByRate(void)
{
  if(config_.dyn_debug_on)
    cout<<"Editing timers with new rates"<<endl;
  timer_state_evol_.setPeriod(ros::Duration(1/(4*maxRate())));
  timer_gps_.setPeriod(ros::Duration(1/config_.dyn_rate_gps));
  timer_imu_.setPeriod(ros::Duration(1/config_.dyn_rate_imu));
  timer_mag_.setPeriod(ros::Duration(1/config_.dyn_rate_mag));
  timer_acc_v3s_.setPeriod(ros::Duration(1/config_.dyn_rate_acc));
  timer_mag_v3s_.setPeriod(ros::Duration(1/config_.dyn_rate_mag));
  timer_gyro_v3s_.setPeriod(ros::Duration(1/config_.dyn_rate_gyro));
}
void
CallBackSimSens::initSensParams(void)
{
  //Set reference
  //insimu<6> Accel reference
  nh_p_.getParam("a0x",sens_imu_.a0(0));
  nh_p_.getParam("a0y",sens_imu_.a0(1));
  nh_p_.getParam("a0z",sens_imu_.a0(2));

  //insimu<6> mag reference
  nh_p_.getParam("m0x",sens_mag_.m0(0));
  nh_p_.getParam("m0y",sens_mag_.m0(1));
  nh_p_.getParam("m0z",sens_mag_.m0(2));

  //insmag mag reference
  nh_p_.getParam("m0x",sens_imu_.m0(0));
  nh_p_.getParam("m0y",sens_imu_.m0(1));
  nh_p_.getParam("m0z",sens_imu_.m0(2));

  //Edit sens noise params
  updateNoiseParams();
}

void
CallBackSimSens::updateNoiseParams(void)
{
  if(config_.dyn_debug_on)
    cout<<"Updating the noise params"<<endl;
  //setNoise
  //gps noise
  sens_gps_.sxy = sqrt(config_.dyn_cov_gps_xy);
  sens_gps_.sz = sqrt(config_.dyn_cov_gps_z);
  //accel noise
  sens_imu_.sra = sqrt(config_.dyn_cov_acc);
  //mag noise
  sens_imu_.srm = sqrt(config_.dyn_cov_mag);
  //gyro noise(missing?)

  //setNoise
  //gps noise
  sens_gps_.sxy = sqrt(config_.dyn_cov_gps_xy);
  sens_gps_.sz = sqrt(config_.dyn_cov_gps_z);
  //accel noise
  sens_imu_.sra = sqrt(config_.dyn_cov_acc);
  //mag noise
  sens_imu_.srm = sqrt(config_.dyn_cov_mag);
  //gyro noise(missing?)

}
void
CallBackSimSens::initCtrls(void)
{
  ctrl_w_true_<<0.2, 0.1, 0.0;// True angular velocity
  ctrl_a_true_.setZero();     // True acceleration but without gravity
}

void
CallBackSimSens::updateCtrls(void)
{
  ctrl_w_drft_ = ctrl_w_true_ + x_.bg;                 // Simulated sensor reading w/o noise
  ctrl_a_drft_ = x_.R.transpose()*sens_imu_.a0 + ctrl_a_true_+ x_.ba; //Simulated sensor reading w/o noise
}
void
CallBackSimSens::initInsState(void)
{
  //initState
  x_.R.setIdentity();
  x_.p.setZero();
  x_.v << 1,0,0;

  nh_p_.getParam("bgx",x_.bg(0));
  nh_p_.getParam("bgy",x_.bg(1));
  nh_p_.getParam("bgz",x_.bg(2));

  nh_p_.getParam("bax",x_.ba(0));
  nh_p_.getParam("bay",x_.ba(1));
  nh_p_.getParam("baz",x_.ba(2));

}

void
CallBackSimSens::setFromParamsConfig()
{
  cout<<"Setting Dynamic Reconfigure params from parameter server params"<<endl;
  nh_p_.getParam("tmax",config_.dyn_tmax);

  // Noise parameters
  nh_p_.getParam("cov_mag",config_.dyn_cov_mag);
  nh_p_.getParam("cov_acc",config_.dyn_cov_acc);
  nh_p_.getParam("cov_gyro",config_.dyn_cov_gyro);
  nh_p_.getParam("cov_gps_xy", config_.dyn_cov_gps_xy);
  nh_p_.getParam("cov_gps_z", config_.dyn_cov_gps_z);

  // Data rate
  nh_p_.getParam("rate_gps",config_.dyn_rate_gps);
  nh_p_.getParam("rate_imu",config_.dyn_rate_imu);
  nh_p_.getParam("rate_mag",config_.dyn_rate_mag);
  nh_p_.getParam("rate_acc",config_.dyn_rate_acc);
  nh_p_.getParam("rate_gyro",config_.dyn_rate_gyro);

  // Origin
  nh_p_.getParam("lat0_deg",config_.dyn_lat0_deg);
  nh_p_.getParam("lon0_deg",config_.dyn_lon0_deg);
  nh_p_.getParam("alt0_m",  config_.dyn_alt0_m);

  // output format either Vector3Stamped or RosDefault
  nh_p_.getParam("type_sensor_msg",config_.dyn_type_sensor_msg);
}

void
CallBackSimSens::cbReconfig(gcop_ros_est::SimSensConfig &config, uint32_t level)
{
  cout<<"* Entering reconfig with level:"<<level<<endl;
  config_ = config;
  if(level==numeric_limits<uint32_t>::max())
    setFromParamsConfig();
  else
  {
    //Check for change in output format
    if((level&2)==2)
      editPubsAndTimersByOutput();
    //Check for change in covariance
    if((level&4)==4)
      updateNoiseParams();
    //Check for change in rate
    if((level&8)==8)
      editTimersByRate();
  }
  config = config_;
}

//------------------------------------------------------------------------
//--------------------------------MAIN -----------------------------------
//------------------------------------------------------------------------

int main(int argc, char** argv)
{
  ros::init(argc,argv,"sim_sens",ros::init_options::NoSigintHandler);
  signal(SIGINT,mySigIntHandler);

  CallBackSimSens cbc;

  while(!g_shutdown_requested)
  {
    ros::spinOnce();
    cbc.loop_rate_.sleep();
  }
  return 0;

}









