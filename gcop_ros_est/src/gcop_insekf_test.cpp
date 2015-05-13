// TODO: GPS gives more information than just x,y,z measurements. Use that.


#include "ros/ros.h"
#include <tf/transform_broadcaster.h>

// ROS standard messages
#include "std_msgs/String.h"
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/NavSatStatus.h>

// ROS rampage messages
#include "rampage_msgs/UavCmds.h"
#include "rampage_msgs/ImuSimple.h"
#include "rampage_msgs/GpsSimple.h"

#include <gis_common/gps_to_local.h>

//GCOP includes
#include "gcop/utils.h"
#include "gcop/so3.h"
#include "gcop/kalmanpredictor.h"
#include "gcop/kalmancorrector.h"
#include "gcop/unscentedpredictor.h"
#include "gcop/unscentedcorrector.h"
#include "gcop/ins.h"
#include "gcop/insimu.h"
#include "gcop/insgps.h"

//Other includes
#include <iostream>
#include <signal.h>
//#include <rampage_estimator_and_controller/uav.h>

#include <Eigen/Dense>

using namespace std;
using namespace gcop;
using namespace Eigen;

typedef KalmanPredictor<InsState, 15, 6, Dynamic> InsKalmanPredictor;
typedef KalmanCorrector<InsState, 15, 6, Dynamic, Vector3d, 3> InsImuKalmanCorrector;
typedef KalmanCorrector<InsState, 15, 6, Dynamic, Vector3d, 3> InsGpsKalmanCorrector;

//-----------------------------------------------------------------------------------
//-----------------------------GLOBAL VARIABLES--------------------------------------
//-----------------------------------------------------------------------------------
sig_atomic_t g_shutdown_requested=0;
InsKalmanPredictor* g_p_kp_ins;
InsImuKalmanCorrector* g_p_kc_insimu;
InsGpsKalmanCorrector* g_p_kc_insgps;
InsState g_xs,g_x_temp;
bool g_first_gps_received = false;
ros::Time g_t_epoch_0;
Vector6d g_u;
double g_acc_gravity=9.81; //m/sec^2
Ins *g_p_ins;

ros::Publisher g_pub_imu; //for ros visualization


void mySigIntHandler(int signal)
{
  g_shutdown_requested=1;
}


void initInsStateSimple(InsState& xs0, double x_m_gps,double y_m_gps, double hdop_cm_gps)
{
  double pos_cov = hdop_cm_gps*hdop_cm_gps*1e-4;

  xs0.p = Vector3d(x_m_gps,y_m_gps,0.0);
  xs0.v = Vector3d(0.0,0.0,0.0);
  xs0.P.topLeftCorner<3,3>().diagonal().setConstant(.1);  // R
  xs0.P.block<3,3>(3,3).diagonal().setConstant(1e-2);     // bg
  xs0.P.block<3,3>(6,6).diagonal().setConstant(1e-10);    // ba
  xs0.P.block<3,3>(9,9).diagonal() <<pos_cov,pos_cov,0.1; // p
  xs0.P.block<3,3>(12,12).diagonal().setConstant(.04);    // v
}

void initInsState(InsState& xs0, double x_m_gps,double y_m_gps, sensor_msgs::NavSatFix::_position_covariance_type pos_cov)
{
  assert(pos_cov.size());
  xs0.p = Vector3d(x_m_gps,y_m_gps,0.0);
  xs0.v = Vector3d(0.0,0.0,0.0);
  xs0.P.topLeftCorner<3,3>().diagonal().setConstant(.1);  // R
  xs0.P.block<3,3>(3,3).diagonal().setConstant(1e-2);     // bg
  xs0.P.block<3,3>(6,6).diagonal().setConstant(1e-10);    // ba
  xs0.P.block<3,3>(9,9) << pos_cov[0],pos_cov[1],pos_cov[2]
                          ,pos_cov[3],pos_cov[4],pos_cov[5]
                          ,pos_cov[6],pos_cov[7],pos_cov[8]; // p
  xs0.P.block<3,3>(12,12).diagonal().setConstant(.04);    // v
}

void cbMsgImu(const sensor_msgs::Imu::ConstPtr& msg_imu)
{
  //ROS_INFO("I got IMU msg");
  static bool first_time = true;
  static ros::Time t_epoch_prev;

  if(g_first_gps_received)
  {
    if(first_time)
    {
      t_epoch_prev = g_t_epoch_0;
      first_time = false;
      cout << "*****t=0*****\n";
      cout<<"  g_t_epoch_0 sec:"<< g_t_epoch_0.sec <<"\t nsec:"<<g_t_epoch_0.nsec<<std::endl;
      cout << "t:"<<(msg_imu->header.stamp - g_t_epoch_0).toSec()<<endl;
      cout << "linear vel: " << g_xs.v.transpose() << endl;
      cout << "position: " << g_xs.p.transpose() << endl;
      cout << "Estim attitude:\n" << g_xs.R << endl;
    }
    else
    {
      //Put imu readings in frame of car
      double ax =  msg_imu->linear_acceleration.x;
      double ay =  msg_imu->linear_acceleration.y;
      double az =  msg_imu->linear_acceleration.z;
      double wx =  msg_imu->angular_velocity.x;
      double wy =  msg_imu->angular_velocity.y;
      double wz =  msg_imu->angular_velocity.z;

      Vector3d a(ax,ay,az);
      Vector3d w(wx,wy,wz);
      Vector6d u;
      u << w, a;
      g_u = u;

      Vector3d za(ax,ay,az);//same as a but defined the second time to preserve notation
      InsState xp;
      double t  = (msg_imu->header.stamp - g_t_epoch_0).toSec();
      double dt = (msg_imu->header.stamp -   t_epoch_prev).toSec();


      g_p_kp_ins->Predict(g_x_temp, t, g_xs, u, dt);
      g_p_kc_insimu->Correct(g_xs, t, g_x_temp, u, za);
      t_epoch_prev = msg_imu->header.stamp;

      //Display suff
      std::cout << "****************\n";
      std::cout << "t:"<<t<< "\tdt:"<<dt <<std::endl;
      std::cout << "u(w,a):"<<u.transpose()<< std::endl;
      std::cout << "linear vel: " << g_xs.v.transpose() << std::endl;
      std::cout << "position: " << g_xs.p.transpose() << std::endl;
      std::cout << "Estim attitude:\n" << g_xs.R << std::endl;
    }
  }
}

void cbMsgImuSimple(const rampage_msgs::ImuSimple::ConstPtr& msg_imu)
{
  //ROS_INFO("I got IMU msg");
  static bool first_time = true;
  static ros::Time t_epoch_prev;

  if(g_first_gps_received)
  {
    if(first_time)
    {
      t_epoch_prev = g_t_epoch_0;
      first_time = false;
      cout << "*****t=0*****\n";
      cout<<"  g_t_epoch_0 sec:"<< g_t_epoch_0.sec <<"\t nsec:"<<g_t_epoch_0.nsec<<std::endl;
      cout << "t:"<<(msg_imu->t_epoch - g_t_epoch_0).toSec()<<endl;
      cout << "linear vel: " << g_xs.v.transpose() << endl;
      cout << "position: " << g_xs.p.transpose() << endl;
      cout << "Estim attitude:\n" << g_xs.R << endl;
    }
    else
    {
      //Correcting for the fact that axis of the car and the IMU are different
      double ax =  msg_imu->ay*g_acc_gravity/1.0829567;
      double ay = -msg_imu->ax*g_acc_gravity/1.0829567;
      double az =  msg_imu->az*g_acc_gravity/1.0829567;
      double wx =  msg_imu->gy;
      double wy = -msg_imu->gx;
      double wz =  msg_imu->gz;

      Vector3d a(ax,ay,az);
      Vector3d w(wx,wy,wz);
      Vector6d u;
      u << w, a;
      g_u = u;

      Vector3d za(ax,ay,az);//same as a but defined the second time to preserve notation
      InsState xp;
      double t  = (msg_imu->t_epoch - g_t_epoch_0).toSec();
      double dt = (msg_imu->t_epoch -   t_epoch_prev).toSec();


      g_p_kp_ins->Predict(g_x_temp, t, g_xs, u, dt);
      g_p_kc_insimu->Correct(g_xs, t, g_x_temp, u, za);
      t_epoch_prev = msg_imu->t_epoch;

      //Display suff
      cout << "****************\n";
      cout << "t:"<<t<< "\tdt:"<<dt <<endl;
      cout << "u(w,a):"<<u.transpose()<< endl;
      cout << "linear vel: " << g_xs.v.transpose() << endl;
      cout << "position: " << g_xs.p.transpose() << endl;
      cout << "Estim attitude:\n" << g_xs.R << endl;
    }
  }
}

void cbMsgGps(const sensor_msgs::NavSatFix::ConstPtr& msg_gps)
{

  double x_m_lcl, y_m_lcl, yaw_m_lcl; //local in reference to the LL point on the JHU map
  gis_common::gpsToLocal(x_m_lcl,y_m_lcl,msg_gps->latitude,msg_gps->longitude);

  double rampage_yaw=0;
  //gis_common::groundCourseToYaw(rampage_yaw,msg_gps->ground_course);

  if(!g_first_gps_received)//perform the pose initialization
  {
    std::cout<<"first gps reading received"<<std::endl;
    initInsState(g_xs,x_m_lcl,y_m_lcl, msg_gps->position_covariance);
    g_t_epoch_0 = msg_gps->header.stamp;
    g_first_gps_received = true;
  }
  else//perform a sensor update
  {
    InsState xs;
    // noisy measurements of position
    Vector3d zp; zp << x_m_lcl, y_m_lcl,0 ;
    double t= (msg_gps->header.stamp - g_t_epoch_0).toSec();
    g_p_kc_insgps->Correct(g_x_temp, t, g_xs, g_u, zp);
    g_xs = g_x_temp;
  }


  //fused visualization
  tf::Transform trfm;
  trfm.setOrigin( tf::Vector3(g_xs.p[0],g_xs.p[1], g_xs.p[2]) );

//  Vector4d wxyz;
//  gcop::SO3::Instance().g2quat(wxyz, g_xs.R);
//  tf::Quaternion q(wxyz[1],wxyz[2],wxyz[3],wxyz[0]);

  Vector3d rpy;
  tf::Quaternion q;
  q.setRPY(rpy[0],rpy[1],rampage_yaw);
  gcop::SO3::Instance().g2q(rpy,g_xs.R);
  trfm.setRotation(q);

  // Send GPS data for visualization
  tf::Transform trfm2;
  trfm2.setOrigin( tf::Vector3(x_m_lcl,y_m_lcl, 0.0) );
  tf::Quaternion q2;
  q2.setRPY(0, 0, rampage_yaw);
  trfm2.setRotation(q2);

  static tf::TransformBroadcaster br;
  br.sendTransform(tf::StampedTransform(trfm, ros::Time::now(), "map", "rampage/base_link"));
  br.sendTransform(tf::StampedTransform(trfm2, ros::Time::now(), "map", "rampage/gps"));
}

void cbMsgGpsSimple(const rampage_msgs::GpsSimple::ConstPtr& msg_gps)
{

  double x_m_lcl, y_m_lcl, yaw_m_lcl; //local in reference to the LL point on the JHU map
  gis_common::gpsToLocal(x_m_lcl, y_m_lcl,msg_gps->lat_times_1e7_deg,msg_gps->lon_times_1e7_deg);

  double rampage_yaw;
  gis_common::groundCourseToYaw(rampage_yaw,msg_gps->ground_course);

  if(!g_first_gps_received)//perform the pose initialization
  {
    std::cout<<"first gps reading received"<<std::endl;
    initInsStateSimple(g_xs,x_m_lcl,y_m_lcl, msg_gps->hdop_cm);
    g_t_epoch_0 = msg_gps->t_epoch;
    g_first_gps_received = true;
  }
  else//perform a sensor update
  {
    InsState xs;
    // noisy measurements of position
    Vector3d zp; zp << x_m_lcl, y_m_lcl,0 ;
    double t= (msg_gps->t_epoch - g_t_epoch_0).toSec();
    g_p_kc_insgps->Correct(g_x_temp, t, g_xs, g_u, zp);
    g_xs = g_x_temp;
  }


  //fused visualization
  tf::Transform trfm;
  trfm.setOrigin( tf::Vector3(g_xs.p[0],g_xs.p[1], g_xs.p[2]) );

//  Vector4d wxyz;
//  gcop::SO3::Instance().g2quat(wxyz, g_xs.R);
//  tf::Quaternion q(wxyz[1],wxyz[2],wxyz[3],wxyz[0]);

  Vector3d rpy;
  tf::Quaternion q;
  q.setRPY(rpy[0],rpy[1],rampage_yaw);
  gcop::SO3::Instance().g2q(rpy,g_xs.R);
  trfm.setRotation(q);

  // Send GPS data for visualization
  tf::Transform trfm2;
  trfm2.setOrigin( tf::Vector3(x_m_lcl,y_m_lcl, 0.0) );
  tf::Quaternion q2;
  q2.setRPY(0, 0, rampage_yaw);
  trfm2.setRotation(q2);

  static tf::TransformBroadcaster br;
  br.sendTransform(tf::StampedTransform(trfm, ros::Time::now(), "map", "rampage/base_link"));
  br.sendTransform(tf::StampedTransform(trfm2, ros::Time::now(), "map", "rampage/gps"));
}
/**
 * Start a timer
 * @param timer timer
 */
inline void timer_start(struct timeval &timer)
{
  gettimeofday(&timer, 0);
}

/**
 * Get elapsed time in microseconds
 * Timer should be started with timer_start(timer)
 * @param timer timer
 * @return elapsed time
 */
inline long timer_us(struct timeval &timer)
{
  struct timeval now;
  gettimeofday(&now, 0);
  return (now.tv_sec - timer.tv_sec)*1000000 + now.tv_usec - timer.tv_usec;
}

void sendImu(const ros::TimerEvent&)
{
  sensor_msgs::Imu msg_imu_ros;
  msg_imu_ros.angular_velocity.x    = g_u[0];
  msg_imu_ros.angular_velocity.y    = g_u[1];
  msg_imu_ros.angular_velocity.z    = g_u[2];
  msg_imu_ros.linear_acceleration.x = g_u[3];
  msg_imu_ros.linear_acceleration.y = g_u[4];
  msg_imu_ros.linear_acceleration.z = g_u[5];

  msg_imu_ros.header.frame_id = "rampage/base_link";

 g_pub_imu.publish(msg_imu_ros);
}

//-----------------------------------------------------------------------------------
//-----------------------------------MAIN--------------------------------------------
//-----------------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ros::init(argc,argv,"rampage_gcop_insekf_test",ros::init_options::NoSigintHandler);
  signal(SIGINT,mySigIntHandler);

  ros::NodeHandle nh;

//  ros::Subscriber sub_imu_simple = nh.subscribe<rampage_msgs::ImuSimple>("imu_simple",1000,cbMsgImuSimple);
//  ros::Subscriber sub_gps_simple = nh.subscribe<rampage_msgs::GpsSimple>("gps_simple",1000,cbMsgGpsSimple);
  ros::Subscriber sub_imu = nh.subscribe<sensor_msgs::Imu>("imu",1000,cbMsgImu);
  ros::Subscriber sub_gps = nh.subscribe<sensor_msgs::NavSatFix>("gps",1000,cbMsgGps);

  g_pub_imu = nh.advertise<sensor_msgs::Imu>("rampage_imu", 3);

  ros::Timer timer_send_imu = nh.createTimer(ros::Duration(0.1), &sendImu);

  Ins ins;
  bool mag = false;
  InsImu<3> imu;
  InsGps<> gps;


  gps.sxy = 3;
  gps.sz = 1;
  gps.R(0,0) = gps.sxy*gps.sxy;
  gps.R(1,1) = gps.sxy*gps.sxy;
  gps.R(2,2) = gps.sz*gps.sz;

  ins.g0[2]=g_acc_gravity;
  imu.a0[2]=g_acc_gravity;

  //Instantiate the kalman predictor and correctors
  InsKalmanPredictor kp_ins(ins);
  InsImuKalmanCorrector kc_insimu(ins.X, imu);
  InsGpsKalmanCorrector kc_insgps(ins.X, gps);

  g_p_kp_ins    = &kp_ins;
  g_p_kc_insimu = &kc_insimu;
  g_p_kc_insgps = &kc_insgps;

  ros::Rate loop_rate(100);

  while(!g_shutdown_requested)
  {
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;
}


