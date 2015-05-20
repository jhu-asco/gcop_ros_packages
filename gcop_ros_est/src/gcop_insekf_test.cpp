// TODO: GPS gives more information than just x,y,z measurements. Use that.


#include "ros/ros.h"
#include <tf/transform_broadcaster.h>

// Dynamic reconfigure
#include <dynamic_reconfigure/server.h>
#include <gcop_ros_est/InsekfConfig.h>

// ROS standard messages
#include "std_msgs/String.h"
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/NavSatStatus.h>
#include <sensor_msgs/MagneticField.h>

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
gcop_ros_est::InsekfConfig g_config;
ros::Timer* g_p_timer_send_tf;
InsKalmanPredictor* g_p_kp_ins;
InsImuKalmanCorrector* g_p_kc_insimu;
InsGpsKalmanCorrector* g_p_kc_insgps;
InsGps<>* g_p_gps;
InsImu<3>* g_p_imu;
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
  g_p_imu->R.topLeftCorner<3,3>().diagonal().setConstant(msg_imu->linear_acceleration_covariance[0]);

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

void cbMsgGps(const sensor_msgs::NavSatFix::ConstPtr& msg_gps)
{
  //Update covariance
  //Normally the should remain the same and should be initialized once
  //Todo: initialize all elements in single line
  for(int i=0;i<3;i++)
    for(int j=0;j<3;j++)
      g_p_gps->R(i,j) = msg_gps->position_covariance[i*3+j]; //row major format

  if(g_config.set_gps_z_cov0)
  {
    g_p_gps->R(2,2) =0.0001;
  }

  double x_m_lcl, y_m_lcl, yaw_m_lcl; //local in reference to the LL point on the JHU map
  gis_common::gpsToLocal(x_m_lcl,y_m_lcl,msg_gps->latitude,msg_gps->longitude,g_config.lat0_deg, g_config.lon0_deg);

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
  // Send GPS data for visualization
  tf::Transform trfm2;
  trfm2.setOrigin( tf::Vector3(x_m_lcl,y_m_lcl, 0.0) );
  tf::Quaternion q2;
  q2.setRPY(0, 0, 0);
  trfm2.setRotation(q2);

  static tf::TransformBroadcaster br;
  if(g_config.enable_tf_publisher)
    br.sendTransform(tf::StampedTransform(trfm2, ros::Time::now(), g_config.name_map, g_config.name_gps_local));

}

void cbMsgMag(const sensor_msgs::MagneticField::ConstPtr& msg_mag)
{

  //g_p_imu->R.template bottomRightCorner<3,3>().diagonal().setConstant(srm*srm);
  std::cout<<"mag msg received"<<std::endl;
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

void cbReconfig(gcop_ros_est::InsekfConfig &config, uint32_t level)
{
  static bool first_time=true;
  static double period_tf_publish = config.period_tf_publish;

  if(first_time)
  {
    first_time=false;
  }
  else
  {

  }

  if(config.reinitialize_filter == true)
  {
    config.reinitialize_filter=false;
    g_first_gps_received = false;
  }

  if(period_tf_publish != config.period_tf_publish)
  {
    g_p_timer_send_tf->setPeriod(ros::Duration(config.period_tf_publish));
    period_tf_publish = config.period_tf_publish;
  }

  g_config = config;
}

void cbTimerSendImu(const ros::TimerEvent&)
{
  sensor_msgs::Imu msg_imu_ros;
  msg_imu_ros.angular_velocity.x    = g_u[0];
  msg_imu_ros.angular_velocity.y    = g_u[1];
  msg_imu_ros.angular_velocity.z    = g_u[2];
  msg_imu_ros.linear_acceleration.x = g_u[3];
  msg_imu_ros.linear_acceleration.y = g_u[4];
  msg_imu_ros.linear_acceleration.z = g_u[5];

  msg_imu_ros.header.frame_id = g_config.name_base_link;

 g_pub_imu.publish(msg_imu_ros);
}

void cbTimerPublishTF(const ros::TimerEvent&)
{
  //fused visualization
  tf::Transform trfm;
  trfm.setOrigin( tf::Vector3(g_xs.p[0],g_xs.p[1], g_xs.p[2]) );

  Vector4d wxyz;
  gcop::SO3::Instance().g2quat(wxyz, g_xs.R);
  tf::Quaternion q_true(wxyz[1],wxyz[2],wxyz[3],wxyz[0]);

  //TODO: yaw has to be set to something or use mag
  double yaw=0;
  Vector3d rpy;
  tf::Quaternion q_hack;

  if(g_config.enable_true_yaw)
  {
    trfm.setRotation(q_true);
  }
  else
  {
    gcop::SO3::Instance().g2q(rpy,g_xs.R);
    q_hack.setRPY(rpy[0],rpy[1],yaw);
    trfm.setRotation(q_hack);
  }

  static tf::TransformBroadcaster br;
  if(g_config.enable_tf_publisher)
    br.sendTransform(tf::StampedTransform(trfm, ros::Time::now(), g_config.name_map, g_config.name_base_link));

}
//-----------------------------------------------------------------------------------
//-----------------------------------MAIN--------------------------------------------
//-----------------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ros::init(argc,argv,"gcop_insekf_test",ros::init_options::NoSigintHandler);
  signal(SIGINT,mySigIntHandler);

  ros::NodeHandle nh;

  dynamic_reconfigure::Server<gcop_ros_est::InsekfConfig> server;
  dynamic_reconfigure::Server<gcop_ros_est::InsekfConfig>::CallbackType f;

  f = boost::bind(&cbReconfig, _1, _2);
  server.setCallback(f);

  ros::Subscriber sub_imu = nh.subscribe<sensor_msgs::Imu>("/imu",1000,cbMsgImu);
  ros::Subscriber sub_gps = nh.subscribe<sensor_msgs::NavSatFix>("/gps",1000,cbMsgGps);
  ros::Subscriber sub_mag = nh.subscribe<sensor_msgs::MagneticField>("/mag",1000,cbMsgMag);

  g_pub_imu = nh.advertise<sensor_msgs::Imu>(g_config.name_topic_imu, 3);

  ros::Timer timer_send_imu = nh.createTimer(ros::Duration(0.1), &cbTimerSendImu);
  ros::Timer timer_send_tf = nh.createTimer(ros::Duration(g_config.period_tf_publish), &cbTimerPublishTF);

  g_p_timer_send_tf = &timer_send_tf;

  Ins ins;
  InsGps<> gps;
  InsImu<3> imu;// if using mag and acc
  //InsImu<3> imu;//if not using mag

  g_p_gps = &gps;
  g_p_imu = &imu;



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


