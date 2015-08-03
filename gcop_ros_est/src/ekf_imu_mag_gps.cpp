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
#include <gcop_ros_est/InsekfConfig.h>


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

// gps, utm, local coord conversions
#include <gis_common/gps_to_local.h>

//GCOP includes
#include <gcop/utils.h>
#include <gcop/so3.h>
#include <gcop/kalmanpredictor.h>
#include <gcop/kalmancorrector.h>
#include <gcop/ins.h>
#include <gcop/insimu.h>
#include <gcop/insgps.h>
#include <gcop/insmag.h>

//Other includes
#include <numeric>
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <assert.h>
#include <queue>
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
using namespace gcop;
using namespace Eigen;


//-------------------------------------------------------------------------
//-----------------------TYPEDEFS ---------------------------------
//-------------------------------------------------------------------------

typedef KalmanPredictor<InsState, 15, 6, Dynamic> InsKalmanPredictor;
typedef KalmanCorrector<InsState, 15, 6, Dynamic, Vector3d, 3> InsImuKalmanCorrector;
typedef KalmanCorrector<InsState, 15, 6, Dynamic, Vector6d, 6> InsImuMagKalmanCorrector;
typedef KalmanCorrector<InsState, 15, 6, Dynamic, Vector3d, 3> InsMagKalmanCorrector;
typedef KalmanCorrector<InsState, 15, 6, Dynamic, Vector3d, 3> InsGpsKalmanCorrector;

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


class CallBackInsEkf
{
public:
  typedef Matrix<double, 4, 4> Matrix4d;
  typedef Matrix<double, 4, 1> Vector4d;
  typedef Matrix<double, 7, 1> Vector7d;

/**
 * Class for checking if filter is ready to be used or not
 */
class FilterReadiness
{
public:
    bool is_ready_p_;
    bool is_ready_p_cov_;
    bool is_ready_v_;
    bool is_ready_v_cov_;
    bool is_ready_R_;
    bool is_ready_R_cov_;
    bool is_ready_bg_;
    bool is_ready_bg_cov_;

    //Needed for finding the initial R matrix and it's covariance
    bool is_ready_gyr_;
    bool is_ready_mag_;
    bool is_ready_acc_;
    bool is_ready_gps_;

    int n_bg_;//number of gyro reading to average
    uint32_t max_buffer_size_;
private:
    InsState& x_;
    Ins&      ins_;
public:

    FilterReadiness(Ins& ins, InsState& x):ins_(ins),x_(x),n_bg_(500), max_buffer_size_(100)
    {
      is_ready_p_       =false;
      is_ready_p_cov_   =false;
      is_ready_v_       =false;
      is_ready_v_cov_   =false;
      is_ready_R_       =false;
      is_ready_R_cov_   =false;
      is_ready_bg_      =false;
      is_ready_bg_cov_  =false;

      is_ready_mag_     =false;
      is_ready_acc_     =false;
      is_ready_gyr_     =false;
      is_ready_gps_     =false;
    }
    ~FilterReadiness(){}
public:

    bool isReady(void)
    {
      return  is_ready_p_ &&   is_ready_p_cov_  && is_ready_v_ && is_ready_v_cov_ &&
              is_ready_R_ &&   is_ready_R_cov_  && is_ready_bg_&& is_ready_bg_cov_;
    }

    bool canInitR(void)
    {
      return is_ready_acc_ && is_ready_mag_;
    }
    void trySetRAndCov(void)
    {
      x_.R.setIdentity();
      x_.P.topLeftCorner<3,3>().diagonal().setConstant(.1);  // R
      is_ready_R_     = true;
      is_ready_R_cov_ = true;
      x_.ba.setZero();
      x_.P.block<3,3>(6,6).diagonal().setConstant(1e-10);    // ba
      trySetvAndCov();
    }

    void trySetpAndCov(Vector3d& xyz_gps, Matrix3d& cov_gps)
    {
      x_.p = xyz_gps;
      x_.P.block<3,3>(9,9) = cov_gps;
      is_ready_p_    = true;
      is_ready_p_cov_ = true;
    }

    /**
     * Estimates the bias of the gyro by averaging the readings over 1 second
     * @param gyro: The sensor value
     * @param reset: Reset the time elapsed to 0 and start averaging again
     */

    void trySetvAndCov(void)
    {
      //We assume that when the filter is started then the car is not moving
      x_.v.setZero();
      x_.P.block<3,3>(12,12).diagonal().setConstant(.001);
      is_ready_v_ = true;
      is_ready_v_cov_ = true;
    }

    void trySetGyroBiasAndCov(const Vector3d &gyro,bool reset)
    {
      static ros::Time t_epoch_first;
      static Vector3d bg;
      static vector<Vector3d> gyro_buffer;

      if(reset)
      {
        bg.setZero();
        t_epoch_first = ros::Time::now();
        gyro_buffer.reserve(n_bg_);
      }
      else if(gyro_buffer.size() < n_bg_)
        gyro_buffer.push_back(gyro);
      else
      {
        Matrix3d cov;
        avgVectInStl(x_.bg,cov,gyro_buffer);
        x_.P.block<3,3>(3,3) = cov;
        ins_.sv = sqrt(cov(0,0));
        is_ready_bg_=true;
        is_ready_bg_cov_ = true;
        gyro_buffer.clear();
      }
    }


    /**
     * Finds the average of Eigen::Vectors stored in stl
     * @param buffer
     * @param online: Online might be slower, Offline may lead to numerical instability
     * @return
     */
    template<typename C, typename M>
    void avgVectInStl(typename C::value_type& avg,M& cov, const C& buffer)
    {
      avg.setZero();
      cov.setZero();
      int n=buffer.size();
      assert(cov.rows()==cov.cols() && cov.rows() == avg.rows() );
      for_each(buffer.rbegin(), buffer.rend(), [&](typename C::value_type val){avg += val;cov+=val*val.transpose(); });
      avg = avg/double(n);
      cov = (cov - avg*avg.transpose()*n)/double(n-1);
    }

};
public:
  CallBackInsEkf();
  ~CallBackInsEkf();

private:
  void cbTimerGeneral(const ros::TimerEvent& event);
  void cbTimerFakeGPS(const ros::TimerEvent& event);
  void cbTimerPublishTF(const ros::TimerEvent& event);
  void cbSubGps(const sensor_msgs::NavSatFix::ConstPtr& msg_gps);
  void cbSubImu(const sensor_msgs::Imu::ConstPtr& msg_imu);
  void cbSubMag(const sensor_msgs::MagneticField::ConstPtr& msg_mag);
  void cbSubAccV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg_acc_v3s);
  void cbSubMagV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg_mag_v3s);
  void cbSubGyrV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg_gyr_v3s);

  void cbReconfig(gcop_ros_est::InsekfConfig &config, uint32_t level);
  void setFromParamsConfig(void);

  void initRvizMarkers(void);
  void sendMarkersAndTF(Vector3d xyz_gps);

  template<typename T>
  void rosParam2Mat(T &mat, std::string param);

  void setupTopicsAndNames(void);
  void initSubsPubsAndTimers(void);

  void initSensParams(void);
  void updateNoiseFromConfig(void);
  void initMagCalib(void);
  void initCtrls(void);
  void updateCtrls(void);



private:
  //ROS relavant members
  ros::NodeHandle nh_, nh_p_;
  gcop_ros_est::InsekfConfig config_;
  dynamic_reconfigure::Server<gcop_ros_est::InsekfConfig> dyn_server_;
  bool dyn_write_;      //save config_ to config
  ros::Timer timer_general_, timer_send_tf_;
  ros::Timer timer_fake_gps_;

  ros::Publisher pub_viz_cov_;
  visualization_msgs::Marker marker_cov_gps_lcl_, marker_cov_base_link_;
  string strfrm_map_, strfrm_robot_, strfrm_gps_lcl_;

  ros::Subscriber sub_gps_;
  string    strtop_gps_;
  ros::Subscriber sub_imu_   , sub_mag_;
  string    strtop_imu_, strtop_mag_;
  ros::Subscriber sub_gyr_v3s_   , sub_acc_v3s_   , sub_mag_v3s_;
  string    strtop_gyr_v3s_, strtop_acc_v3s_, strtop_mag_v3s_;

  //Rotation matrix to put sensor in frame of reference of robot
  Quaternion<double> q_r2gyr_, q_r2acc_, q_r2imu_;

  //ins state evolution and sensor messages
  InsState x_,x_temp_;
  Vector6d u_;
  Ins ins_;
  double t_;
  ros::Time t_epoch_start_;
  FilterReadiness fr_;
  int insstate_initialized_;//0:uninitialized, 1:gps available 2:gyro bias estimated
  int cov_sel_;
  Vector3d mag_, acc_, gyr_;
  Vector3d map0_;//map reference in lat(deg) lon(deg) and alt(m)
  double scale2si_gyr_, scale2si_acc_;
  Transform<double,3, Affine> magcal_trfm_, acccal_trfm_;

  //Kalman filter
  InsKalmanPredictor kp_ins_;
  InsImuKalmanCorrector kc_insimu_;
  InsGpsKalmanCorrector kc_insgps_;
  InsImuMagKalmanCorrector kc_insimumag_;
  InsMagKalmanCorrector    kc_insmag_;

  //Sensors
  InsImu<3> sens_imu_;
  InsImu<6> sens_imu_mag_;
  InsGps<>    sens_gps_;
  InsMag<>    sens_mag_;
  //Vector3d  ctrl_w_true_, ctrl_a_true_, ctrl_w_drft_, ctrl_a_drft_;

public:
  ros::Rate loop_rate_;
};

CallBackInsEkf::CallBackInsEkf():
    x_(),
    ins_(),
    fr_(ins_,x_),
    nh_p_("~"),
    loop_rate_(500),
    t_(0),
    kp_ins_(ins_),
    kc_insimu_(ins_.X, sens_imu_),
    kc_insgps_(ins_.X, sens_gps_),
    kc_insimumag_(ins_.X, sens_imu_mag_),
    kc_insmag_(ins_.X, sens_mag_)

{
  cout<<"*Entering constructor of cbc"<<endl;
  //setup dynamic reconfigure
  dynamic_reconfigure::Server<gcop_ros_est::InsekfConfig>::CallbackType dyn_cb_f;
  dyn_cb_f = boost::bind(&CallBackInsEkf::cbReconfig, this, _1, _2);
  dyn_server_.setCallback(dyn_cb_f);

  // Setup topic names
  setupTopicsAndNames();

  //Setup publishers and Timers
  initSubsPubsAndTimers();

  //Setup rviz markers
  initRvizMarkers();

  //Setup ins state and evolution
  initSensParams();

  //Fake a GPS message
  // Say that gps is reading origin
  timer_fake_gps_ = nh_.createTimer(ros::Duration(0.1), &CallBackInsEkf::cbTimerFakeGPS, this);
  timer_fake_gps_.start();
}

CallBackInsEkf::~CallBackInsEkf()
{

}

void
CallBackInsEkf::cbTimerFakeGPS(const ros::TimerEvent& event)
{
  static sensor_msgs::NavSatFix::Ptr msg_gps(new sensor_msgs::NavSatFix);
  msg_gps->latitude  = map0_(0);
  msg_gps->longitude = map0_(1);
  msg_gps->altitude  = map0_(2);
  msg_gps->header.frame_id="/multisense";
  msg_gps->header.stamp = ros::Time::now();
  nh_p_.getParam("cov_gps_xy",msg_gps->position_covariance[0]);
  nh_p_.getParam("cov_gps_xy",msg_gps->position_covariance[4]);
  nh_p_.getParam("cov_gps_z" ,msg_gps->position_covariance[8]);
  cbSubGps((sensor_msgs::NavSatFix::ConstPtr)msg_gps);
}

void
CallBackInsEkf::cbTimerGeneral(const ros::TimerEvent& event)
{

}


void
CallBackInsEkf::initRvizMarkers(void)
{
  if(config_.dyn_debug_on)
  {
    cout<<"*Initializing all rviz markers."<<endl;
  }
  int id=-1;
  //Marker for path
  id++;
  //Marker for displaying gps covariance
  nh_p_.getParam("strfrm_gps_lcl", marker_cov_gps_lcl_.header.frame_id);
  marker_cov_gps_lcl_.ns = "insekf";
  marker_cov_gps_lcl_.id = id;
  marker_cov_gps_lcl_.type = visualization_msgs::Marker::SPHERE;
  marker_cov_gps_lcl_.action = visualization_msgs::Marker::ADD;
  marker_cov_gps_lcl_.pose.position.x = 0;
  marker_cov_gps_lcl_.pose.position.y = 0;
  marker_cov_gps_lcl_.pose.position.z = 0;
  marker_cov_gps_lcl_.pose.orientation.x = 0.0;
  marker_cov_gps_lcl_.pose.orientation.y = 0.0;
  marker_cov_gps_lcl_.pose.orientation.z = 0.0;
  marker_cov_gps_lcl_.pose.orientation.w = 1.0;
  marker_cov_gps_lcl_.color.r = 0.0;
  marker_cov_gps_lcl_.color.g = 1.0;
  marker_cov_gps_lcl_.color.b = 0.0;

  //Marker for displaying covariance of estimate
  id++;
  nh_p_.getParam("strfrm_robot", marker_cov_base_link_.header.frame_id);
  marker_cov_base_link_.ns = "insekf";
  marker_cov_base_link_.id = id;
  marker_cov_base_link_.type = visualization_msgs::Marker::SPHERE;
  marker_cov_base_link_.action = visualization_msgs::Marker::ADD;
  marker_cov_base_link_.pose.position.x = 0;
  marker_cov_base_link_.pose.position.y = 0;
  marker_cov_base_link_.pose.position.z = 0;
  marker_cov_base_link_.pose.orientation.x = 0;
  marker_cov_base_link_.pose.orientation.y = 0;
  marker_cov_base_link_.pose.orientation.z = 0;
  marker_cov_base_link_.pose.orientation.w = 1.0;
  marker_cov_base_link_.color.r = 1.0;
  marker_cov_base_link_.color.g = 0.0;
  marker_cov_base_link_.color.b = 0.0;
}
void
CallBackInsEkf::sendMarkersAndTF(Vector3d xyz_gps)
{
  static tf::TransformBroadcaster br;
  if(config_.dyn_enable_tf_publisher)
  {  // Send GPS data for visualization
    tf::Transform trfm2;
    trfm2.setOrigin( tf::Vector3(xyz_gps(0),xyz_gps(1),xyz_gps(2)) );
    tf::Quaternion q2;
    q2.setRPY(0, 0, 0);
    trfm2.setRotation(q2);

    br.sendTransform(tf::StampedTransform(trfm2, ros::Time::now(), strfrm_map_,strfrm_gps_lcl_));
  }

  if(config_.dyn_enable_cov_disp_gps)
  {
    marker_cov_gps_lcl_.header.stamp = ros::Time();
    marker_cov_gps_lcl_.scale.x = sens_gps_.R.diagonal()(0);
    marker_cov_gps_lcl_.scale.y = sens_gps_.R.diagonal()(1);
    marker_cov_gps_lcl_.scale.z = 0.1;
    marker_cov_gps_lcl_.color.a = config_.dyn_alpha_cov; // Don't forget to set the alpha!
    pub_viz_cov_.publish( marker_cov_gps_lcl_ );
  }

  //    Vector3d rpy;
  //    tf::Quaternion q_hack;
  //    SO3::Instance().g2q(rpy,x_.R);
  //    q_hack.setRPY(rpy[0],rpy[1],0);

  if(config_.dyn_enable_cov_disp_est)
  {
    marker_cov_base_link_.header.stamp = ros::Time();
    marker_cov_base_link_.scale.x = x_.P(9,9);//hack. TODO: set to corresponding eigen values of P
    marker_cov_base_link_.scale.y = x_.P(10,10);//hack. TODO: set to corresponding eigen values of P
    marker_cov_base_link_.scale.z = x_.P(11,11);//hack. TODO: set to corresponding eigen values of P
    marker_cov_base_link_.color.a = config_.dyn_alpha_cov; // Don't forget to set the alpha!
    pub_viz_cov_.publish( marker_cov_base_link_ );
  }
}
void
CallBackInsEkf::cbSubGps(const sensor_msgs::NavSatFix::ConstPtr& msg_gps)
{
  static bool first_call=true;
  //Set covariance of gps measurement
  if(cov_sel_==0)
  {
    Matrix3d cov_gps;cov_gps.diagonal()<<config_.dyn_cov_gps_xy,config_.dyn_cov_gps_xy,config_.dyn_cov_gps_z;
    Map<Matrix3d>((double*)msg_gps->position_covariance.data()) = cov_gps;
  }
  else if(cov_sel_!=1)
  {
    cout<<"cov_sel set to wrong value in param file"<<endl;
    assert(0);
  }

  //Update the noise covariance in gcop::sensor(and derived) object
  sens_gps_.R = Map<Matrix3d>((double*)msg_gps->position_covariance.data());

  //Get local coordinates(with map0_ being origin and X-Y-Z axis being East-North-Up
  double x_m_lcl, y_m_lcl, yaw_m_lcl; //local in reference to the LL point on the JHU map
  gis_common::gpsToLocal(x_m_lcl,y_m_lcl,msg_gps->latitude,msg_gps->longitude,map0_(0),map0_(1));
  Vector3d xyz_gps;xyz_gps << x_m_lcl,y_m_lcl, 0;

  if(!fr_.is_ready_p_)
    fr_.trySetpAndCov(xyz_gps,sens_gps_.R);

  if(fr_.isReady())//perform a sensor update
  {
    InsState xs;
    Vector3d &zp=xyz_gps;// noisy measurements of position
    double t= (msg_gps->header.stamp - t_epoch_start_).toSec();
    kc_insgps_.Correct(x_temp_, t, x_, u_, zp);
    x_ = x_temp_;
  }

  sendMarkersAndTF(xyz_gps);

   if(first_call)
   {
     fr_.is_ready_gps_ =true;
     first_call = false;
   }
}

void
CallBackInsEkf::cbSubImu(const sensor_msgs::Imu::ConstPtr& msg_imu)
{
  static bool first_call=true;

  static bool first_time = true;
  static ros::Time t_epoch_prev;

  //Set covariance of imu measurement
  if(cov_sel_==0)
  {
    Matrix3d cov_gyr;cov_gyr.diagonal().setConstant(config_.dyn_cov_gyr);
    Matrix3d cov_acc;cov_acc.diagonal().setConstant(config_.dyn_cov_acc);
    Map<Matrix3d>((double*)msg_imu->angular_velocity_covariance.data()) = cov_gyr;
    Map<Matrix3d>((double*)msg_imu->linear_acceleration_covariance.data()) = cov_acc;
  }
  else if(cov_sel_!=1)
  {
    cout<<"cov_sel set to wrong value in param file"<<endl;
    assert(0);
  }

  //Update the noise covariance in gcop::sensor(and derived) object
  sens_imu_.R.topLeftCorner<3,3>() = Map<Matrix3d>((double*)msg_imu->linear_acceleration_covariance.data());
  sens_imu_mag_.R.topLeftCorner<3,3>() = Map<Matrix3d>((double*)msg_imu->linear_acceleration_covariance.data());
  sens_imu_mag_.R.bottomRightCorner<3,3>().diagonal().setConstant(sens_imu_mag_.srm*sens_imu_mag_.srm);

  //Initialize the x_.R if both mag and acc readings are available

  if(fr_.canInitR()&& !fr_.is_ready_R_)
    fr_.trySetRAndCov();

  //Kalman filter prediction setp
  if(fr_.isReady())
  {
    if(first_time)
    {
      t_epoch_start_= msg_imu->header.stamp;
      t_epoch_prev = t_epoch_start_;
      first_time = false;
      cout << "*****t=0*****\n";
      cout << "initial state is as follows:"<<endl;
      cout << "R:\n"<< x_.R <<endl;
      cout << "bg:\n"<< x_.bg.transpose()<<endl;
      cout << "ba:\n"<< x_.ba.transpose()<<endl;
      cout << "p:\n"<< x_.p.transpose()<<endl;
      cout << "v:\n"<< x_.v.transpose()<<endl;
      cout << "P:\n"<< x_.P <<endl;
    }
    else
    {
      Vector3d a,w,m;
      Vector6d u;
      a << msg_imu->linear_acceleration.x, msg_imu->linear_acceleration.y, msg_imu->linear_acceleration.z;
      w << msg_imu->angular_velocity.x,    msg_imu->angular_velocity.y,    msg_imu->angular_velocity.z;
      u << w, a;
      u_ = u;

      Vector6d zam; zam<< a,mag_;
      InsState xp;
      double t  = (msg_imu->header.stamp - t_epoch_start_).toSec();
      double dt = (msg_imu->header.stamp -   t_epoch_prev).toSec();


      kp_ins_.Predict(x_temp_, t, x_, u, dt);
//      x_temp_.v.setZero();
//      x_temp_.p.setZero();
//      x_ = x_temp_;
      kc_insimu_.Correct(x_, t, x_temp_, u, a);
      //kc_insimumag_.Correct(x_, t, x_temp_, u, zam);
      t_epoch_prev = msg_imu->header.stamp;

      //Display suff
      Vector6d u_bias; u_bias<<x_.bg, (x_.ba+x_.R.transpose()*sens_imu_.a0);
      cout << "****************\n";
      cout << "t:"<<t<< "\tdt:"<<dt <<endl;
      //cout << ""
      cout << "u(w,a):"<<u.transpose()<< endl;
      cout << "(bg,ba):"<<x_.bg.transpose()<<x_.ba.transpose()<< endl;
      cout << "u(w,a)-(bw,ba) - R'*g:"<<u.transpose()- u_bias.transpose()<< endl;
      cout << "linear vel: " << x_.v.transpose() << endl;
      cout << "position: " << x_.p.transpose() << endl;
      cout << "Estim attitude:\n" << x_.R << endl;
    }
  }
  else
    cout<<"filter is not ready"<<endl;

  if(first_call)
  {
    //     fr_.is_ready_acc_=true;
    //     fr_.is_ready_gyr_=true;
    first_call = false;
  }
}
void
CallBackInsEkf::cbSubMag(const sensor_msgs::MagneticField::ConstPtr& msg_mag)
{
  static bool first_call=true;

  Vector3d mag_raw;
  mag_raw << msg_mag->magnetic_field.x, msg_mag->magnetic_field.y, msg_mag->magnetic_field.z;
  mag_= magcal_trfm_*mag_raw; mag_.normalize();

  //Sensor update
  double t  = (msg_mag->header.stamp - t_epoch_start_).toSec();
  if(fr_.isReady())
  {
    kc_insmag_.Correct(x_temp_, t, x_, u_, mag_);
    x_ = x_temp_;
  }

  if(first_call)
  {
    first_call = false;
    fr_.is_ready_gps_=true;
  }
}

void
CallBackInsEkf::cbSubAccV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg_acc_v3s)
{
  static bool first_call=true;

  Vector3d acc_raw;
  acc_raw << msg_acc_v3s->vector.x,msg_acc_v3s->vector.y,msg_acc_v3s->vector.z;
  acc_ = scale2si_acc_* (q_r2acc_* (acccal_trfm_* acc_raw));
  if(first_call)
  {
    first_call = false;
    fr_.is_ready_acc_= true;
  }
}
void
CallBackInsEkf::cbSubMagV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg_mag_v3s)
{
  static bool first_call=true;
  Vector3d mag_raw;
  mag_raw << msg_mag_v3s->vector.x, msg_mag_v3s->vector.y, msg_mag_v3s->vector.z;
  mag_= magcal_trfm_*mag_raw;mag_.normalize();
  double t  = (msg_mag_v3s->header.stamp - t_epoch_start_).toSec();
  if(fr_.isReady())
  {
    kc_insmag_.Correct(x_temp_, t, x_, u_, mag_);
    x_ = x_temp_;
  }
  if(first_call)
  {
    first_call = false;
    fr_.is_ready_mag_=true;
  }
}

void
CallBackInsEkf::cbSubGyrV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg_gyr_v3s)
{
  static bool first_call=true;
  Vector3d gyr_raw;
  gyr_raw << msg_gyr_v3s->vector.x, msg_gyr_v3s->vector.y, msg_gyr_v3s->vector.z;
  gyr_= scale2si_gyr_ * (q_r2gyr_ * gyr_raw);

  sensor_msgs::Imu::Ptr msg_imu(new sensor_msgs::Imu);
  msg_imu->header = msg_gyr_v3s->header;
  msg_imu->angular_velocity.x = gyr_(0);
  msg_imu->angular_velocity.y = gyr_(1);
  msg_imu->angular_velocity.z = gyr_(2);

  msg_imu->linear_acceleration.x = acc_(0);
  msg_imu->linear_acceleration.y = acc_(1);
  msg_imu->linear_acceleration.z = acc_(2);

  //gyro bias and gyrobias covariance estimation
  if(!fr_.is_ready_bg_)
    fr_.trySetGyroBiasAndCov(gyr_,first_call);

  cbSubImu((sensor_msgs::Imu::ConstPtr)msg_imu);

  if(first_call)
  {
    fr_.is_ready_gyr_=true;
    first_call = false;
  }
}

template<typename T>
void
CallBackInsEkf::rosParam2Mat(T &mat, std::string param)
{
  XmlRpc::XmlRpcValue mat_xml;
  nh_p_.getParam(param,mat_xml);
  xml2Mat(mat, mat_xml);
}

void
CallBackInsEkf::setupTopicsAndNames(void)
{
  if(config_.dyn_debug_on)
    cout<<"setting up topic names"<<endl;

  nh_p_.getParam("strtop_gps",strtop_gps_);

  nh_p_.getParam("strtop_imu",strtop_imu_);
  nh_p_.getParam("strtop_mag",strtop_mag_);

  nh_p_.getParam("strtop_gyr_v3s",strtop_gyr_v3s_);
  nh_p_.getParam("strtop_acc_v3s",strtop_acc_v3s_);
  nh_p_.getParam("strtop_mag_v3s",strtop_mag_v3s_);

  nh_p_.getParam("strfrm_map",strfrm_map_);
  nh_p_.getParam("strfrm_robot",strfrm_robot_);
  nh_p_.getParam("strfrm_gps_lcl",strfrm_gps_lcl_);

  if(config_.dyn_debug_on)
  {
    cout<<"Topics are:  "<<endl;

    cout<<"strtop_gps:  "<<strtop_gps_<<endl;

    cout<<"strtop_imu:  "<<strtop_imu_<<endl;
    cout<<"strtop_mag:  "<<strtop_mag_<<endl;

    cout<<"strtop_gyr_v3s:  "<<strtop_gyr_v3s_<<endl;
    cout<<"strtop_acc_v3s:  "<<strtop_acc_v3s_<<endl;
    cout<<"strtop_mag_v3s:  "<<strtop_mag_v3s_<<endl;
  }


}


void
CallBackInsEkf::initSubsPubsAndTimers(void)
{
  //Publishers
  pub_viz_cov_ = nh_.advertise<visualization_msgs::Marker>( "visualization_marker", 0 );

  //Subscribers
  sub_gps_  = nh_.subscribe<sensor_msgs::NavSatFix>(strtop_gps_,1000,&CallBackInsEkf::cbSubGps, this);

  sub_imu_ = nh_.subscribe<sensor_msgs::Imu>(strtop_imu_,1000,&CallBackInsEkf::cbSubImu, this);
  sub_mag_ = nh_.subscribe<sensor_msgs::MagneticField>(strtop_mag_,1000,&CallBackInsEkf::cbSubMag, this);

  sub_acc_v3s_  = nh_.subscribe<geometry_msgs::Vector3Stamped>(strtop_acc_v3s_,1000,&CallBackInsEkf::cbSubAccV3S, this);
  sub_mag_v3s_  = nh_.subscribe<geometry_msgs::Vector3Stamped>(strtop_mag_v3s_,1000,&CallBackInsEkf::cbSubMagV3S, this);
  sub_gyr_v3s_  = nh_.subscribe<geometry_msgs::Vector3Stamped>(strtop_gyr_v3s_,1000,&CallBackInsEkf::cbSubGyrV3S, this);

  //Timers
  timer_general_ = nh_.createTimer(ros::Duration(0.05), &CallBackInsEkf::cbTimerGeneral, this);
  timer_send_tf_   =  nh_.createTimer(ros::Duration(config_.dyn_period_tf_publish), &CallBackInsEkf::cbTimerPublishTF, this);
  timer_general_.start();
  timer_send_tf_.start();

}

void
CallBackInsEkf::cbTimerPublishTF(const ros::TimerEvent& event)
{
  cout<<"publishing tf:"<<endl;
  //fused visualization
  tf::Transform trfm;
  trfm.setOrigin( tf::Vector3(x_.p[0],x_.p[1], x_.p[2]) );
  //trfm.setOrigin( tf::Vector3(0,0,0));
  Vector4d wxyz;
  SO3::Instance().g2quat(wxyz, x_.R);
  tf::Quaternion q_true(wxyz[1],wxyz[2],wxyz[3],wxyz[0]);
  trfm.setRotation(q_true);
//  //TODO: yaw has to be set to something or use mag
//  double yaw=0;
//  Vector3d rpy;
//  tf::Quaternion q_hack;
//
//  if(config_.dyn_enable_true_yaw)
//  {
//    trfm.setRotation(q_true);
//  }
//  else
//  {
//    SO3::Instance().g2q(rpy,x_.R);
//    q_hack.setRPY(rpy[0],rpy[1],yaw);
//    trfm.setRotation(q_hack);
//  }

  static tf::TransformBroadcaster br;
  if(config_.dyn_enable_tf_publisher)
    br.sendTransform(tf::StampedTransform(trfm, ros::Time::now(), strfrm_map_, strfrm_robot_));
}

void
CallBackInsEkf::initSensParams(void)
{
  ins_.sv = 1e-6;
  //covariance selection
  //0: use from paramfile/reconfigure
  //1: use from the sensor messages if available
  nh_p_.getParam("cov_sel",cov_sel_);

  //Set reference
  //a0: accelerometer, m0:magnetometer, map0_:gps(lat0(deg), lon0(deg), alt0(m)
  Vector3d a0,m0;
  rosParam2Mat(a0,"a0");
  rosParam2Mat(m0,"m0");
  ins_.g0         = a0;
  sens_imu_mag_.a0 = a0;
  sens_imu_mag_.m0 = m0;
  sens_imu_.a0 = a0;
  sens_imu_.m0 = m0.normalized();
  sens_mag_.m0 = m0.normalized();
  rosParam2Mat(map0_,"map0");

  //set robot to sensors rotation
  Vector7d tfm_r2gyr, tfm_r2mag, tfm_r2acc;
  rosParam2Mat(tfm_r2gyr, "robot2gyr");
  rosParam2Mat(tfm_r2acc, "robot2acc");
  q_r2gyr_ = Quaternion<double>(tfm_r2gyr(6),tfm_r2gyr(3),tfm_r2gyr(4),tfm_r2gyr(5));
  q_r2acc_ = Quaternion<double>(tfm_r2acc(6),tfm_r2acc(3),tfm_r2acc(4),tfm_r2acc(5));

  //set scale2si for mag, gyr, acc
  nh_p_.getParam("scale2si_gyr",scale2si_gyr_);
  nh_p_.getParam("scale2si_acc",scale2si_acc_);

  //Edit sens noise params
  updateNoiseFromConfig();

  //MagCalib setup
  initMagCalib();
}

void
CallBackInsEkf::initMagCalib(void)
{
  Matrix3d magcal_linear;
  Vector3d magcal_translation;
  rosParam2Mat(magcal_linear , "magcal_linear");
  rosParam2Mat(magcal_translation, "magcal_translation");
  magcal_trfm_.linear()      = magcal_linear;
  magcal_trfm_.translation() = magcal_translation;

  Matrix3d acccal_linear;
  Vector3d acccal_translation;
  rosParam2Mat(acccal_linear , "acccal_linear");
  rosParam2Mat(acccal_translation, "acccal_translation");
  acccal_trfm_.linear()      = acccal_linear;
  acccal_trfm_.translation() = acccal_translation;
}

void
CallBackInsEkf::updateNoiseFromConfig(void)
{
  if(config_.dyn_debug_on)
    cout<<"Updating the noise params"<<endl;

  //setNoise
  //gps noise
  sens_gps_.sxy = sqrt(config_.dyn_cov_gps_xy);
  sens_gps_.sz = sqrt(config_.dyn_cov_gps_z);

  //accel noise
  sens_imu_.sra = sqrt(config_.dyn_cov_acc);
  ins_.sra      = sens_imu_.sra;

  //mag noise
  sens_imu_.srm = sqrt(config_.dyn_cov_mag);
  sens_mag_.srm = sqrt(config_.dyn_cov_mag);
  sens_imu_mag_.srm = sqrt(config_.dyn_cov_mag);

  //gyr noise is a control noise and is set after finding the gyro bias
  //ins_.sv

}


void
CallBackInsEkf::setFromParamsConfig()
{
  cout<<"Setting Dynamic Reconfigure params from parameter server params"<<endl;

  // Noise parameters
  nh_p_.getParam("cov_mag",config_.dyn_cov_mag);
  nh_p_.getParam("cov_acc",config_.dyn_cov_acc);
  nh_p_.getParam("cov_gyr",config_.dyn_cov_gyr);
  nh_p_.getParam("cov_gps_xy", config_.dyn_cov_gps_xy);
  nh_p_.getParam("cov_gps_z", config_.dyn_cov_gps_z);

}

void
CallBackInsEkf::cbReconfig(gcop_ros_est::InsekfConfig &config, uint32_t level)
{
  cout<<"* Entering reconfig with level:"<<level<<endl;
  config_ = config;
  if(level==numeric_limits<uint32_t>::max())
    setFromParamsConfig();
  else
  {

  }
  config = config_;
}

//------------------------------------------------------------------------
//--------------------------------MAIN -----------------------------------
//------------------------------------------------------------------------

int main(int argc, char** argv)
{
  ros::init(argc,argv,"ekf_imu_mag_gps",ros::init_options::NoSigintHandler);
  signal(SIGINT,mySigIntHandler);

  CallBackInsEkf cbc;

  while(!g_shutdown_requested)
  {
    ros::spinOnce();
    cbc.loop_rate_.sleep();
  }
  return 0;

}









