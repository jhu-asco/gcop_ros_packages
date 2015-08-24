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
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <sensor_msgs/MagneticField.h>
#include <gcop_ros_est/InsekfDiag.h>


//gcop_comm msgs
#include <gcop_comm/State.h>
#include <gcop_comm/CtrlTraj.h>
#include <gcop_comm/Trajectory_req.h>

// gps, utm, local coord conversions
#include "llh_enu_cov.h"
#include <enu/enu.h>  // ROS wrapper for conversion functions

// utils
#include <eigen_ros_conv.h>

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

//Eigen Lib Includes
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>


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

template<typename T>
string xml2StringMat(T &mat, XmlRpc::XmlRpcValue &my_list)
{
  assert(my_list.getType() == XmlRpc::XmlRpcValue::TypeArray);
  assert(my_list.size() > 0);
  if(my_list.size()==1)
    return static_cast<string>(my_list[0]);
  else if(mat.size()!=my_list.size()-1)
    return "invalid";
  else
  {
    for (int i = 0; i < mat.rows(); i++)
    {
      for(int j=0; j<mat.cols();j++)
      {
        int k = j+ i*mat.cols();
        assert(my_list[k+1].getType() == XmlRpc::XmlRpcValue::TypeDouble);
        mat(i,j) =  (double)(my_list[k+1]);
      }
    }
    return static_cast<string>(my_list[0]);
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

template<typename T>
void
rosParam2Mat(T &mat, const std::string param)
{
  static ros::NodeHandle nh_p("~");
  XmlRpc::XmlRpcValue mat_xml;
  nh_p.getParam(param,mat_xml);
  xml2Mat(mat, mat_xml);
}

template<typename T>
string
rosParam2StringMat(T &mat, const std::string param)
{
  static ros::NodeHandle nh_p("~");
  XmlRpc::XmlRpcValue mat_xml;
  nh_p.getParam(param,mat_xml);
  return xml2StringMat( mat, mat_xml);
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

//------------------------------------------------------------------------
//------------------------MAIN CALLBACK CLASS ----------------------------
//------------------------------------------------------------------------


class CallBackInsEkf
{
public:
  typedef Matrix<double, 4, 4> Matrix4d;
  typedef Matrix<double, 4, 1> Vector4d;
  typedef Matrix<double, 7, 1> Vector7d;

  class RetReady
  {
  public:
    virtual
    bool ready(void) const=0;
  };

class SelInitState: public RetReady
{
private:
    InsState x0_;
    bool is_ready_R_and_cov_;
    bool is_ready_bg_and_cov_;
    bool is_ready_ba_and_cov_;
    bool is_ready_p_and_cov_;
    bool is_ready_v_and_cov_;

    int type_R_;//-1: uninitialized 0:yaml 1:est
    int type_R_cov_;
    int type_bg_;
    int type_bg_cov_;
    int type_ba_;
    int type_ba_cov_;
    int type_p_;
    int type_p_cov_;
    int type_v_;
    int type_v_cov_;

    vector<Vector3d> gyro_buffer_;
    vector<Vector3d> acc_buffer_;
    vector<Vector3d> rpy_buffer_;
public:
    int n_avg_; // The number of readings to average for any computation
public:
    SelInitState():
      is_ready_R_and_cov_(false),
      is_ready_bg_and_cov_(false),
      is_ready_ba_and_cov_(false),
      is_ready_p_and_cov_(false),
      is_ready_v_and_cov_(false),
      type_R_(-1),
      type_R_cov_(-1),
      type_bg_(-1),
      type_bg_cov_(-1),
      type_ba_(-1),
      type_ba_cov_(-1),
      type_p_(-1),
      type_p_cov_(-1),
      type_v_(-1),
      type_v_cov_(-1),
      n_avg_(200)
    {

    }
    const InsState& state(void) const
    {
      return x0_;
    }
    //delete this function later
    void setAllType(int val)
    {
//      type_R_     =val;
//      type_R_cov_ =val;
//      type_bg_    =val;
//      type_bg_cov_=val;
      type_ba_    =val;
      type_ba_cov_=val;
      type_p_     =val;
      type_p_cov_ =val;
      type_v_     =val;
      type_v_cov_ =val;
    }

    virtual
    bool ready(void) const
    {
      return is_ready_R_and_cov_&&
             is_ready_bg_and_cov_&&
             is_ready_ba_and_cov_&&
             is_ready_p_and_cov_&&
             is_ready_v_and_cov_;
    }

    bool readyRAndCov(void) const
    {
      return is_ready_R_and_cov_;
    }

    bool readyBgAndCov(void) const
    {
      return is_ready_bg_and_cov_;
    }

    bool readyBaAndCov(void) const
    {
      return is_ready_ba_and_cov_;
    }

    bool readypAndCov(void) const
    {
      return is_ready_p_and_cov_;
    }

    bool readyvAndCov(void) const
    {
      return is_ready_v_and_cov_;
    }

    void tryComputeRAndCov(const Vector3d& acc,const Vector3d& acc0,
                           const Vector3d& mag,const Vector3d& mag0, bool reset )
    {
      if(type_R_ < 0 || type_R_cov_ < 0)
      {
        cout<<"User tried to set initial R and cov without initializing it first"<<endl;
        assert(0);
      }

      if(!is_ready_R_and_cov_)
      {

        if(reset)
        {
          rpy_buffer_.reserve(n_avg_);
          is_ready_R_and_cov_ = false;
        }
        else if(rpy_buffer_.size() < n_avg_)
        {
          Vector3d an; an = acc.normalized();
          Vector3d mn; mn = mag.normalized();
          Vector3d m0; m0 = mag0.normalized();

          double p = -asin(an(0));
          double cp = cos(p);
          double sr =  an(1)/cos(p);
          double cr =  an(2)/cos(p);
          double r = atan2(sr,cr);

          Matrix3d rot_yp_xr;
          rot_yp_xr = AngleAxisd(p,Vector3d::UnitY())* AngleAxisd(r, Vector3d::UnitX());

          Vector3d mt = rot_yp_xr*mn;

          double cy = ( m0(0)*mt(0) + m0(1)*mt(1))/(m0(0)*m0(0)+m0(1)*m0(1));
          double sy = ( m0(1)*mt(0) - m0(0)*mt(1))/(m0(0)*m0(0)+m0(1)*m0(1));

          double y = atan2(sy,cy);
          Vector3d rpy; rpy<<r,p,y;
          rpy_buffer_.push_back(rpy);
        }
        else
        {
          Matrix3d cov;
          Vector3d rpy_avg;
          avgVectInStl(rpy_avg,cov,rpy_buffer_);
          gyro_buffer_.clear();
          Matrix3d rot_zy_yp_xr;
          rot_zy_yp_xr =  AngleAxisd(rpy_avg(2),Vector3d::UnitZ())
                         *AngleAxisd(rpy_avg(1),Vector3d::UnitY())
                         *AngleAxisd(rpy_avg(0),Vector3d::UnitX());

          setRAndCov(rot_zy_yp_xr,0.01*Matrix3d::Identity());
        }
      }
      else
        cout<<"R and its cov already set"<<endl;
    }
    void setRAndCov(const Matrix3d& R, const Matrix3d& cov)
    {
      if(type_R_ < 0 || type_R_cov_< 0)
      {
        cout<<"User tried to set initial R and cov without initializing it first"<<endl;
        assert(0);
      }
      if(!is_ready_R_and_cov_)
      {
        x0_.R = R;
        x0_.P.topLeftCorner<3,3>() = cov;
        is_ready_R_and_cov_=true;
      }
      else
        cout<<"R and its cov already set"<<endl;
    }

    void initRAndCov(const string& type_rot, const Matrix3d& rot, const string& type_vars, const Vector3d& vars)
    {
      if(type_rot.compare(type_vars))
      {
        cout<<"Selection method for rotation matrix and it's covariance is not the same"<<endl;
        assert(0);
      }
      if(!type_rot.compare("yaml"))
      {
        type_R_=0;
        type_R_cov_=0;
        Matrix3d cov; cov.setZero();
        cov.diagonal() = vars;
        setRAndCov(rot,cov);
      }
      else if(!type_rot.compare("est"))
      {
        type_R_=1;
        type_R_cov_=1;
      }
      else
        assert(0);
    }

    void tryComputeBgAndCov(const Vector3d& gyr, const bool reset)
    {
      if(type_bg_ < 0 || type_bg_cov_< 0)
      {
        cout<<"User tried to set initial bg and cov without initializing it first"<<endl;
        assert(0);
      }
      if(!is_ready_bg_and_cov_)
      {
        Vector3d bg;

        if(reset)
        {
          bg.setZero();
          gyro_buffer_.reserve(n_avg_);
          is_ready_bg_and_cov_ = false;
        }
        else if(gyro_buffer_.size() < n_avg_)
          gyro_buffer_.push_back(gyr);
        else
        {
          Matrix3d cov;
          avgVectInStl(bg,cov,gyro_buffer_);
          gyro_buffer_.clear();
          setBgAndCov(bg,cov);
        }
      }
      else
        cout<<"bg and its cov already set"<<endl;
    }

    void setBgAndCov(const Vector3d& bg, const Matrix3d& cov)
    {
      if(type_bg_< 0 || type_bg_cov_< 0)
      {
        cout<<"User tried to set initial  bg and cov without initializing it first"<<endl;
        assert(0);
      }
      if(!is_ready_bg_and_cov_)
      {
      x0_.bg = bg;
      x0_.P.block<3,3>(3,3) = cov;
      is_ready_bg_and_cov_=true;
      }
      else
        cout<<"bg and its cov already set"<<endl;
    }

    void initBgAndCov(const string& type_val,const Vector3d& val,const string& type_vars, const Vector3d& vars)
    {
      if(type_val.compare(type_vars))
      {
        cout<<"Selection method for Bg and it's covariance is not the same"<<endl;
        assert(0);
      }
      if(!type_val.compare("yaml"))
      {
        type_bg_=0;
        type_bg_cov_=0;
        Matrix3d cov; cov.setZero();
        cov.diagonal() = vars;
        setBgAndCov(val,cov);
      }
      else if(!type_val.compare("est"))
      {
        type_bg_=1;
        type_bg_cov_=1;
      }
      else
        assert(0);
    }

    void tryComputeBaAndCov(const Vector3d& acc, const bool reset)
    {
      if(type_ba_ < 0 || type_ba_cov_< 0)
      {
        cout<<"User tried to set initial ba and cov without initializing it first"<<endl;
        assert(0);
      }
      if(!is_ready_ba_and_cov_)
      {
        Vector3d ba;

        if(reset)
        {
          ba.setZero();
          acc_buffer_.reserve(n_avg_);
          is_ready_ba_and_cov_ = false;
        }
        else if(acc_buffer_.size() < n_avg_)
          acc_buffer_.push_back(acc);
        else
        {
          Matrix3d cov;
          avgVectInStl(ba,cov,acc_buffer_);
          acc_buffer_.clear();
          setBaAndCov(Vector3d::Zero(),cov);
        }
      }
      else
        cout<<"ba and its cov already set"<<endl;
    }

    void setBaAndCov(const Vector3d& ba, const Matrix3d& cov)
    {
      if(type_ba_ < 0 || type_ba_cov_< 0)
      {
        cout<<"User tried to set initial ba and cov without initializing it first"<<endl;
        assert(0);
      }
      if(!is_ready_ba_and_cov_)
      {
        x0_.ba = ba;
        x0_.P.block<3,3>(6,6) = cov;
        is_ready_ba_and_cov_=true;
      }
      else
        cout<<"Ba and its cov already set"<<endl;
    }

    void initBaAndCov(const string& type_val,const Vector3d& val,const string& type_vars, const Vector3d& vars)
    {
      if(type_val.compare(type_vars))
      {
        cout<<"Selection method for Ba and it's covariance is not the same"<<endl;
        assert(0);
      }
      if(!type_val.compare("yaml"))
      {
        type_ba_=0;
        type_ba_cov_=0;
        Matrix3d cov; cov.setZero();
        cov.diagonal() = vars;
        setBgAndCov(val,cov);
      }
      else if(!type_val.compare("est"))
      {
        type_ba_=1;
        type_ba_cov_=1;
      }
      else
        assert(0);
    }
    void setpAndCov(const Vector3d& p,const Matrix3d& cov)
    {
      if(type_p_ < 0 || type_p_cov_< 0)
      {
        cout<<"User tried to set initial p and cov without initializing it first"<<endl;
        assert(0);
      }
      if(!is_ready_p_and_cov_)
      {

      x0_.p = p;
      x0_.P.block<3,3>(9,9) = cov;
      is_ready_p_and_cov_=true;
      }
      else
        cout<<"p and its cov already set"<<endl;
    }

    void setvAndCov(const Vector3d& v,const Matrix3d& cov)
    {
      if(type_v_ < 0 || type_v_cov_< 0)
      {
        cout<<"User tried to set initial v and cov without initializing it first"<<endl;
        assert(0);
      }
      if(!is_ready_v_and_cov_)
      {
        x0_.v = v;
        x0_.P.block<3,3>(12,12) = cov;
        is_ready_v_and_cov_=true;
      }
      else
        cout<<"v and its cov already set"<<endl;
    }

};

/**
 * This class provides a way to select covariance from either reconfigure or an estimate.
 * It takes care of making sure that all the places that need the cov value remains updated
 */
class SelCov: public RetReady
{
private:

    Matrix3d cov_;
    double& config_cov_x_;
    double& config_cov_y_;
    double& config_cov_z_;
    vector<double*> vect_sd_[3];
    vector<Matrix3d*> vect_cov_;

    int type_;//-1: uninitialized 0:dyn 1:msg 2:est
    bool ready_;

    vector<Vector3d> val_buffer_;
public:

    int n_avg_;
    int msg_recvd_;

public:

    SelCov(double& cov_x,double& cov_y,double& cov_z)
    :n_avg_(200), type_(-1), msg_recvd_(false), ready_(false),
    config_cov_x_(cov_x),config_cov_y_(cov_y),config_cov_z_(cov_z)
    {

    }

    int type() const
    {
      return type_;
    }

    virtual
    bool ready() const
    {
      return ready_;
    }

    const Matrix3d& cov() const
    {
      return cov_;
    }

    void msgCovReady(bool ready)
    {
      if(type_ == 1)
        ready_ = ready;
    }

    void tryEstCov(Vector3d val, bool reset)
    {
      Vector3d val_mean;
      if(reset)
      {
        val_mean.setZero();
        val_buffer_.reserve(n_avg_);
        if(type_==2)
          ready_ = false;
      }
      else if(val_buffer_.size() < n_avg_)
        val_buffer_.push_back(val);
      else
      {
        Matrix3d cov;
        avgVectInStl(val_mean,cov,val_buffer_);
        if(type_==2)
        {
          ready_ = true;
          cov_ = cov;
          sync();
        }
        val_buffer_.clear();
      }
    }

    void setPointersOfSD(const int id, double* p_sd)
    {
      if(id<0 || id>2)
      {
        cout<<"wrong id used for setPointers"<<endl;
        assert(0);
      }
      vect_sd_[id].push_back(p_sd);
      sync();
    }

    void setPointersOfCov(Matrix3d* p_cov)
    {
      assert(p_cov!=nullptr);
      vect_cov_.push_back(p_cov);
      sync();
    }

    void initCov(string type,Vector3d val)
    {
      if(!type.compare("dyn"))
      {
        type_=0;
        config_cov_x_=val(0);
        config_cov_y_=val(1);
        config_cov_z_=val(2);
        ready_=true;
        cov_.diagonal()<<config_cov_x_, config_cov_y_,config_cov_z_;
      }
      else if(!type.compare("msg"))
        type_=1;
      else if(!type.compare("est"))
        type_=2;
      else
        assert(0);
      sync();
    }

    template <typename M>
    bool updateCov(MatrixBase<M>& cov)
    {
      EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(M, 3, 3);

      if(ready_)
      {
        if(type_==0)//dyn
        {
          cov_.setZero();
          cov_.diagonal()<< config_cov_x_, config_cov_y_,config_cov_z_;
          sync();
          cov = cov_;
        }
        else if(type_==1)//msg
        {
          config_cov_x_=cov.diagonal()(0);
          config_cov_y_=cov.diagonal()(1);
          config_cov_z_=cov.diagonal()(2);
          cov_ = cov;
          sync();
        }
        else if(type_==2)//est
        {
          cov = cov_;
          config_cov_x_=cov.diagonal()(0);
          config_cov_y_=cov.diagonal()(1);
          config_cov_z_=cov.diagonal()(2);
          sync();
        }
      }
      else
      {
        cout<<"tried to update before cov is ready"<<endl;
        assert(0);
      }

      return ready_;
    }

    bool updateCov(void)
    {
      if(ready_)
      {
        if(type_==0)//dyn
        {
          cov_.setZero();
          cov_.diagonal()<< config_cov_x_, config_cov_y_,config_cov_z_;
          sync();
        }
        else if(type_==1)//msg
        {
          cout<<"For updating cov from msg you must enter msg cov"<<endl;
          assert(0);
        }
        else if(type_==2)//est
        {
          config_cov_x_=cov_.diagonal()(0);
          config_cov_y_=cov_.diagonal()(1);
          config_cov_z_=cov_.diagonal()(2);
          sync();
        }
      }
      else
      {
        cout<<"tried to update before cov is ready"<<endl;
        assert(0);
      }

      return ready_;
    }

    void sync(void)
    {
      for(int i=0;i<3;i++)
        for(vector<double*>::iterator it=vect_sd_[i].begin();it!=vect_sd_[i].end();it++)
          **it=sqrt(cov_.diagonal()(i));
      for(vector<Matrix3d*>::iterator it=vect_cov_.begin();it!=vect_cov_.end();it++)
        **it = cov_;

    }
};
/**
 * Class for checking if filter is ready to be used or not
 */
class FilterReadiness
{
public:

    bool is_filtering_on_;

    vector<RetReady*> check_ready_;

public:

    FilterReadiness()
    {
      is_filtering_on_  = false;
    }
    ~FilterReadiness(){}
public:

    bool isReady(void)
    {
      bool ready=true;
      for_each(check_ready_.rbegin(), check_ready_.rend(),[&](RetReady* obj){ready=ready && obj->ready();});
      return  ready;
    }
};

public:
  CallBackInsEkf();
  ~CallBackInsEkf();

private:
  void cbTimerPubTFCov(const ros::TimerEvent& event);
  void cbTimerGeneral(const ros::TimerEvent& event);
  void cbSubGps(const sensor_msgs::NavSatFix::ConstPtr& msg_gps);
  void cbSubImu(const sensor_msgs::Imu::ConstPtr& msg_imu);
  void cbSubMag(const sensor_msgs::MagneticField::ConstPtr& msg_mag);
  void cbSubAccV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg_acc_v3s);
  void cbSubMagV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg_mag_v3s);
  void cbSubGyrV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg_gyr_v3s);

  void cbReconfig(gcop_ros_est::InsekfConfig &config, uint32_t level);
  void setFromParamsConfig(void);

  void initRvizMarkers(void);

  void setupTopicsAndNames(void);
  void initSubsPubsAndTimers(void);

  void loadYamlParams(void);
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

  ros::Publisher pub_odom_, pub_diag_, pub_viz_cov_;
  string strtop_odom_, strtop_diag_, strtop_marker_cov_;

  visualization_msgs::Marker marker_cov_gps_lcl_, marker_cov_base_link_;
  string strfrm_map_, strfrm_robot_, strfrm_gps_lcl_;



  int type_sensor_msg_;
  ros::Subscriber sub_gps_;
  string    strtop_gps_;
  ros::Subscriber sub_imu_   , sub_mag_;
  string    strtop_imu_, strtop_mag_;
  ros::Subscriber sub_gyr_v3s_   , sub_acc_v3s_   , sub_mag_v3s_;
  string    strtop_gyr_v3s_, strtop_acc_v3s_, strtop_mag_v3s_;

  //Rotation matrix to put sensor in frame of reference of robot
  Quaternion<double> q_r2gyr_, q_r2acc_, q_r2imu_;

  //ins state evolution and sensor messages
  SelInitState x0_;
  InsState x_,x_temp_;
  ros::Time t_ep_x_, t_ep_gps_;
  Vector3d xyz_gps_;
  Vector6d u_;
  Ins ins_;
  double t_;
  ros::Time t_epoch_start_;
  FilterReadiness fr_;
  int insstate_initialized_;//0:uninitialized, 1:gps available 2:gyro bias estimated
  SelCov cov_sens_mag_,cov_sens_acc_, cov_sens_gps_;
  SelCov cov_ctrl_gyr_, cov_ctrl_acc_, cov_ctrl_su_, cov_ctrl_sa_;
  Vector3d mag_, acc_, gyr_, acc_raw_, gyr_raw_;
  Vector3d map0_;//map reference in lat(deg) lon(deg) and alt(m)
  double scale2si_gyr_, scale2si_acc_;
  Transform<double,3, Affine> magcal_trfm_, acccal_trfm_;
  bool pause_getchar_;

  //Kalman filter
  InsKalmanPredictor kp_ins_;
  InsImuKalmanCorrector kc_insimu_;
  InsGpsKalmanCorrector kc_insgps_;
  InsMagKalmanCorrector    kc_insmag_;

  //Sensors
  InsImu<3>   sens_acc_;
  InsGps<>    sens_gps_;
  InsMag<>    sens_mag_;
  //Vector3d  ctrl_w_true_, ctrl_a_true_, ctrl_w_drft_, ctrl_a_drft_;

public:
  ros::Rate loop_rate_;
};

CallBackInsEkf::CallBackInsEkf():
    nh_p_("~"),
    loop_rate_(1000),
    t_(0),
    kp_ins_(ins_),
    kc_insimu_(ins_.X, sens_acc_),
    kc_insgps_(ins_.X, sens_gps_),
    kc_insmag_(ins_.X, sens_mag_),
    config_(),
    cov_sens_mag_(config_.dyn_cov_sens_mag,   config_.dyn_cov_sens_mag,   config_.dyn_cov_sens_mag),
    cov_sens_acc_(config_.dyn_cov_sens_acc,   config_.dyn_cov_sens_acc,   config_.dyn_cov_sens_acc),
    cov_sens_gps_(config_.dyn_cov_sens_gps_xy,config_.dyn_cov_sens_gps_xy,config_.dyn_cov_sens_gps_z),
    cov_ctrl_gyr_(config_.dyn_cov_ctrl_gyr,   config_.dyn_cov_ctrl_gyr,   config_.dyn_cov_ctrl_gyr),
    cov_ctrl_acc_(config_.dyn_cov_ctrl_acc,   config_.dyn_cov_ctrl_acc,   config_.dyn_cov_ctrl_acc),
    cov_ctrl_su_(config_.dyn_cov_ctrl_su,   config_.dyn_cov_ctrl_su,   config_.dyn_cov_ctrl_su),
    cov_ctrl_sa_(config_.dyn_cov_ctrl_sa,   config_.dyn_cov_ctrl_sa,   config_.dyn_cov_ctrl_sa)

{
  cout<<"*Entering constructor of cbc"<<endl;
  //setup dynamic reconfigure
  dynamic_reconfigure::Server<gcop_ros_est::InsekfConfig>::CallbackType dyn_cb_f;
  dyn_cb_f = boost::bind(&CallBackInsEkf::cbReconfig, this, _1, _2);
  dyn_server_.setCallback(dyn_cb_f);

  // Setup topic names
  setupTopicsAndNames();
  cout<<"Setup topic names from yaml file done"<<endl;

  //Setup publishers and Timers
  initSubsPubsAndTimers();
  cout<<"Initialized publishers, subscriber and timers"<<endl;

  //Setup rviz markers
  initRvizMarkers();
  cout<<"Initialized Rviz Markers"<<endl;

  //Setup ins state and evolution
  loadYamlParams();
  cout<<"loaded yaml params"<<endl;

  //setup x0
  x0_.setAllType(0);
  //x0_.setRAndCov(Matrix3d::Identity(),0.1*Matrix3d::Identity());
  x0_.setvAndCov(Vector3d::Zero(),0.0001*Matrix3d::Identity());
  x0_.setBaAndCov(Vector3d::Zero(),1e-5*Matrix3d::Identity());
  cout<<"assigned some initial values to the state"<<endl;

  //Setup Readiness object
  fr_.check_ready_.push_back(&cov_ctrl_gyr_);
  fr_.check_ready_.push_back(&cov_ctrl_acc_);
  fr_.check_ready_.push_back(&cov_ctrl_su_);
  fr_.check_ready_.push_back(&cov_ctrl_sa_);

  fr_.check_ready_.push_back(&cov_sens_mag_);
  fr_.check_ready_.push_back(&cov_sens_acc_);
  fr_.check_ready_.push_back(&cov_sens_gps_);
  fr_.check_ready_.push_back(&x0_);
  cout<<"Pushed readiness check to FilterReadiness.check_ready_ member "<<endl;

}

CallBackInsEkf::~CallBackInsEkf()
{

}


void
CallBackInsEkf::cbTimerPubTFCov(const ros::TimerEvent& event)
{
  static tf::TransformBroadcaster br;
  static SelfAdjointEigenSolver<Matrix3d> eig_solver;

  //send gps tf
  if(config_.dyn_tf_on && cov_sens_gps_.ready())
  {
    // Send GPS data for visualization
    tf::Transform trfm2;
    trfm2.setOrigin( tf::Vector3(xyz_gps_(0),xyz_gps_(1),xyz_gps_(2)) );
    tf::Quaternion q2;
    q2.setRPY(0, 0, 0);
    trfm2.setRotation(q2);

    br.sendTransform(tf::StampedTransform(trfm2, t_ep_gps_, strfrm_map_,strfrm_gps_lcl_));
  }

  //Send gps cov
  if(config_.dyn_enable_cov_disp_gps && cov_sens_gps_.ready())
  {
    marker_cov_gps_lcl_.header.stamp = t_ep_gps_;
    marker_cov_gps_lcl_.pose.position.x = xyz_gps_(0);
    marker_cov_gps_lcl_.pose.position.y = xyz_gps_(1);
    marker_cov_gps_lcl_.pose.position.z = xyz_gps_(2);
    marker_cov_gps_lcl_.scale.x = 2*sqrt(sens_gps_.R.diagonal()(0));
    marker_cov_gps_lcl_.scale.y = 2*sqrt(sens_gps_.R.diagonal()(1));
    marker_cov_gps_lcl_.scale.z = 2*sqrt(sens_gps_.R.diagonal()(2));
    marker_cov_gps_lcl_.color.a = config_.dyn_alpha_cov; // Don't forget to set the alpha!
    pub_viz_cov_.publish( marker_cov_gps_lcl_ );
  }

  //Sen base_link TF
  if(fr_.isReady())
  {
    Vector4d wxyz;
    SO3::Instance().g2quat(wxyz, x_.R);
    tf::Quaternion q_true(wxyz[1],wxyz[2],wxyz[3],wxyz[0]);

    tf::Transform trfm;
    trfm.setOrigin( tf::Vector3(x_.p[0],x_.p[1], x_.p[2]) );
    trfm.setRotation(q_true);

    if(config_.dyn_tf_on)
      br.sendTransform(tf::StampedTransform(trfm, ros::Time::now(), strfrm_map_, strfrm_robot_));

    // Send base_link cov
    if(config_.dyn_enable_cov_disp_est)
    {


      eig_solver.compute((x_.P.block<3,3>(9,9)).selfadjointView<Eigen::Upper>());
      Vector3d eig_vals = eig_solver.eigenvalues();
      Matrix3d eig_vecs = eig_solver.eigenvectors();
      if((eig_vecs.determinant()) < 0) // Make sure the rotation matrix is right handed
      {
        eig_vecs.col(0).swap(eig_vecs.col(1));
        eig_vals.row(0).swap(eig_vals.row(1));
      }
      Vector4d wxyz; SO3::Instance().g2quat(wxyz,eig_vecs);
      marker_cov_base_link_.header.stamp = ros::Time();
      marker_cov_base_link_.scale.x = 2*sqrt(eig_vals(0));
      marker_cov_base_link_.scale.y = 2*sqrt(eig_vals(1));
      marker_cov_base_link_.scale.z = 2*sqrt(eig_vals(2));
      marker_cov_base_link_.pose.position.x = x_.p[0];
      marker_cov_base_link_.pose.position.y = x_.p[1];
      marker_cov_base_link_.pose.position.z = x_.p[2];
      marker_cov_base_link_.pose.orientation.w=wxyz(0);
      marker_cov_base_link_.pose.orientation.x=wxyz(1);
      marker_cov_base_link_.pose.orientation.y=wxyz(2);
      marker_cov_base_link_.pose.orientation.z=wxyz(3);
      marker_cov_base_link_.color.a = config_.dyn_alpha_cov; // Don't forget to set the alpha!
      pub_viz_cov_.publish( marker_cov_base_link_ );
    }
  }
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
  nh_p_.getParam("strfrm_map", marker_cov_gps_lcl_.header.frame_id);
  marker_cov_gps_lcl_.ns = "insekf";
  marker_cov_gps_lcl_.id = id;
  marker_cov_gps_lcl_.type = visualization_msgs::Marker::SPHERE;
  marker_cov_gps_lcl_.action = visualization_msgs::Marker::ADD;
  marker_cov_gps_lcl_.pose.orientation.x = 0.0;
  marker_cov_gps_lcl_.pose.orientation.y = 0.0;
  marker_cov_gps_lcl_.pose.orientation.z = 0.0;
  marker_cov_gps_lcl_.pose.orientation.w = 1.0;
  marker_cov_gps_lcl_.color.r = 0.0;
  marker_cov_gps_lcl_.color.g = 1.0;
  marker_cov_gps_lcl_.color.b = 0.0;
  marker_cov_gps_lcl_.lifetime = ros::Duration(1);

  //Marker for displaying covariance of estimate
  id++;
  nh_p_.getParam("strfrm_map", marker_cov_base_link_.header.frame_id);
  marker_cov_base_link_.ns = "insekf";
  marker_cov_base_link_.id = id;
  marker_cov_base_link_.type = visualization_msgs::Marker::SPHERE;
  marker_cov_base_link_.action = visualization_msgs::Marker::ADD;
  marker_cov_base_link_.color.r = 1.0;
  marker_cov_base_link_.color.g = 0.0;
  marker_cov_base_link_.color.b = 0.0;
  marker_cov_gps_lcl_.lifetime = ros::Duration(1);
}


void
CallBackInsEkf::cbSubGps(const sensor_msgs::NavSatFix::ConstPtr& msg_gps)
{
  static bool first_call=true;

  Map<Matrix3d>cov_gps_msg((double*)msg_gps->position_covariance.data());

  //Get local coordinates(with map0_ being origin and X-Y-Z axis being East-North-Up

  Vector3d llh;     llh  << msg_gps->latitude*M_PI/180.0,msg_gps->longitude*M_PI/180.0, msg_gps->altitude ;
  Vector3d llh0;    llh0 << map0_(0)*M_PI/180.0         ,map0_(1)*M_PI/180.0          , msg_gps->altitude ;
  Vector3d xyz_gps; llhSI2EnuSI(xyz_gps, llh, llh0);


  //filter ready stuff
  if(!cov_sens_gps_.msg_recvd_)
    cov_sens_gps_.msg_recvd_=true;

  if(!cov_sens_gps_.ready())
  {
    if(cov_sens_gps_.type() == 1) //cov from msg
      cov_sens_gps_.msgCovReady(true);
    else if(cov_sens_gps_.type() == 2)//est cov
      cov_sens_gps_.tryEstCov(xyz_gps, first_call);
  }
  else
    cov_sens_gps_.updateCov(cov_gps_msg);//It updates everything based on what is the cov provider(either dyn, est or msg)


  if(!x0_.readypAndCov() && cov_sens_gps_.ready())
    x0_.setpAndCov(xyz_gps,cov_sens_gps_.cov());

  if(fr_.is_filtering_on_ && config_.dyn_gps_on)//perform a sensor update
  {
    InsState xs;
    Vector3d &zp=xyz_gps;// noisy measurements of position
    double t= (msg_gps->header.stamp - t_epoch_start_).toSec();
    kc_insgps_.Correct(x_temp_, t, x_, u_, zp);
    x_ = x_temp_;
    t_ep_x_ = msg_gps->header.stamp;
  }


  //Set the following variables for publishing tf and cov
  xyz_gps_ = xyz_gps;
  t_ep_gps_= msg_gps->header.stamp;

   if(first_call)
     first_call = false;
}

void
CallBackInsEkf::cbSubImu(const sensor_msgs::Imu::ConstPtr& msg_imu)
{
  static bool first_call=true;
  static ros::Time t_epoch_prev;

  acc_raw_ << msg_imu->linear_acceleration.x, msg_imu->linear_acceleration.y, msg_imu->linear_acceleration.z;
  gyr_raw_ << msg_imu->angular_velocity.x,    msg_imu->angular_velocity.y,    msg_imu->angular_velocity.z;
  acc_ = scale2si_acc_* (q_r2acc_* (acccal_trfm_* acc_raw_));
  gyr_ =   gyr_= scale2si_gyr_ * (q_r2gyr_ * gyr_raw_);
  u_ << gyr_, acc_;

  Matrix3d net_tfm_acc = scale2si_acc_* scale2si_acc_* q_r2acc_.matrix()*acccal_trfm_.linear();
  Matrix3d net_tfm_gyr = scale2si_gyr_* scale2si_gyr_* q_r2gyr_.matrix();

  //Get the aligned msg covariance
  Map<Matrix3d> cov_gyr_msg_unaligned((double*)msg_imu->angular_velocity_covariance.data());
  Map<Matrix3d> cov_acc_msg_unaligned((double*)msg_imu->linear_acceleration_covariance.data());
  Matrix3d cov_gyr_msg;cov_gyr_msg = net_tfm_gyr*cov_gyr_msg_unaligned*net_tfm_gyr.transpose();
  Matrix3d cov_acc_msg;cov_acc_msg = net_tfm_acc*cov_acc_msg_unaligned*net_tfm_acc.transpose();


  //Set filter ready stuff for accl
  if(!cov_sens_acc_.msg_recvd_)
    cov_sens_acc_.msg_recvd_ = true;

  if(!cov_ctrl_acc_.msg_recvd_)
    cov_ctrl_acc_.msg_recvd_ = true;

  if(!cov_sens_acc_.ready())
  {
    if(cov_sens_acc_.type() == 1) //cov from msg
      cov_sens_acc_.msgCovReady(true);
    else if(cov_sens_acc_.type() == 2)//est cov
      cov_sens_acc_.tryEstCov(acc_, first_call);
  }
  else
    cov_sens_acc_.updateCov(cov_acc_msg);//It updates everything based on what is the cov provider(either dyn, est or msg)

  if(!cov_ctrl_acc_.ready())
  {
    if(cov_ctrl_acc_.type() == 1) //cov from msg
      cov_ctrl_acc_.msgCovReady(true);
    else if(cov_ctrl_acc_.type() == 2)//est cov
      cov_ctrl_acc_.tryEstCov(acc_, first_call);
  }
  else
    cov_ctrl_acc_.updateCov(cov_acc_msg);//It updates everything based on what is the cov provider(either dyn, est or msg)

  if(!x0_.readyRAndCov() && cov_sens_mag_.msg_recvd_)
    x0_.tryComputeRAndCov(acc_, sens_acc_.a0,mag_, sens_mag_.m0, first_call);

  if(!x0_.readyBaAndCov() )
    x0_.tryComputeBaAndCov(acc_,first_call);

  //Set filter ready stuff  for Gyr stuff
  if(!cov_ctrl_gyr_.msg_recvd_ )
    cov_ctrl_gyr_.msg_recvd_ = true;

  if(!cov_ctrl_gyr_.ready())
  {
    if(cov_ctrl_gyr_.type() == 1) //cov from msg
      cov_ctrl_gyr_.msgCovReady(true);
    else if(cov_ctrl_gyr_.type() == 2)//est cov
      cov_ctrl_gyr_.tryEstCov(gyr_, first_call);
  }
  else
    cov_ctrl_gyr_.updateCov(cov_gyr_msg);//It updates everything based on what is the cov provider(either dyn, est or msg)

  if(!x0_.readyBgAndCov() && !fr_.isReady())
    x0_.tryComputeBgAndCov(gyr_,first_call);

  //Kalman filter prediction and update
   if(fr_.isReady())
   {
     if(!fr_.is_filtering_on_)
     {
       t_epoch_start_= msg_imu->header.stamp;
       t_epoch_prev = t_epoch_start_;
       fr_.is_filtering_on_=true;

       x_=x0_.state();

       cout << "*****t=0*****\n";
       cout << "initial state is as follows**************:"<<endl;
       cout << "R:\n"     << x_.R <<endl;
       cout << "R_cov:\n" << x_.P.block<3,3>(0,0)<<endl;
       cout << "bg:\n"    << x_.bg.transpose()<<endl;
       cout << "bg_cov:\n"<< x_.P.block<3,3>(3,3)<<endl;
       cout << "ba:\n"    << x_.ba.transpose()<<endl;
       cout << "ba_cov:\n"<< x_.P.block<3,3>(6,6)<<endl;
       cout << "p:\n"     << x_.p.transpose()<<endl;
       cout << "p_cov:\n" << x_.P.block<3,3>(9,9)<<endl;
       cout << "v:\n"     << x_.v.transpose()<<endl;
       cout << "v_cov:\n" << x_.P.block<3,3>(12,12)<<endl;
       cout <<"Initial Sens noise is as follows***********"<<endl;
       cout <<"sens_mag_.R:\n"<< sens_mag_.R <<endl;
       cout <<"sens_acc_.R:\n"<< sens_acc_.R <<endl;
       cout <<"sens_gps_.R:\n"<< sens_gps_.R <<endl;
       cout <<"Initial Ctrl noise is as follows***********"<<endl;
       cout <<"cov_ctrl_gyr_\n"<< cov_ctrl_gyr_.cov() <<endl;
       cout <<"ins.sv: "<< ins_.sv <<endl;
       cout <<"cov_ctrl_acc_\n"<< cov_ctrl_acc_.cov() <<endl;
       cout <<"ins.sra: "<< ins_.sra <<endl;
       cout <<"cov_ctrl_su_\n"<< cov_ctrl_su_.cov() <<endl;
       cout <<"ins.su: "<< ins_.su <<endl;
       cout <<"cov_ctrl_sa_\n"<< cov_ctrl_sa_.cov() <<endl;
       cout <<"ins.sa: "<< ins_.sa <<endl;
       if(pause_getchar_)
         getchar();
     }
     else
     {
       Vector3d& a = acc_;
       Vector3d& w = gyr_;
       Vector6d u;

       u << w, a;

       double t  = (msg_imu->header.stamp - t_epoch_start_).toSec();
       double dt = (msg_imu->header.stamp -   t_epoch_prev).toSec();


       t_epoch_prev = msg_imu->header.stamp;




       kp_ins_.Predict(x_temp_, t, x_, u, dt);
       double a0_tol; nh_p_.getParam("a0_tol",a0_tol);
       if(abs(a.norm() - sens_acc_.a0.norm())<a0_tol)
         kc_insimu_.Correct(x_, t, x_temp_, u_, a);
       else
       {
         cout<<"Acc of magnitude: "<< a.norm()<<" exceeded the a0_tol of "<<a0_tol<<". Skipping update step"<<endl;
         x_=x_temp_;
       }
       t_ep_x_ = msg_imu->header.stamp;

       //Display suff
       Vector6d u_b; u_b << x_.bg, x_.ba;
       Vector6d u_g; u_g <<Vector3d::Zero(),x_.R.transpose()*sens_acc_.a0;
       Vector6d u_f; u_f = u -u_b - u_g;
       Vector3d rpy;SO3::Instance().g2q(rpy,x_.R);
       cout << "****************\n";
       cout << "t:"<<t<< "\tdt:"<<dt <<endl;
       cout << "u(w,a):"<<u.transpose()<< endl;
       cout << "(bg,ba):"<<u_b.transpose()<< endl;
       cout << "u(w,a)-(bw,ba) - R'*g:"<<u_f.transpose()<< endl;
       cout << "position: " << x_.p.transpose() << endl;
       cout << "Estim attitude:\n" << x_.R << endl;

       //Publishing odometric message
       static nav_msgs::Odometry msg_odom;
       static uint32_t seq=0;seq++;

       msg_odom.header.frame_id= strfrm_map_;
       msg_odom.header.seq = seq;
       msg_odom.header.stamp = msg_imu->header.stamp;
       msg_odom.child_frame_id = strfrm_robot_;
       eig2PoseMsg(msg_odom.pose.pose,x_.R, x_.p);
       eig2TwistMsg(msg_odom.twist.twist, gyr_-x_.bg, x_.R.transpose()*x_.v);
       Map<Matrix6d>(msg_odom.pose.covariance.data()).block<3,3>(0,0).setZero();
       Map<Matrix6d>(msg_odom.pose.covariance.data()).block<3,3>(0,0) = x_.P.block<3,3>(9,9);//pos cov
       Map<Matrix6d>(msg_odom.pose.covariance.data()).block<3,3>(3,3) = x_.P.block<3,3>(0,0);//rot cov
       Map<Matrix6d>(msg_odom.twist.covariance.data()).block<3,3>(0,0).setZero();
       Map<Matrix6d>(msg_odom.twist.covariance.data()).block<3,3>(0,0) = x_.R.transpose()*x_.P.block<3,3>(12,12)*x_.R;//v cov
       Map<Matrix6d>(msg_odom.twist.covariance.data()).block<3,3>(3,3) = cov_ctrl_gyr_.cov()+ x_.P.block<3,3>(3,3);//w cov
       pub_odom_.publish(msg_odom);

       //publish for debugging
       static gcop_ros_est::InsekfDiag msg_diag;
       msg_diag.wx  = u(0);
       msg_diag.wy  = u(1);
       msg_diag.wz  = u(2);
       msg_diag.ax  = u(3);
       msg_diag.ay  = u(4);
       msg_diag.az  = u(5);

       msg_diag.bgx = u_b(0);
       msg_diag.bgy = u_b(1);
       msg_diag.bgz = u_b(2);
       msg_diag.bax = u_b(3);
       msg_diag.bay = u_b(4);
       msg_diag.baz = u_b(5);


       msg_diag.wfx = u_f(0);
       msg_diag.wfy = u_f(1);
       msg_diag.wfz = u_f(2);
       msg_diag.afx = u_f(3);
       msg_diag.afy = u_f(4);
       msg_diag.afz = u_f(5);

       msg_diag.x     = x_.p(0);
       msg_diag.y     = x_.p(1);
       msg_diag.z     = x_.p(2);

       msg_diag.vx    = x_.v(0);
       msg_diag.vy    = x_.v(1);
       msg_diag.vz    = x_.v(2);

       msg_diag.roll  = rpy(0);
       msg_diag.pitch = rpy(1);
       msg_diag.yaw   = rpy(2);

       msg_diag.p00   = x_.P(0,0);
       msg_diag.p11   = x_.P(1,1);
       msg_diag.p22   = x_.P(2,2);
       msg_diag.p33   = x_.P(3,3);
       msg_diag.p44   = x_.P(4,4);
       msg_diag.p55   = x_.P(5,5);
       msg_diag.p66   = x_.P(6,6);
       msg_diag.p77   = x_.P(7,7);
       msg_diag.p88   = x_.P(8,8);
       msg_diag.p99   = x_.P(9,9);
       msg_diag.p1010 = x_.P(10,10);
       msg_diag.p1111 = x_.P(11,11);
       msg_diag.p1212 = x_.P(12,12);
       msg_diag.p1313 = x_.P(13,13);
       msg_diag.p1414 = x_.P(14,14);

       if(config_.dyn_diag_on)
         pub_diag_.publish(msg_diag);
     }
   }
   else
   {
     cout<<"Initial State readiness "
         <<"R(" <<x0_.readyRAndCov()<<"),"
         <<"Bg("<<x0_.readyBgAndCov()<<"),"
         <<"Ba("<<x0_.readyBaAndCov()<<"),"
         <<"p(" <<x0_.readypAndCov()<<"),"<<endl;
     cout<<"Sensor & control readiness gps,acc,mag,sa,su,acc,gyr:";
     for_each(fr_.check_ready_.rbegin(), fr_.check_ready_.rend(),[&](RetReady* obj){cout<<obj->ready()<<",";});
     cout<<endl;
   }

  if(first_call)
    first_call=false;
}
void
CallBackInsEkf::cbSubMag(const sensor_msgs::MagneticField::ConstPtr& msg_mag)
{
  static bool first_call=true;

  // Extract sensor value
  Vector3d mag_raw;
  mag_raw << msg_mag->magnetic_field.x, msg_mag->magnetic_field.y, msg_mag->magnetic_field.z;
  mag_= magcal_trfm_*mag_raw;
  double norm_mag = mag_.norm();
  mag_.normalize();

  // net Linear transformation between raw data and aligned final mag data
  Matrix3d net_tfm_mag;
  net_tfm_mag = magcal_trfm_.linear()/(norm_mag*norm_mag);

  //Get aligned mag msg covariance
  Map<Matrix3d> cov_mag_msg_unaligned((double*)msg_mag->magnetic_field_covariance.data());
  Matrix3d cov_mag_msg; cov_mag_msg = net_tfm_mag * cov_mag_msg_unaligned * net_tfm_mag.transpose();

//  // Set filter ready stuff
  if(!cov_sens_mag_.msg_recvd_)
    cov_sens_mag_.msg_recvd_=true;

  if(!cov_sens_mag_.ready())
  {
    if(cov_sens_mag_.type() == 1) //cov from msg
      cov_sens_mag_.msgCovReady(true);
    else if(cov_sens_mag_.type() == 2)//est cov
      cov_sens_mag_.tryEstCov(mag_, first_call);
  }
  else
    cov_sens_mag_.updateCov(cov_mag_msg);
//  Matrix3d cov_mag0; cov_mag0.setZero();cov_mag0.diagonal()<<1.0,1.0,1e-4;
//  sens_mag_.R = x_.R.transpose()*cov_mag0*x_.R.transpose();

  if(fr_.is_filtering_on_ && config_.dyn_mag_on)
  {
    //Sensor update
    double t  = (msg_mag->header.stamp - t_epoch_start_).toSec();
    kc_insmag_.Correct(x_temp_, t, x_, u_, mag_);
    x_ = x_temp_;
    t_ep_x_ = msg_mag->header.stamp;
  }

  if(first_call)
    first_call=false;
}


void
CallBackInsEkf::cbSubMagV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg_mag)
{
  cout<<"got magnetometer vector"<<endl;
  static bool first_call=true;

   // Extract sensor value
   Vector3d mag_raw;
   Matrix3d cov_mag;
   mag_raw << msg_mag->vector.x, msg_mag->vector.y, msg_mag->vector.z;
   mag_= magcal_trfm_*mag_raw;
   double norm_mag = mag_.norm();
   mag_.normalize();

   //Get aligned mag msg covariance
   //not available here

   // Set filter ready stuff
   if(!cov_sens_mag_.msg_recvd_)
     cov_sens_mag_.msg_recvd_=true;

   if(!cov_sens_mag_.ready())
   {
     if(cov_sens_mag_.type() == 1) //cov from msg
     {
       cout<<"Tried to use message covariance but this mag message doesn't have one"<<endl;
       assert(0);
     }
     else if(cov_sens_mag_.type() == 2)//est cov
       cov_sens_mag_.tryEstCov(mag_, first_call);//use when msg has covariance
   }
   else
     cov_sens_mag_.updateCov(cov_mag);

   if(fr_.is_filtering_on_ && config_.dyn_mag_on)
   {
     //Sensor update
     double t  = (msg_mag->header.stamp - t_epoch_start_).toSec();
     kc_insmag_.Correct(x_temp_, t, x_, u_, mag_);
     x_ = x_temp_;
     t_ep_x_ = msg_mag->header.stamp;
   }

   if(first_call)
     first_call=false;
}

void
CallBackInsEkf::cbSubAccV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg_acc_v3s)
{
  static bool first_call=true;
  acc_raw_ << msg_acc_v3s->vector.x,msg_acc_v3s->vector.y,msg_acc_v3s->vector.z;
  acc_ = scale2si_acc_* (q_r2acc_* (acccal_trfm_* acc_raw_));

  // Set sensor ready stuff
  if(!cov_sens_acc_.msg_recvd_)
    cov_sens_acc_.msg_recvd_ = true;

  if(!cov_ctrl_acc_.msg_recvd_)
    cov_ctrl_acc_.msg_recvd_ = true;

  if(!cov_sens_acc_.ready())
  {
    if(cov_sens_acc_.type() == 1) //cov from msg
    {
      cout<<"Tried to use message covariance but this acc message doesn't have one"<<endl;
      assert(0);
    }
    else if(cov_sens_acc_.type() == 2)//est cov
      cov_sens_acc_.tryEstCov(acc_, first_call);
  }
  else
    cov_sens_acc_.updateCov();//It updates everything based on what is the cov provider(either dyn, est or msg)

  if(!cov_ctrl_acc_.ready())
  {
    if(cov_ctrl_acc_.type() == 1) //cov from msg
    {
      cout<<"Tried to use message covariance but this acc message doesn't have one"<<endl;
      assert(0);
    }
    else if(cov_ctrl_acc_.type() == 2)//est cov
      cov_ctrl_acc_.tryEstCov(acc_, first_call);
  }
  else
    cov_ctrl_acc_.updateCov();//It updates everything based on what is the cov provider(either dyn, est or msg)

  if(!x0_.readyRAndCov() && cov_sens_mag_.msg_recvd_)
    x0_.tryComputeRAndCov(acc_, sens_acc_.a0,mag_, sens_mag_.m0, first_call);

  if(!x0_.readyBaAndCov() )
    x0_.tryComputeBaAndCov(acc_,first_call);

 if(first_call)
   first_call=false;
}

void
CallBackInsEkf::cbSubGyrV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg_gyr_v3s)
{
  static bool first_call=true;
  static ros::Time t_epoch_prev;

 gyr_raw_ << msg_gyr_v3s->vector.x, msg_gyr_v3s->vector.y, msg_gyr_v3s->vector.z;
 gyr_ = scale2si_gyr_ * (q_r2gyr_ * gyr_raw_);
 u_ << gyr_, acc_;
 Matrix3d cov_gyr;

 // Set filter ready stuff
 if(!cov_ctrl_gyr_.msg_recvd_)
     cov_ctrl_gyr_.msg_recvd_ = true;


 if(!cov_ctrl_gyr_.ready())
 {
   if(cov_ctrl_gyr_.type() == 1) //cov from msg
   {
     cout<<"Tried to use message covariance but this gyr message doesn't have one"<<endl;
     assert(0);
   }
   else if(cov_ctrl_gyr_.type() == 2)//est cov
     cov_ctrl_gyr_.tryEstCov(gyr_, first_call);
 }
 else
   cov_ctrl_gyr_.updateCov(cov_gyr);//It updates everything based on what is the cov provider(either dyn, est or msg)

 if(!x0_.readyBgAndCov() )
   x0_.tryComputeBgAndCov(gyr_,first_call);

 //Kalman filter prediction and update
 if(fr_.isReady())
 {
   if(!fr_.is_filtering_on_)
   {
     t_epoch_start_= msg_gyr_v3s->header.stamp;
     t_epoch_prev = t_epoch_start_;
     fr_.is_filtering_on_=true;

     x_=x0_.state();

     cout << "*****t=0*****\n";
     cout << "initial state is as follows**************:"<<endl;
     cout << "R:\n"     << x_.R <<endl;
     cout << "R_cov:\n" << x_.P.block<3,3>(0,0)<<endl;
     cout << "bg:\n"    << x_.bg.transpose()<<endl;
     cout << "bg_cov:\n"<< x_.P.block<3,3>(3,3)<<endl;
     cout << "ba:\n"    << x_.ba.transpose()<<endl;
     cout << "ba_cov:\n"<< x_.P.block<3,3>(6,6)<<endl;
     cout << "p:\n"     << x_.p.transpose()<<endl;
     cout << "p_cov:\n" << x_.P.block<3,3>(9,9)<<endl;
     cout << "v:\n"     << x_.v.transpose()<<endl;
     cout << "v_cov:\n" << x_.P.block<3,3>(12,12)<<endl;
     cout <<"Initial Sens noise is as follows***********"<<endl;
     cout <<"sens_mag_.R:\n"<< sens_mag_.R <<endl;
     cout <<"sens_acc_.R:\n"<< sens_acc_.R <<endl;
     cout <<"sens_gps_.R:\n"<< sens_gps_.R <<endl;
     cout <<"Initial Ctrl noise is as follows***********"<<endl;
     cout <<"cov_ctrl_gyr_\n"<< cov_ctrl_gyr_.cov() <<endl;
     cout <<"ins.sv: "<< ins_.sv <<endl;
     cout <<"cov_ctrl_acc_\n"<< cov_ctrl_acc_.cov() <<endl;
     cout <<"ins.sra: "<< ins_.sra <<endl;
     cout <<"cov_ctrl_su_\n"<< cov_ctrl_su_.cov() <<endl;
     cout <<"ins.su: "<< ins_.su <<endl;
     cout <<"cov_ctrl_sa_\n"<< cov_ctrl_sa_.cov() <<endl;
     cout <<"ins.sa: "<< ins_.sa <<endl;
     if(pause_getchar_)
       getchar();
   }
   else
   {
     Vector3d& a = acc_;
     Vector3d& w = gyr_;
     Vector6d u;

     u << w, a;

     double t  = (msg_gyr_v3s->header.stamp - t_epoch_start_).toSec();
     double dt = (msg_gyr_v3s->header.stamp -   t_epoch_prev).toSec();


     t_epoch_prev = msg_gyr_v3s->header.stamp;




     kp_ins_.Predict(x_temp_, t, x_, u, dt);
     double a0_tol; nh_p_.getParam("a0_tol",a0_tol);
     if(abs(a.norm() - sens_acc_.a0.norm())<a0_tol)
       kc_insimu_.Correct(x_, t, x_temp_, u_, a);
     else
     {
       cout<<"Acc of magnitude: "<< a.norm()<<" exceeded the a0_tol of "<<a0_tol<<". Skipping update step"<<endl;
       x_=x_temp_;
     }
     t_ep_x_ = msg_gyr_v3s->header.stamp;

     //Display suff
     Vector6d u_b; u_b << x_.bg, x_.ba;
     Vector6d u_g; u_g <<Vector3d::Zero(),x_.R.transpose()*sens_acc_.a0;
     Vector6d u_f; u_f = u -u_b - u_g;
     Vector3d rpy;SO3::Instance().g2q(rpy,x_.R);
     cout << "****************\n";
     cout << "t:"<<t<< "\tdt:"<<dt <<endl;
     cout << "u(w,a):"<<u.transpose()<< endl;
     cout << "(bg,ba):"<<u_b.transpose()<< endl;
     cout << "u(w,a)-(bw,ba) - R'*g:"<<u_f.transpose()<< endl;
     cout << "position: " << x_.p.transpose() << endl;
     cout << "Estim attitude:\n" << x_.R << endl;

     //Publishing odometric message
     static nav_msgs::Odometry msg_odom;
     static uint32_t seq=0;seq++;

     msg_odom.header.frame_id= strfrm_map_;
     msg_odom.header.seq = seq;
     msg_odom.header.stamp = msg_gyr_v3s->header.stamp;
     msg_odom.child_frame_id = strfrm_robot_;
     eig2PoseMsg(msg_odom.pose.pose,x_.R, x_.p);
     eig2TwistMsg(msg_odom.twist.twist, gyr_-x_.bg, x_.R*x_.v);
     Map<Matrix6d>(msg_odom.pose.covariance.data()).block<3,3>(0,0).setZero();
     Map<Matrix6d>(msg_odom.pose.covariance.data()).block<3,3>(0,0) = x_.P.block<3,3>(9,9);//pos cov
     Map<Matrix6d>(msg_odom.pose.covariance.data()).block<3,3>(3,3) = x_.P.block<3,3>(0,0);//rot cov
     Map<Matrix6d>(msg_odom.twist.covariance.data()).block<3,3>(0,0).setZero();
     Map<Matrix6d>(msg_odom.twist.covariance.data()).block<3,3>(0,0) = x_.R.transpose()*x_.P.block<3,3>(12,12)*x_.R;//v cov
     Map<Matrix6d>(msg_odom.twist.covariance.data()).block<3,3>(3,3) = cov_ctrl_gyr_.cov()+ x_.P.block<3,3>(3,3);//w cov
     pub_odom_.publish(msg_odom);

     //publish for debugging
     static gcop_ros_est::InsekfDiag msg_diag;
     msg_diag.wx  = u(0);
     msg_diag.wy  = u(1);
     msg_diag.wz  = u(2);
     msg_diag.ax  = u(3);
     msg_diag.ay  = u(4);
     msg_diag.az  = u(5);

     msg_diag.bgx = u_b(0);
     msg_diag.bgy = u_b(1);
     msg_diag.bgz = u_b(2);
     msg_diag.bax = u_b(3);
     msg_diag.bay = u_b(4);
     msg_diag.baz = u_b(5);


     msg_diag.wfx = u_f(0);
     msg_diag.wfy = u_f(1);
     msg_diag.wfz = u_f(2);
     msg_diag.afx = u_f(3);
     msg_diag.afy = u_f(4);
     msg_diag.afz = u_f(5);

     msg_diag.x     = x_.p(0);
     msg_diag.y     = x_.p(1);
     msg_diag.z     = x_.p(2);

     msg_diag.vx    = x_.v(0);
     msg_diag.vy    = x_.v(1);
     msg_diag.vz    = x_.v(2);

     msg_diag.roll  = rpy(0);
     msg_diag.pitch = rpy(1);
     msg_diag.yaw   = rpy(2);

     msg_diag.p00   = x_.P(0,0);
     msg_diag.p11   = x_.P(1,1);
     msg_diag.p22   = x_.P(2,2);
     msg_diag.p33   = x_.P(3,3);
     msg_diag.p44   = x_.P(4,4);
     msg_diag.p55   = x_.P(5,5);
     msg_diag.p66   = x_.P(6,6);
     msg_diag.p77   = x_.P(7,7);
     msg_diag.p88   = x_.P(8,8);
     msg_diag.p99   = x_.P(9,9);
     msg_diag.p1010 = x_.P(10,10);
     msg_diag.p1111 = x_.P(11,11);
     msg_diag.p1212 = x_.P(12,12);
     msg_diag.p1313 = x_.P(13,13);
     msg_diag.p1414 = x_.P(14,14);

     if(config_.dyn_diag_on)
       pub_diag_.publish(msg_diag);
   }
 }
 else
 {
   cout<<"Initial State readiness "
       <<"R(" <<x0_.readyRAndCov()<<"),"
       <<"Bg("<<x0_.readyBgAndCov()<<"),"
       <<"Ba("<<x0_.readyBaAndCov()<<"),"
       <<"p(" <<x0_.readypAndCov()<<"),"<<endl;
   cout<<"Sensor & control readiness gps,acc,mag,sa,su,acc,gyr:";
   for_each(fr_.check_ready_.rbegin(), fr_.check_ready_.rend(),[&](RetReady* obj){cout<<obj->ready()<<",";});
   cout<<endl;
 }

 if(first_call)
   first_call=false;
 }
void
CallBackInsEkf::setupTopicsAndNames(void)
{
  if(config_.dyn_debug_on)
    cout<<"setting up topic names"<<endl;

  nh_p_.getParam("type_sensor_msg",type_sensor_msg_);

  nh_p_.getParam("strtop_gps",strtop_gps_);
  nh_p_.getParam("strtop_imu",strtop_imu_);
  nh_p_.getParam("strtop_mag",strtop_mag_);
  nh_p_.getParam("strtop_gyr_v3s",strtop_gyr_v3s_);
  nh_p_.getParam("strtop_acc_v3s",strtop_acc_v3s_);
  nh_p_.getParam("strtop_mag_v3s",strtop_mag_v3s_);

  nh_p_.getParam("strtop_odom",strtop_odom_);
  nh_p_.getParam("strtop_diag",strtop_diag_);
  nh_p_.getParam("strtop_marker_cov",strtop_marker_cov_);

  nh_p_.getParam("strfrm_map",strfrm_map_);
  nh_p_.getParam("strfrm_robot",strfrm_robot_);
  nh_p_.getParam("strfrm_gps_lcl",strfrm_gps_lcl_);

  if(config_.dyn_debug_on)
  {
    cout<<"Topics are:  "<<endl;
    if(type_sensor_msg_==1)
    {
      cout<<"strtop_gps:  "<<strtop_gps_<<endl;
      cout<<"strtop_imu:  "<<strtop_imu_<<endl;
      cout<<"strtop_mag:  "<<strtop_mag_<<endl;
    }
    else if(type_sensor_msg_==0)
    {
      cout<<"strtop_gyr_v3s:  "<<strtop_gyr_v3s_<<endl;
      cout<<"strtop_acc_v3s:  "<<strtop_acc_v3s_<<endl;
      cout<<"strtop_mag_v3s:  "<<strtop_mag_v3s_<<endl;
    }
    cout<<"strtop_odom:  "<<strtop_odom_<<endl;
    cout<<"strtop_diag:  "<<strtop_diag_<<endl;
    cout<<"strtop_marker_cov:  "<<strtop_marker_cov_<<endl;
  }
}

void
CallBackInsEkf::initSubsPubsAndTimers(void)
{
  //Publishers
  pub_viz_cov_ = nh_.advertise<visualization_msgs::Marker>( strtop_marker_cov_, 0 );
  pub_diag_ = nh_.advertise<gcop_ros_est::InsekfDiag>( strtop_diag_, 0 );
  pub_odom_ = nh_.advertise<nav_msgs::Odometry>( strtop_odom_, 0 );


  //Subscribers
  sub_gps_  = nh_.subscribe<sensor_msgs::NavSatFix>(strtop_gps_,1000,&CallBackInsEkf::cbSubGps, this,ros::TransportHints().tcpNoDelay());

  if(type_sensor_msg_==1)
  {
    sub_imu_ = nh_.subscribe<sensor_msgs::Imu>(strtop_imu_,1000,&CallBackInsEkf::cbSubImu, this, ros::TransportHints().tcpNoDelay());
    sub_mag_ = nh_.subscribe<sensor_msgs::MagneticField>(strtop_mag_,1000,&CallBackInsEkf::cbSubMag, this, ros::TransportHints().tcpNoDelay());
  }
  else if(type_sensor_msg_==0)
  {
    sub_acc_v3s_  = nh_.subscribe<geometry_msgs::Vector3Stamped>(strtop_acc_v3s_,1000,&CallBackInsEkf::cbSubAccV3S, this,ros::TransportHints().tcpNoDelay());
    sub_mag_v3s_  = nh_.subscribe<geometry_msgs::Vector3Stamped>(strtop_mag_v3s_,1000,&CallBackInsEkf::cbSubMagV3S, this,ros::TransportHints().tcpNoDelay());
    sub_gyr_v3s_  = nh_.subscribe<geometry_msgs::Vector3Stamped>(strtop_gyr_v3s_,1000,&CallBackInsEkf::cbSubGyrV3S, this,ros::TransportHints().tcpNoDelay());
  }
  else
  {
    cout<<"Wrong type_sensor_msg selected in yaml file"<<endl;
    assert(0);
  }

  //Timers
  //timer_general_ = nh_.createTimer(ros::Duration(1.0), &CallBackInsEkf::cbTimerGeneral, this);
  //timer_general_.start();
  double hz_tf;
  nh_p_.getParam("hz_tf",hz_tf);
  timer_send_tf_   =  nh_.createTimer(ros::Duration(1.0/hz_tf), &CallBackInsEkf::cbTimerPubTFCov, this);
  timer_send_tf_.start();

}

void
CallBackInsEkf::loadYamlParams(void)
{
  nh_p_.getParam("pause",pause_getchar_);

  //Number of reading for mean and covariance computation
  nh_p_.getParam("n_avg",x0_.n_avg_);
  nh_p_.getParam("n_avg",cov_sens_mag_.n_avg_);
  nh_p_.getParam("n_avg",cov_sens_acc_.n_avg_);
  nh_p_.getParam("n_avg",cov_ctrl_gyr_.n_avg_);
  nh_p_.getParam("n_avg",cov_sens_gps_.n_avg_);

  //InsState initialization
  Matrix3d rot;
  Vector3d vars,val;
  string type_rot, type_val, type_vars;

  type_rot = rosParam2StringMat(rot,"x0_R");
  type_vars = rosParam2StringMat(vars,"x0_R_cov");
  if(type_rot.compare("invalid") && type_vars.compare("invalid"))
    x0_.initRAndCov(type_rot,rot,type_vars,vars);
  else
    assert(0);

  type_val = rosParam2StringMat(val,"x0_bg");
  type_vars = rosParam2StringMat(vars,"x0_bg_cov");
  if(type_val.compare("invalid") && type_vars.compare("invalid"))
    x0_.initBgAndCov(type_val,val,type_vars,vars);
  else
    assert(0);


  //Set reference
  //a0: accelerometer, m0:magnetometer, map0_:gps(lat0(deg), lon0(deg), alt0(m)
  Vector3d a0,m0;
  rosParam2Mat(a0,"a0");
  rosParam2Mat(m0,"m0");
  ins_.g0         = a0;
  sens_acc_.a0 = a0;
  sens_acc_.m0 = m0.normalized();
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

  //Set Ctrl Noise
  //gyr noise
  cov_ctrl_gyr_.setPointersOfSD(0,&ins_.sv);
  //accel noise
  cov_ctrl_acc_.setPointersOfSD(0,&ins_.sra);
  //gyro bias rate-of-change white noise cov (spectral density)
  cov_ctrl_su_.setPointersOfSD(0,&ins_.su);
  //acceleration bias rate-of-change white noise cov (spectral density)
  cov_ctrl_sa_.setPointersOfSD(0,&ins_.sa);

  //Set Sens Noise
  //mag noise
  cov_sens_mag_.setPointersOfCov(&sens_mag_.R);
  //accel noise
  cov_sens_acc_.setPointersOfCov(&sens_acc_.R);
  //gps noise
  cov_sens_gps_.setPointersOfCov(&sens_gps_.R);

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
CallBackInsEkf::setFromParamsConfig()
{
  cout<<"*Loading params from yaml file to dynamic reconfigure"<<endl;

  nh_p_.getParam("debug_on",config_.dyn_debug_on);
  nh_p_.getParam("mag_on",config_.dyn_mag_on);
  nh_p_.getParam("gps_on",config_.dyn_gps_on);
  nh_p_.getParam("diag_on",config_.dyn_diag_on);
  nh_p_.getParam("tf_on",config_.dyn_tf_on);

  Vector3d cov;
  string type;

  //Sensor covariance
  type = rosParam2StringMat(cov,"cov_sens_mag");
  if(type.compare("invalid"))
    cov_sens_mag_.initCov(type,cov);
  else
    assert(0);

  type = rosParam2StringMat(cov,"cov_sens_acc");
  if(type.compare("invalid"))
    cov_sens_acc_.initCov(type,cov);
  else
    assert(0);

  type = rosParam2StringMat(cov,"cov_sens_gps");
  if(type.compare("invalid"))
    cov_sens_gps_.initCov(type,cov);
  else
    assert(0);

  //Ctrl covariance
  type = rosParam2StringMat(cov,"cov_ctrl_gyr");
  if(type.compare("invalid"))
    cov_ctrl_gyr_.initCov(type,cov);
  else
    assert(0);

  type = rosParam2StringMat(cov,"cov_ctrl_acc");
  if(type.compare("invalid"))
    cov_ctrl_acc_.initCov(type,cov);
  else
    assert(0);

  type = rosParam2StringMat(cov,"cov_ctrl_su");
  if(type.compare("invalid"))
    cov_ctrl_su_.initCov(type,cov);
  else
    assert(0);

  type = rosParam2StringMat(cov,"cov_ctrl_sa");
  if(type.compare("invalid"))
    cov_ctrl_sa_.initCov(type,cov);
  else
    assert(0);
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
    switch(level)
    {
      case 1:
        break;
      case 2:
        break;
      case 4://sens cov change
        cov_sens_acc_.updateCov();
        cov_sens_gps_.updateCov();
        cov_sens_mag_.updateCov();
        break;
      case 8://ctrl cov change
        cov_ctrl_acc_.updateCov();
        cov_ctrl_gyr_.updateCov();
        cov_ctrl_su_.updateCov();
        cov_ctrl_sa_.updateCov();
        break;
      case 16:
        break;
    }
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









