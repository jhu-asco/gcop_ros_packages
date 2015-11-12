/*
 * sim_ins_mag_gps.cpp
 *
 *  Created on: Jul 20, 2015
 *      Author: subhransu
 *  TODO: Remove all assert(0) lines and handle them gracefully
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
#include <fstream>
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
#include <yaml-cpp/yaml.h>
#include "yaml_eig_conv.h"
#include <algorithm>

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

typedef Eigen::Matrix<double, 4, 4> Matrix4d;
typedef Eigen::Matrix<double, 4, 1> Vector4d;
typedef Eigen::Matrix<double, 7, 1> Vector7d;
typedef Eigen::Transform<double,3, Affine> Transform3d;
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

class AccNode
{
public:
  ros::Time t_ep_;
  Vector3d a_;
  AccNode(ros::Time t_ep, Vector3d a):t_ep_(t_ep), a_(a){}
  bool operator < (ros::Time t_ep) const
  {
    return t_ep_ < t_ep;
  }
};

enum SensorType{
  GPS, MAG, POSE3D
};

class SensNode
{
public:
  SensNode(double t, SensorType type):t_(t),type_(type)
{

}

  void addVec3d(const Vector3d& vec, const Vector3d& var)
  {
    val_ .reset( new Vector3d(vec));
    var_ .reset( new Vector3d(var));
    assert(type_==SensorType::GPS || type_==SensorType::MAG);
  }

  void addPose3d(const Transform3d& tfm, const Vector6d& vec)
  {
    pose_ .reset( new Transform3d(tfm));
    pose_var_ .reset( new Vector6d(vec));
    assert(type_==SensorType::POSE3D);
  }

  bool operator < (const SensNode& sn) const
  {
    return (t_< sn.t_);
  }

  double t_;
  shared_ptr<Vector3d> val_;
  shared_ptr<Vector3d> var_;
  shared_ptr<Transform3d> pose_;
  shared_ptr<Vector6d>    pose_var_;
  SensorType type_;
};

class FilterNode
{
public:
  FilterNode(double t, InsState x, Vector3d a, Vector3d a_var, Vector3d w, Vector3d w_var):
    t_(t), x_(x),
    ctrl_a_(a),          ctrl_a_var_(a_var),
    ctrl_w_(w),          ctrl_w_var_(w_var)
{

}
  //adding a sensor reading will perform a sensor update on the present state value
  void addSensAcc(const Vector3d& val, const Vector3d& var)
  {
    sens_a_       .reset( new Vector3d(val));
    sens_a_var_   .reset( new Vector3d(var));
  }

  void addSensPos(const Vector3d& val, const Vector3d& var)
  {
    sens_pos_     .reset( new Vector3d(val));
    sens_pos_var_ .reset( new Vector3d(var));
  }

  void addSensMag(const Vector3d& val, const Vector3d& var)
  {
    sens_mag_     .reset( new Vector3d(val));
    sens_mag_var_ .reset( new Vector3d(var));

  }

  void addSensPose(const Transform3d& val, const Vector6d& var)
  {
    sens_pose_    .reset( new Transform3d(val));
    sens_pose_var_.reset( new Vector6d(var));
  }

  bool operator < (const SensNode& sn) const
  {
    return (t_< sn.t_);
  }

  bool operator < (const FilterNode& fn) const
  {
    return (t_< fn.t_);
  }

public:
  double                  t_;
  InsState                x_;           //State after prediction from prev x using ctrl below and update step if any as below
  Vector3d                ctrl_a_;      //Ctrl
  Vector3d                ctrl_a_var_;
  Vector3d                ctrl_w_;
  Vector3d                ctrl_w_var_;
  shared_ptr<Vector3d>    sens_a_;
  shared_ptr<Vector3d>    sens_a_var_;
  shared_ptr<Vector3d>    sens_pos_;
  shared_ptr<Vector3d>    sens_pos_var_;
  shared_ptr<Vector3d>    sens_mag_;
  shared_ptr<Vector3d>    sens_mag_var_;
  shared_ptr<Transform3d> sens_pose_;
  shared_ptr<Vector6d>    sens_pose_var_;        //First 3 for rotation and next 3 for translation

};

class CallBackInsEkf
{
public:

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

    void initRAndCov(const pair<string,Matrix3d>&type_n_rot,const pair<string,Vector3d>&type_n_vars)
    {
      const string&   type_rot  = type_n_rot.first;
      const Matrix3d& rot       = type_n_rot.second;
      const string&   type_vars = type_n_vars.first;
      const Vector3d& vars      = type_n_vars.second;
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

    void initBgAndCov(const pair<string,Vector3d>&type_n_val, const pair<string,Vector3d>& type_n_vars)
    {
      const string&   type_val  = type_n_val.first;
      const Vector3d& val       = type_n_val.second;
      const string&   type_vars = type_n_vars.first;
      const Vector3d& vars      = type_n_vars.second;
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

    void initBaAndCov(const pair<string,Vector3d>&type_n_val, const pair<string,Vector3d>& type_n_vars)
    {
      const string&   type_val  = type_n_val.first;
      const Vector3d& val       = type_n_val.second;
      const string&   type_vars = type_n_vars.first;
      const Vector3d& vars      = type_n_vars.second;

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

    void initpAndCov(const pair<string,Vector3d>&type_n_val, const pair<string,Vector3d>& type_n_vars)
     {
       const string&   type_val  = type_n_val.first;
       const Vector3d& val       = type_n_val.second;
       const string&   type_vars = type_n_vars.first;
       const Vector3d& vars      = type_n_vars.second;

       if(type_val.compare(type_vars))
       {
         cout<<"Selection method for p and it's covariance is not the same"<<endl;
         assert(0);
       }
       if(!type_val.compare("yaml"))
       {
         type_p_=0;
         type_p_cov_=0;
         Matrix3d cov; cov.setZero();
         cov.diagonal() = vars;
         setpAndCov(val,cov);
       }
       else if(!type_val.compare("est"))
       {
         type_p_=1;
         type_p_cov_=1;
       }
       else
         assert(0);
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

    void initvAndCov(const pair<string,Vector3d>&type_n_val, const pair<string,Vector3d>& type_n_vars)
      {
        const string&   type_val  = type_n_val.first;
        const Vector3d& val       = type_n_val.second;
        const string&   type_vars = type_n_vars.first;
        const Vector3d& vars      = type_n_vars.second;

        if(type_val.compare(type_vars))
        {
          cout<<"Selection method for p and it's covariance is not the same"<<endl;
          assert(0);
        }
        if(!type_val.compare("yaml"))
        {
          type_v_=0;
          type_v_cov_=0;
          Matrix3d cov; cov.setZero();
          cov.diagonal() = vars;
          setvAndCov(val,cov);
        }
        else if(!type_val.compare("est"))
        {
          type_v_=1;
          type_v_cov_=1;
        }
        else
          assert(0);
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

    void initCov(const pair<string,Vector3d>& type_n_vars)
    {
      const string&   type =  type_n_vars.first;
      const Vector3d& vars  = type_n_vars.second;
      if(!type.compare("dyn"))
      {
        type_=0;
        config_cov_x_=vars(0);
        config_cov_y_=vars(1);
        config_cov_z_=vars(2);
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
  void cbTimerPubOdom(const ros::TimerEvent& event);
  void cbTimerPubDiag(const ros::TimerEvent& event);
  void cbTimerGeneral(const ros::TimerEvent& event);
  void cbSubGps(const sensor_msgs::NavSatFix::ConstPtr& msg_gps);
  void cbSubImu(const sensor_msgs::Imu::ConstPtr& msg);
  void updateOdomAndDiagMsg(void);
  void cbSubMag(const sensor_msgs::MagneticField::ConstPtr& msg_mag);
  void cbSubAccV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg);
  void cbSubMagV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg_mag_v3s);
  void cbSubGyrV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg);

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
  ros::Timer timer_general_, timer_send_tf_, timer_send_odom_, timer_send_diag_;
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
  SelCov cov_sens_mag_,cov_sens_acc_, cov_sens_pos_;
  SelCov cov_ctrl_gyr_, cov_ctrl_acc_, cov_ctrl_su_, cov_ctrl_sa_;
  Vector3d mag_, acc_, gyr_, acc_raw_, gyr_raw_;
  Vector3d map0_;//map reference in lat(deg) lon(deg) and alt(m)
  double a0_tol_;
  double scale2si_gyr_, scale2si_acc_;
  Transform<double,3, Affine> magcal_trfm_, acccal_trfm_;
  bool pause_getchar_;
  nav_msgs::Odometry msg_odom_;
  gcop_ros_est::InsekfDiag msg_diag_;
  double hz_tf_, hz_odom_, hz_diag_;
  YAML::Node yaml_node_;
  deque<FilterNode> filt_nodes_;
  deque<SensNode> sens_nodes_;
  deque<AccNode> acc_buffer_;
  int n_max_scs_;

  //Kalman filter
  InsKalmanPredictor kp_ins_;
  InsImuKalmanCorrector kc_insimu_;
  InsGpsKalmanCorrector kc_insgps_;
  InsMagKalmanCorrector    kc_insmag_;

  //Sensors
  InsImu<3>   sens_acc_;
  InsGps<>    sens_pos_;
  InsMag<>    sens_mag_;
  //Vector3d  ctrl_w_true_, ctrl_a_true_, ctrl_w_drft_, ctrl_a_drft_;

  public:
  ros::Rate loop_rate_;
};

CallBackInsEkf::CallBackInsEkf():
        nh_p_("~"),
        n_max_scs_(100),
        loop_rate_(1000),
        t_(0),
        kp_ins_(ins_),
        kc_insimu_(ins_.X, sens_acc_),
        kc_insgps_(ins_.X, sens_pos_),
        kc_insmag_(ins_.X, sens_mag_),
        config_(),
        cov_sens_mag_(config_.dyn_cov_sens_mag,   config_.dyn_cov_sens_mag,   config_.dyn_cov_sens_mag),
        cov_sens_acc_(config_.dyn_cov_sens_acc,   config_.dyn_cov_sens_acc,   config_.dyn_cov_sens_acc),
        cov_sens_pos_(config_.dyn_cov_sens_pos_xy,config_.dyn_cov_sens_pos_xy,config_.dyn_cov_sens_pos_z),
        cov_ctrl_gyr_(config_.dyn_cov_ctrl_gyr,   config_.dyn_cov_ctrl_gyr,   config_.dyn_cov_ctrl_gyr),
        cov_ctrl_acc_(config_.dyn_cov_ctrl_acc,   config_.dyn_cov_ctrl_acc,   config_.dyn_cov_ctrl_acc),
        cov_ctrl_su_(config_.dyn_cov_ctrl_su,   config_.dyn_cov_ctrl_su,   config_.dyn_cov_ctrl_su),
        cov_ctrl_sa_(config_.dyn_cov_ctrl_sa,   config_.dyn_cov_ctrl_sa,   config_.dyn_cov_ctrl_sa)

{
  cout<<"*Entering constructor of cbc"<<endl;

  //Setup YAML reading and parsing
  string strfile_params;nh_p_.getParam("strfile_params",strfile_params);
#ifdef HAVE_NEW_YAMLCPP
  cout<<"loading yaml param file into yaml_node"<<endl;
  yaml_node_ = YAML::LoadFile(strfile_params);
#else
  cout<<"Wrong Yaml version used. Change source code or install version>0.5"<<endl;
  assert(0);
#endif

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

  //Setup Readiness object
  fr_.check_ready_.push_back(&cov_ctrl_gyr_);
  fr_.check_ready_.push_back(&cov_ctrl_acc_);
  fr_.check_ready_.push_back(&cov_ctrl_su_);
  fr_.check_ready_.push_back(&cov_ctrl_sa_);

  fr_.check_ready_.push_back(&cov_sens_mag_);
  fr_.check_ready_.push_back(&cov_sens_acc_);
  fr_.check_ready_.push_back(&cov_sens_pos_);
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
  if(hz_tf_>=0.0 && cov_sens_pos_.ready())
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
  if(config_.dyn_enable_cov_disp_gps && cov_sens_pos_.ready())
  {
    marker_cov_gps_lcl_.header.stamp = t_ep_gps_;
    marker_cov_gps_lcl_.pose.position.x = xyz_gps_(0);
    marker_cov_gps_lcl_.pose.position.y = xyz_gps_(1);
    marker_cov_gps_lcl_.pose.position.z = xyz_gps_(2);
    marker_cov_gps_lcl_.scale.x = 2*sqrt(sens_pos_.R.diagonal()(0));
    marker_cov_gps_lcl_.scale.y = 2*sqrt(sens_pos_.R.diagonal()(1));
    marker_cov_gps_lcl_.scale.z = 2*sqrt(sens_pos_.R.diagonal()(2));
    marker_cov_gps_lcl_.color.a = config_.dyn_alpha_cov; // Don't forget to set the alpha!
    pub_viz_cov_.publish( marker_cov_gps_lcl_ );
  }

  //Sen base_link TF
  if(fr_.is_filtering_on_)
  {
    Vector4d wxyz;
    SO3::Instance().g2quat(wxyz, x_.R);
    tf::Quaternion q_true(wxyz[1],wxyz[2],wxyz[3],wxyz[0]);
    tf::Transform trfm;
    trfm.setOrigin( tf::Vector3(x_.p[0],x_.p[1], x_.p[2]) );
    trfm.setRotation(q_true);

    if(hz_tf_>=0)
      br.sendTransform(tf::StampedTransform(trfm, t_ep_x_, strfrm_map_, strfrm_robot_));

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
      marker_cov_base_link_.header.stamp = t_ep_x_;
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
CallBackInsEkf::cbTimerPubOdom(const ros::TimerEvent& event)
{
  static int seq=0;
  static ros::Time t_ep_prev;
  bool cond_filt_on = fr_.is_filtering_on_;
  bool cond_first_pub = t_ep_prev.isZero();
  bool cond_not_duplicate = !(msg_odom_.header.stamp-t_ep_prev).isZero();
  if(  cond_filt_on && (cond_first_pub || cond_not_duplicate))
  {
    pub_odom_.publish(msg_odom_);
    seq++;
    t_ep_prev = msg_odom_.header.stamp;
  }
}

void
CallBackInsEkf::cbTimerPubDiag(const ros::TimerEvent& event)
{
  static int seq=0;
  static ros::Time t_ep_prev;
  bool cond_filt_on = fr_.is_filtering_on_;
  bool cond_first_pub = t_ep_prev.isZero();
  bool cond_not_duplicate = !(msg_diag_.header.stamp-t_ep_prev).isZero();
  if(  cond_filt_on && (cond_first_pub || cond_not_duplicate))
  {
    pub_diag_.publish(msg_diag_);
    seq++;
    t_ep_prev = msg_diag_.header.stamp;
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
  marker_cov_gps_lcl_.header.frame_id = strfrm_map_;
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
  marker_cov_base_link_.header.frame_id = strfrm_map_;
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
  if(!cov_sens_pos_.msg_recvd_)
    cov_sens_pos_.msg_recvd_=true;

  if(!cov_sens_pos_.ready())
  {
    if(cov_sens_pos_.type() == 1) //cov from msg
      cov_sens_pos_.msgCovReady(true);
    else if(cov_sens_pos_.type() == 2)//est cov
      cov_sens_pos_.tryEstCov(xyz_gps, first_call);
  }
  else
    cov_sens_pos_.updateCov(cov_gps_msg);//It updates everything based on what is the cov provider(either dyn, est or msg)


  if(!x0_.readypAndCov() && cov_sens_pos_.ready())
    x0_.setpAndCov(xyz_gps,cov_sens_pos_.cov());

  if(fr_.is_filtering_on_ && config_.dyn_gps_on)//perform a sensor update
  {
    sens_nodes_.push_back(SensNode((msg_gps->header.stamp - t_epoch_start_).toSec(), SensorType::GPS));
    sens_nodes_.back().addVec3d(xyz_gps,cov_sens_pos_.cov().diagonal());
  }


  //Set the following variables for publishing tf and cov
  xyz_gps_ = xyz_gps;
  t_ep_gps_= msg_gps->header.stamp;

  if(first_call)
    first_call = false;
}

void
CallBackInsEkf::cbSubImu(const sensor_msgs::Imu::ConstPtr& msg)
{
  static bool first_call=true;
  static ros::Time t_epoch_prev;

  acc_raw_ << msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z;
  gyr_raw_ << msg->angular_velocity.x,    msg->angular_velocity.y,    msg->angular_velocity.z;
  acc_ = scale2si_acc_* (q_r2acc_* (acccal_trfm_* acc_raw_));
  gyr_ =   gyr_= scale2si_gyr_ * (q_r2gyr_ * gyr_raw_);
  u_ << gyr_, acc_;

  Matrix3d net_tfm_acc = scale2si_acc_* scale2si_acc_* q_r2acc_.matrix()*acccal_trfm_.linear();
  Matrix3d net_tfm_gyr = scale2si_gyr_* scale2si_gyr_* q_r2gyr_.matrix();

  //Get the aligned msg covariance
  Map<Matrix3d> cov_gyr_msg_unaligned((double*)msg->angular_velocity_covariance.data());
  Map<Matrix3d> cov_acc_msg_unaligned((double*)msg->linear_acceleration_covariance.data());
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
      t_epoch_start_= msg->header.stamp;
      t_epoch_prev = t_epoch_start_;
      fr_.is_filtering_on_=true;

      x_=x0_.state();
      filt_nodes_.push_back(FilterNode(0, x_,acc_, cov_ctrl_acc_.cov().diagonal(), gyr_, cov_ctrl_gyr_.cov().diagonal()));
      filt_nodes_.back().addSensAcc(acc_, cov_ctrl_acc_.cov().diagonal());


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
      cout <<"sens_pos_.R:\n"<< sens_pos_.R <<endl;
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

      //keep the size of the buffer at n_max_scs_
      while(filt_nodes_.size()>n_max_scs_-1)
        filt_nodes_.pop_front();

      double t  = (msg->header.stamp - t_epoch_start_).toSec();
      double dt = (msg->header.stamp -   t_epoch_prev).toSec();
      t_epoch_prev = msg->header.stamp;
      if(config_.dyn_debug_on)
      {
        cout << "****************\n";
        cout << "t:"<<t<< "\tdt:"<<dt <<endl;
      }
      //Enter a new FilterNode to filt_nodes for the current time step
      filt_nodes_.push_back(FilterNode(t, x0_.state(),acc_, cov_ctrl_acc_.cov().diagonal(), gyr_, cov_ctrl_gyr_.cov().diagonal()));

      //Add acc_sens reading to the latest filter node
      filt_nodes_.back().addSensAcc(acc_, cov_ctrl_acc_.cov().diagonal());

      //sort all the sensor readings which have not been added to FilterNode and add them at appropriate places
      sort(sens_nodes_.begin(), sens_nodes_.end());

      //add the sorted sensor reading at the right node in filt_nodes_ or just reject it if it's too old
      deque<FilterNode>::iterator plb = filt_nodes_.begin(); //prev lower bound
      deque<FilterNode>::iterator flb = filt_nodes_.end()-1; //first lower bound
      bool first_valid_update=true;
      deque<SensNode>::iterator it_sn;
      for(it_sn = sens_nodes_.begin(); it_sn!=sens_nodes_.end();it_sn++)
      {
        if(it_sn->t_>filt_nodes_.back().t_)
          break;

        deque<FilterNode>::iterator lb = lower_bound(plb,filt_nodes_.end(), *it_sn);
        if(lb == filt_nodes_.begin()) //reject a reading because it's old
          continue;
        else                        //add a reading to the right filter_node
        {
          if(first_valid_update)
          {
            flb = lb;
            first_valid_update = false;
          }
          plb = lb;
          switch(it_sn->type_)
          {
            case SensorType::GPS:
              lb->addSensPos(*(it_sn->val_),*(it_sn->var_));
              break;
            case SensorType::MAG:
              lb->addSensMag(*(it_sn->val_),*(it_sn->var_));
              break;
            case SensorType::POSE3D:
              lb->addSensPose(*(it_sn->pose_),*(it_sn->pose_var_));
              break;
          }
        }
      }
      //delete the sensor measurements which were either added or rejected
      sens_nodes_.erase(sens_nodes_.begin(),it_sn);

      //predict and update from the first update node to the last node
      for(deque<FilterNode>::iterator it=flb; it!=filt_nodes_.end(); it++)
      {
        //Perform prediction from prev state( (it-1)->x_) to current state
        Vector6d u; u <<(it-1)->ctrl_w_, (it-1)->ctrl_a_;
        ins_.sra = sqrt((it-1)->ctrl_a_var_(0));
        ins_.sv  = sqrt((it-1)->ctrl_w_var_(0));
        kp_ins_.Predict(it->x_, (it-1)->t_, (it-1)->x_, u, dt);

        //perform update for all the sensors
        if(it->sens_a_)
        {
          InsState xa = it->x_;
          sens_acc_.R.setZero();sens_acc_.R.diagonal()= *(it->sens_a_var_);
          if(abs(it->sens_a_->norm() - sens_acc_.a0.norm()) < a0_tol_)
            kc_insimu_.Correct(it->x_, it->t_, xa, u, *(it->sens_a_));
        }
        if(it->sens_mag_)
        {
          InsState xa = it->x_;
          sens_mag_.R.setZero();sens_mag_.R.diagonal()= *(it->sens_mag_var_);
          kc_insmag_.Correct(it->x_, it->t_, xa, u, *(it->sens_mag_));
        }
        if(it->sens_pos_)
        {
          InsState xa = it->x_;
          sens_pos_.R.setZero();sens_pos_.R.diagonal()= *(it->sens_pos_var_);
          kc_insgps_.Correct(it->x_, it->t_, x_, u, *(it->sens_pos_));
        }
        if(it->sens_pose_)
        {
          cout<<"sens_pose_ update unimplemented"<<endl;
          assert(0);
        }
      }
      x_ = filt_nodes_.back().x_;
      t_ep_x_ = msg->header.stamp;

      updateOdomAndDiagMsg();
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
CallBackInsEkf::updateOdomAndDiagMsg(void)
{

  Vector6d u_b; u_b << x_.bg, x_.ba;
  Vector6d u_g; u_g <<Vector3d::Zero(),x_.R.transpose()*sens_acc_.a0;
  Vector6d u_f; u_f = u_ -u_b - u_g;
  Vector3d rpy;SO3::Instance().g2q(rpy,x_.R);

  //Display suff
  if(config_.dyn_debug_on)
  {
    cout << "u(w,a):"<<u_.transpose()<< endl;
    cout << "(bg,ba):"<<u_b.transpose()<< endl;
    cout << "u(w,a)-(bw,ba) - R'*g:"<<u_f.transpose()<< endl;
    cout << "position: " << x_.p.transpose() << endl;
    cout << "Estim attitude:\n" << x_.R << endl;
  }

  //Publishing odometric message
  static int seq_odom=0;
  msg_odom_.header.frame_id= strfrm_map_;
  msg_odom_.header.stamp = t_ep_x_;
  msg_odom_.child_frame_id = strfrm_robot_;
  eig2PoseMsg(msg_odom_.pose.pose,x_.R, x_.p);
  eig2TwistMsg(msg_odom_.twist.twist, gyr_-x_.bg, x_.R.transpose()*x_.v);
  Map<Matrix6d>(msg_odom_.pose.covariance.data()).block<3,3>(0,0).setZero();
  Map<Matrix6d>(msg_odom_.pose.covariance.data()).block<3,3>(0,0) = x_.P.block<3,3>(9,9);//pos cov
  Map<Matrix6d>(msg_odom_.pose.covariance.data()).block<3,3>(3,3) = x_.P.block<3,3>(0,0);//rot cov
  Map<Matrix6d>(msg_odom_.twist.covariance.data()).block<3,3>(0,0).setZero();
  Map<Matrix6d>(msg_odom_.twist.covariance.data()).block<3,3>(0,0) = x_.R.transpose()*x_.P.block<3,3>(12,12)*x_.R;//v cov
  Map<Matrix6d>(msg_odom_.twist.covariance.data()).block<3,3>(3,3) = cov_ctrl_gyr_.cov()+ x_.P.block<3,3>(3,3);//w cov
  if(hz_odom_==0)
  {
    msg_odom_.header.seq=seq_odom;
    pub_odom_.publish(msg_odom_);
    seq_odom++;
  }

  //publish diagnostics
  static int seq_diag=0;
  msg_diag_.header.frame_id= "none";
  msg_diag_.header.stamp = t_ep_x_;
  msg_diag_.wx  = u_(0);
  msg_diag_.wy  = u_(1);
  msg_diag_.wz  = u_(2);
  msg_diag_.ax  = u_(3);
  msg_diag_.ay  = u_(4);
  msg_diag_.az  = u_(5);

  msg_diag_.bgx = u_b(0);
  msg_diag_.bgy = u_b(1);
  msg_diag_.bgz = u_b(2);
  msg_diag_.bax = u_b(3);
  msg_diag_.bay = u_b(4);
  msg_diag_.baz = u_b(5);


  msg_diag_.wfx = u_f(0);
  msg_diag_.wfy = u_f(1);
  msg_diag_.wfz = u_f(2);
  msg_diag_.afx = u_f(3);
  msg_diag_.afy = u_f(4);
  msg_diag_.afz = u_f(5);

  msg_diag_.x     = x_.p(0);
  msg_diag_.y     = x_.p(1);
  msg_diag_.z     = x_.p(2);

  msg_diag_.vx    = x_.v(0);
  msg_diag_.vy    = x_.v(1);
  msg_diag_.vz    = x_.v(2);

  msg_diag_.roll  = rpy(0);
  msg_diag_.pitch = rpy(1);
  msg_diag_.yaw   = rpy(2);

  msg_diag_.p00   = x_.P(0,0);
  msg_diag_.p11   = x_.P(1,1);
  msg_diag_.p22   = x_.P(2,2);
  msg_diag_.p33   = x_.P(3,3);
  msg_diag_.p44   = x_.P(4,4);
  msg_diag_.p55   = x_.P(5,5);
  msg_diag_.p66   = x_.P(6,6);
  msg_diag_.p77   = x_.P(7,7);
  msg_diag_.p88   = x_.P(8,8);
  msg_diag_.p99   = x_.P(9,9);
  msg_diag_.p1010 = x_.P(10,10);
  msg_diag_.p1111 = x_.P(11,11);
  msg_diag_.p1212 = x_.P(12,12);
  msg_diag_.p1313 = x_.P(13,13);
  msg_diag_.p1414 = x_.P(14,14);
  if(hz_diag_==0)
  {
    msg_diag_.header.seq=seq_diag;
    pub_diag_.publish(msg_diag_);
    seq_diag++;
  }
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
    sens_nodes_.push_back(SensNode((msg_mag->header.stamp - t_epoch_start_).toSec(), SensorType::MAG));
    sens_nodes_.back().addVec3d(mag_,cov_sens_mag_.cov().diagonal());
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
    sens_nodes_.push_back(SensNode((msg_mag->header.stamp - t_epoch_start_).toSec(), SensorType::MAG));
    sens_nodes_.back().addVec3d(mag_,cov_sens_mag_.cov().diagonal());
  }

  if(first_call)
    first_call=false;
}

void
CallBackInsEkf::cbSubAccV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg)
{
  static bool first_call=true;
  acc_raw_ << msg->vector.x,msg->vector.y,msg->vector.z;
  Vector3d acc = scale2si_acc_* (q_r2acc_* (acccal_trfm_* acc_raw_));


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
      cov_sens_acc_.tryEstCov(acc, first_call);
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
      cov_ctrl_acc_.tryEstCov(acc, first_call);
  }
  else
    cov_ctrl_acc_.updateCov();//It updates everything based on what is the cov provider(either dyn, est or msg)

  if(!x0_.readyRAndCov() && cov_sens_mag_.msg_recvd_)
    x0_.tryComputeRAndCov(acc, sens_acc_.a0,mag_, sens_mag_.m0, first_call);

  if(!x0_.readyBaAndCov() )
    x0_.tryComputeBaAndCov(acc,first_call);

  acc_buffer_.push_back(AccNode(msg->header.stamp,acc));

  if(first_call)
    first_call=false;
}

void
CallBackInsEkf::cbSubGyrV3S(const geometry_msgs::Vector3Stamped::ConstPtr& msg)
{
  static bool first_call=true;
  static ros::Time t_epoch_prev;
  t_ep_x_ = msg->header.stamp;
  gyr_raw_ << msg->vector.x, msg->vector.y, msg->vector.z;
  gyr_ = scale2si_gyr_ * (q_r2gyr_ * gyr_raw_);

  //Find a matching acc reading for the gyro reading
  deque<AccNode>::iterator it_an_lb = lower_bound(acc_buffer_.begin(), acc_buffer_.end(),msg->header.stamp);
  if(it_an_lb==acc_buffer_.end())
  {
    acc_ = (it_an_lb-1)->a_;
    acc_buffer_.erase(acc_buffer_.begin(), (acc_buffer_.end()-1));
  }
  else if(it_an_lb==acc_buffer_.begin())
    acc_ = it_an_lb->a_;
  else
  {
    double dtl = abs((t_ep_x_ - (it_an_lb-1)->t_ep_).toSec());
    double dtr = abs((it_an_lb->t_ep_ - t_ep_x_).toSec());
    acc_ = dtl<dtr?(it_an_lb-1)->a_:it_an_lb->a_;
    acc_buffer_.erase(acc_buffer_.begin(), (it_an_lb-1));
  }

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
       t_epoch_start_= msg->header.stamp;
       t_epoch_prev = t_epoch_start_;
       fr_.is_filtering_on_=true;

       x_=x0_.state();
       filt_nodes_.push_back(FilterNode(0, x_,acc_, cov_ctrl_acc_.cov().diagonal(), gyr_, cov_ctrl_gyr_.cov().diagonal()));
       filt_nodes_.back().addSensAcc(acc_, cov_ctrl_acc_.cov().diagonal());


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
       cout <<"sens_pos_.R:\n"<< sens_pos_.R <<endl;
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

       //keep the size of the buffer at n_max_scs_
       while(filt_nodes_.size()>n_max_scs_-1)
         filt_nodes_.pop_front();

       double t  = (msg->header.stamp - t_epoch_start_).toSec();
       double dt = (msg->header.stamp -   t_epoch_prev).toSec();
       t_epoch_prev = msg->header.stamp;
       if(config_.dyn_debug_on)
       {
         cout << "****************\n";
         cout << "t:"<<t<< "\tdt:"<<dt <<endl;
       }
       //Enter a new FilterNode to filt_nodes for the current time step
       filt_nodes_.push_back(FilterNode(t, x0_.state(),acc_, cov_ctrl_acc_.cov().diagonal(), gyr_, cov_ctrl_gyr_.cov().diagonal()));

       //Add acc_sens reading to the latest filter node
       filt_nodes_.back().addSensAcc(acc_, cov_ctrl_acc_.cov().diagonal());

       //sort all the sensor readings which have not been added to FilterNode and add them at appropriate places
       sort(sens_nodes_.begin(), sens_nodes_.end());

       //add the sorted sensor reading at the right node in filt_nodes_ or just reject it if it's too old
       deque<FilterNode>::iterator plb = filt_nodes_.begin(); //prev lower bound
       deque<FilterNode>::iterator flb = filt_nodes_.end()-1; //first lower bound
       bool first_valid_update=true;
       deque<SensNode>::iterator it_sn;
       for(it_sn = sens_nodes_.begin(); it_sn!=sens_nodes_.end();it_sn++)
       {
         if(it_sn->t_>filt_nodes_.back().t_)
           break;

         deque<FilterNode>::iterator lb = lower_bound(plb,filt_nodes_.end(), *it_sn);
         if(lb == filt_nodes_.begin()) //reject a reading because it's old
           continue;
         else                        //add a reading to the right filter_node
         {
           if(first_valid_update)
           {
             flb = lb;
             first_valid_update = false;
           }
           plb = lb;
           switch(it_sn->type_)
           {
             case SensorType::GPS:
               lb->addSensPos(*(it_sn->val_),*(it_sn->var_));
               break;
             case SensorType::MAG:
               lb->addSensMag(*(it_sn->val_),*(it_sn->var_));
               break;
             case SensorType::POSE3D:
               lb->addSensPose(*(it_sn->pose_),*(it_sn->pose_var_));
               break;
           }
         }
       }
       //delete the sensor measurements which were either added or rejected
       sens_nodes_.erase(sens_nodes_.begin(),it_sn);

       //predict and update from the first update node to the last node
       for(deque<FilterNode>::iterator it=flb; it!=filt_nodes_.end(); it++)
       {
         //Perform prediction from prev state( (it-1)->x_) to current state
         Vector6d u; u <<(it-1)->ctrl_w_, (it-1)->ctrl_a_;
         ins_.sra = sqrt((it-1)->ctrl_a_var_(0));
         ins_.sv  = sqrt((it-1)->ctrl_w_var_(0));
         kp_ins_.Predict(it->x_, (it-1)->t_, (it-1)->x_, u, dt);

         //perform update for all the sensors
         if(it->sens_a_)
         {
           InsState xa = it->x_;
           sens_acc_.R.setZero();sens_acc_.R.diagonal()= *(it->sens_a_var_);
           if(abs(it->sens_a_->norm() - sens_acc_.a0.norm()) < a0_tol_)
             kc_insimu_.Correct(it->x_, it->t_, xa, u, *(it->sens_a_));
         }
         if(it->sens_mag_)
         {
           InsState xa = it->x_;
           sens_mag_.R.setZero();sens_mag_.R.diagonal()= *(it->sens_mag_var_);
           kc_insmag_.Correct(it->x_, it->t_, xa, u, *(it->sens_mag_));
         }
         if(it->sens_pos_)
         {
           InsState xa = it->x_;
           sens_pos_.R.setZero();sens_pos_.R.diagonal()= *(it->sens_pos_var_);
           kc_insgps_.Correct(it->x_, it->t_, x_, u, *(it->sens_pos_));
         }
         if(it->sens_pose_)
         {
           cout<<"sens_pose_ update unimplemented"<<endl;
           assert(0);
         }
       }
       x_ = filt_nodes_.back().x_;
       t_ep_x_ = msg->header.stamp;

       updateOdomAndDiagMsg();
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

  type_sensor_msg_   = yaml_node_["type_sensor_msg"].as<int>();

  strtop_gps_        = yaml_node_["strtop_gps"].as<string>();
  strtop_imu_        = yaml_node_["strtop_imu"].as<string>();
  strtop_mag_        = yaml_node_["strtop_mag"].as<string>();
  strtop_gyr_v3s_    = yaml_node_["strtop_gyr_v3s"].as<string>();
  strtop_acc_v3s_    = yaml_node_["strtop_acc_v3s"].as<string>();
  strtop_mag_v3s_    = yaml_node_["strtop_mag_v3s"].as<string>();

  strtop_odom_       = yaml_node_["strtop_odom"].as<string>();
  strtop_diag_       = yaml_node_["strtop_diag"].as<string>();
  strtop_marker_cov_ = yaml_node_["strtop_marker_cov"].as<string>();

  strfrm_map_        = yaml_node_["strfrm_map"].as<string>();
  strfrm_robot_      = yaml_node_["strfrm_robot"].as<string>();
  strfrm_gps_lcl_    = yaml_node_["strfrm_gps_lcl"].as<string>();

  if(config_.dyn_debug_on)
  {
    cout<<"Topics are:"<<endl;
    if(type_sensor_msg_==1)
    {
      cout<<"strtop_gps:"<<strtop_gps_<<endl;
      cout<<"strtop_imu:"<<strtop_imu_<<endl;
      cout<<"strtop_mag:"<<strtop_mag_<<endl;
    }
    else if(type_sensor_msg_==0)
    {
      cout<<"strtop_gyr_v3s:"<<strtop_gyr_v3s_<<endl;
      cout<<"strtop_acc_v3s:"<<strtop_acc_v3s_<<endl;
      cout<<"strtop_mag_v3s:"<<strtop_mag_v3s_<<endl;
    }
    cout<<"strtop_odom:"<<strtop_odom_<<endl;
    cout<<"strtop_diag:"<<strtop_diag_<<endl;
    cout<<"strtop_marker_cov:"<<strtop_marker_cov_<<endl;
  }
}

void
CallBackInsEkf::initSubsPubsAndTimers(void)
{
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

  //Publishers and Timers

  hz_tf_ = yaml_node_["hz_tf"].as<double>();
  if(hz_tf_>=0)
    pub_viz_cov_ = nh_.advertise<visualization_msgs::Marker>( strtop_marker_cov_, 0 );
  if(hz_tf_>0)
  {
    timer_send_tf_   =  nh_.createTimer(ros::Duration(1.0/hz_tf_), &CallBackInsEkf::cbTimerPubTFCov, this);
    timer_send_tf_.start();
  }

  hz_odom_ = yaml_node_["hz_odom"].as<double>();
  if(hz_odom_>=0)
    pub_odom_ = nh_.advertise<nav_msgs::Odometry>( strtop_odom_, 0 );
  if(hz_odom_>0)
  {
    timer_send_odom_   =  nh_.createTimer(ros::Duration(1.0/hz_odom_), &CallBackInsEkf::cbTimerPubOdom, this);
    timer_send_odom_.start();
  }

  hz_diag_ = yaml_node_["hz_diag"].as<double>();
  if(hz_diag_>=0)
    pub_diag_ = nh_.advertise<gcop_ros_est::InsekfDiag>( strtop_diag_, 0 );
  if(hz_diag_>0)
  {
    timer_send_diag_   =  nh_.createTimer(ros::Duration(1.0/hz_diag_), &CallBackInsEkf::cbTimerPubDiag, this);
    timer_send_diag_.start();
  }
}

void
CallBackInsEkf::loadYamlParams(void)
{
  pause_getchar_ = yaml_node_["pause"].as<bool>();

  x0_.n_avg_ = yaml_node_["n_avg"].as<int>();
  cov_sens_mag_.n_avg_ = x0_.n_avg_;
  cov_sens_acc_.n_avg_ = x0_.n_avg_;
  cov_ctrl_gyr_.n_avg_ = x0_.n_avg_;
  cov_sens_pos_.n_avg_ = x0_.n_avg_;

  //InsState initialization
  Matrix3d rot;
  Vector3d vars,val;
  string type_rot, type_val, type_vars;
  pair<string,Matrix3d> str_mat3d_val;
  pair<string,Vector3d> str_vec3d_vars, str_vec3d_val;
  str_mat3d_val = yaml_node_["x0_R"].as<pair<string,Matrix3d>>();
  str_vec3d_vars = yaml_node_["x0_R_cov"].as<pair<string,Vector3d>>();
  x0_.initRAndCov(str_mat3d_val,str_vec3d_vars);

  str_vec3d_val = yaml_node_["x0_bg"].as<pair<string,Vector3d>>();
  str_vec3d_vars = yaml_node_["x0_bg_cov"].as<pair<string,Vector3d>>();
  x0_.initBgAndCov(str_vec3d_val,str_vec3d_vars);

  str_vec3d_val = yaml_node_["x0_ba"].as<pair<string,Vector3d>>();
  str_vec3d_vars = yaml_node_["x0_ba_cov"].as<pair<string,Vector3d>>();
  x0_.initBaAndCov(str_vec3d_val,str_vec3d_vars);

  str_vec3d_val = yaml_node_["x0_p"].as<pair<string,Vector3d>>();
  str_vec3d_vars = yaml_node_["x0_p_cov"].as<pair<string,Vector3d>>();
  x0_.initpAndCov(str_vec3d_val,str_vec3d_vars);

  str_vec3d_val = yaml_node_["x0_v"].as<pair<string,Vector3d>>();
  str_vec3d_vars = yaml_node_["x0_v_cov"].as<pair<string,Vector3d>>();
  x0_.initvAndCov(str_vec3d_val,str_vec3d_vars);

  //Set reference
  //a0: accelerometer, m0:magnetometer, map0_:gps(lat0(deg), lon0(deg), alt0(m)
  a0_tol_ = yaml_node_["a0_tol"].as<double>();
  Vector3d a0 = yaml_node_["a0"].as<Vector3d>();
  Vector3d m0 = yaml_node_["m0"].as<Vector3d>();
  ins_.g0         = a0;
  sens_acc_.a0 = a0;
  sens_acc_.m0 = m0.normalized();
  sens_mag_.m0 = m0.normalized();
  map0_ = yaml_node_["map0"].as<Vector3d>();


  //set robot to sensors rotation
  Vector7d tfm_r2gyr = yaml_node_["robot2gyr"].as<Vector7d>();
  Vector7d tfm_r2acc = yaml_node_["robot2acc"].as<Vector7d>();;
  q_r2gyr_ = Quaternion<double>(tfm_r2gyr(6),tfm_r2gyr(3),tfm_r2gyr(4),tfm_r2gyr(5));
  q_r2acc_ = Quaternion<double>(tfm_r2acc(6),tfm_r2acc(3),tfm_r2acc(4),tfm_r2acc(5));

  //set scale2si for mag, gyr, acc
  scale2si_gyr_ = yaml_node_["scale2si_gyr"].as<double>();
  scale2si_acc_ = yaml_node_["scale2si_acc"].as<double>();

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
  cov_sens_pos_.setPointersOfCov(&sens_pos_.R);

  //MagCalib setup
  initMagCalib();
}

void
CallBackInsEkf::initMagCalib(void)
{
  magcal_trfm_.linear()      = yaml_node_["magcal_linear"].as<Matrix3d>();
  magcal_trfm_.translation() = yaml_node_["magcal_translation"].as<Vector3d>();

  acccal_trfm_.linear()      = yaml_node_["acccal_linear"].as<Matrix3d>();
  acccal_trfm_.translation() = yaml_node_["acccal_translation"].as<Vector3d>();
}


void
CallBackInsEkf::setFromParamsConfig()
{
  cout<<"*Loading params from yaml file to dynamic reconfigure"<<endl;

  config_.dyn_debug_on = yaml_node_["debug_on"].as<bool>();
  config_.dyn_mag_on = yaml_node_["mag_on"].as<bool>();
  config_.dyn_gps_on = yaml_node_["gps_on"].as<bool>();

  //sensor cov
  pair<string,Vector3d> type_n_vars;
  type_n_vars = yaml_node_["cov_sens_mag"].as<pair<string,Vector3d>>();
  cov_sens_mag_.initCov(type_n_vars);
  type_n_vars = yaml_node_["cov_sens_acc"].as<pair<string,Vector3d>>();
  cov_sens_acc_.initCov(type_n_vars);
  type_n_vars = yaml_node_["cov_sens_gps"].as<pair<string,Vector3d>>();
  cov_sens_pos_.initCov(type_n_vars);

  //ctrl cov
  type_n_vars = yaml_node_["cov_ctrl_gyr"].as<pair<string,Vector3d>>();
  cov_ctrl_gyr_.initCov(type_n_vars);
  type_n_vars = yaml_node_["cov_ctrl_acc"].as<pair<string,Vector3d>>();
  cov_ctrl_acc_.initCov(type_n_vars);
  type_n_vars = yaml_node_["cov_ctrl_su"].as<pair<string,Vector3d>>();
  cov_ctrl_su_.initCov(type_n_vars);
  type_n_vars = yaml_node_["cov_ctrl_sa"].as<pair<string,Vector3d>>();
  cov_ctrl_sa_.initCov(type_n_vars);

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
        cov_sens_pos_.updateCov();
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









