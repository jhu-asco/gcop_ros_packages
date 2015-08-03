/*
 * gcar_ddp_planner.cpp
 *
 *  Created on: Jul 13, 2015
 *      Author: subhransu mishra
 */

#include <ros/ros.h>
#include <ros/package.h>
#include <tf_conversions/tf_eigen.h>

// ROS standard messages
#include <std_msgs/String.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>

//gcop_comm msgs
#include <gcop_comm/State.h>
#include <gcop_comm/CtrlTraj.h>
#include <gcop_comm/Trajectory_req.h>

//ROS dynamic reconfigure
#include <dynamic_reconfigure/server.h>
#include <gcop_ctrl/GcarDDPConfig.h>

//GCOP includes
#include <gcop/lqcost.h>
#include <gcop/gcar.h>
#include <gcop/utils.h>
#include <gcop/se2.h>
#include <gcop/ddp.h>

//Other includes
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>

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
//-----------------------------DDP Params class ----------------------------
//------------------------------------------------------------------------



//------------------------------------------------------------------------
//-----------------------------CALLBACK CLASS ----------------------------
//------------------------------------------------------------------------

class CallBackGcarDDP
{
public:
  typedef Matrix<double, 4, 4> Matrix4d;
  typedef Matrix<double, 4, 1> Vector4d;
  typedef LqCost< M3V1d, 4, 2> GcarLqCost;
  typedef Ddp<M3V1d, 4, 2>     GcarDdp;

  struct DDPParams
  {
    M3V1d  x0;
    M3V1d  xf;
    double tf;
    int    n;
    int    nit;
    int    it; //iteration number
    double mu;

    Matrix4d Q;
    Matrix4d Qf;
    Matrix2d R;
  };
private:

public:
  CallBackGcarDDP();
  ~CallBackGcarDDP();

private:

  void cbReconfig(gcop_ctrl::GcarDDPConfig &config, uint32_t level);
  void setDDPParamsReconfig(void);
  void cbTimerGeneral(const ros::TimerEvent& event);
  void setDDPParamsRequest(gcop_comm::Trajectory_req::Request &req);
  void setTrajResp(gcop_comm::Trajectory_req::Response &resp);
  bool cbSrvDDP(gcop_comm::Trajectory_req::Request &req, gcop_comm::Trajectory_req::Response &resp);
  void initTraj(void);
  void initCtrl(void);
  void initDDP(void);

  void coutCost(void);
  void coutResults(void);

  void rvizInitMarker(void);
  void rvizEndMarker(void);
  void rvizDispPath(void);

private:
  //ROS relavant members
  ros::NodeHandle nh_;
  ros::ServiceServer srvsvr_traj_;
  gcop_ctrl::GcarDDPConfig config_;
  dynamic_reconfigure::Server<gcop_ctrl::GcarDDPConfig> dyn_server_;
  bool dyn_write_;      //save config_ to config
  ros::Timer timer_general_;
  std::string strsrv_traj;

  ros::Publisher pub_rviz_;
  std::string strtop_rviz_lines_;

  //rviz relavant variables
  visualization_msgs::Marker marker_path_;

  //trajectory message
  gcop_comm::CtrlTraj traj_;

  //ddp relevant members
  DDPParams        ddp_pars_;           //All temporary params
  M3V1d            ddp_xf_;             //final state
  vector<double>   ddp_ts_;             //All times
  vector<M3V1d>    ddp_xs_;             //All states
  vector<Vector2d> ddp_us_;             //All controls
  Gcar             ddp_sys_;            //The gcar system
  GcarLqCost*      ddp_cost_;           //Lq cost for Gcar
  GcarDdp*         ddp_algo_;           //The ddp algorithm

};

CallBackGcarDDP::CallBackGcarDDP():
    strsrv_traj("req_ddp_lcl"),
    strtop_rviz_lines_("marker_traj"),
    ddp_cost_(nullptr),
    ddp_algo_(nullptr),
    dyn_write_(false)
{
  cout<<"*Entering constructor of cbc"<<endl;

  //setup dynamic reconfigure
  dynamic_reconfigure::Server<gcop_ctrl::GcarDDPConfig>::CallbackType dyn_cb_f;
  dyn_cb_f = boost::bind(&CallBackGcarDDP::cbReconfig, this, _1, _2);
  dyn_server_.setCallback(dyn_cb_f);

  //Setup timers
  timer_general_ = nh_.createTimer(ros::Duration(0.05), &CallBackGcarDDP::cbTimerGeneral, this);
  timer_general_.start();

  //Setup Publishers
  if(config_.debug_on_all)
  {
    cout<<"  Setting up ros publishers"<<endl;
  }
  pub_rviz_     = nh_.advertise<visualization_msgs::Marker>( strtop_rviz_lines_, 0 );

  //Setup Subscriber

  //Setup service:
  if(config_.debug_on_all)
  {
    cout<<"  Setting up service server"<<endl;
  }
  srvsvr_traj_ = nh_.advertiseService(strsrv_traj,&CallBackGcarDDP::cbSrvDDP, this);

  //initialize path markers
  if(config_.debug_on_all)
  {
    cout<<"  Initializing all rviz markers."<<endl;
  }
  rvizInitMarker();

  //Setup ddp stuff
  initDDP();

}

CallBackGcarDDP::~CallBackGcarDDP()
{
  if(config_.debug_on_all)
  {
    cout<<"*Destroying all allocated objects"<<endl;
  }
  rvizEndMarker();
  delete ddp_cost_;           //Lq cost for Gcar
  delete ddp_algo_;
}



void
CallBackGcarDDP::cbReconfig(gcop_ctrl::GcarDDPConfig &config, uint32_t level)
{
  static bool first_time=true;
  config_ = config;

  if(config_.debug_on_all)
  {
    cout<<"*Entering reconfigure callback"<<endl;
  }

  if(first_time)
  {
    setDDPParamsReconfig();
    first_time = false;
  }
  if(config.iterate)
  {
    ddp_algo_->Iterate();
    ddp_pars_.it++;
    if(config.debug_on_all)
    {
      coutResults();
    }
    config.iterate = false;
  }

  if(config.reinit)
  {
    setDDPParamsReconfig();
    initDDP();
    config.reinit = false;
  }
}

void
CallBackGcarDDP::cbTimerGeneral(const ros::TimerEvent& event)
{
  if(dyn_write_)
  {
    dyn_server_.updateConfig(config_);
    dyn_write_=false;
  }
}
void
CallBackGcarDDP::setDDPParamsReconfig(void)
{
  if(config_.debug_on_all)
  {
    cout<<"*Setting ddp params from reconfigure"<<endl;
  }

  ddp_pars_.it = 0;     //Reset iteration number back to 0

  ddp_pars_.tf  = config_.tf;
  ddp_pars_.n = config_.n;
  ddp_pars_.nit = config_.nit;
  ddp_pars_.mu  = config_.mu;

  //set state params
  Vector3d q0(config_.th0, config_.x0, config_.y0);
  SE2::Instance().q2g( ddp_pars_.x0.first, q0);
  ddp_pars_.x0.second = config_.v0;

  Vector3d qf(config_.thf, config_.xf, config_.yf);
  SE2::Instance().q2g( ddp_pars_.xf.first, qf);
  ddp_pars_.xf.second = config_.vf;

  //set cost params
  ddp_pars_.Q.setZero();
  ddp_pars_.Qf.setZero();
  ddp_pars_.Qf(0,0) = config_.qf00;
  ddp_pars_.Qf(1,1) = config_.qf11;
  ddp_pars_.Qf(2,2) = config_.qf22;
  ddp_pars_.Qf(3,3) = config_.qf33;

  ddp_pars_.R.setZero();
  ddp_pars_.R(0,0)  = config_.r00;
  ddp_pars_.R(1,1)  = config_.r11;

  if(config_.debug_on_all)
  {
    cout<<" The params are set to the following value"<<endl;
    cout<<"nit:"<<ddp_pars_.nit<<endl;
    cout<<"tf:"<<ddp_pars_.tf<<endl;
    cout<<"x0.first:\n"<<ddp_pars_.x0.first<<endl;
    cout<<"x0.second:\n"<<ddp_pars_.x0.second<<endl;
    cout<<"xf.first:\n"<<ddp_pars_.xf.first<<endl;
    cout<<"x0.second:\n"<<ddp_pars_.x0.second<<endl;
    cout<<"Q:\n"<<ddp_pars_.Q<<endl;
    cout<<"Qf:\n"<<ddp_pars_.Qf<<endl;
    cout<<"R:\n"<<ddp_pars_.R<<endl;
  }
}

void
CallBackGcarDDP::setDDPParamsRequest(gcop_comm::Trajectory_req::Request &req)
{
  if(config_.debug_on_all)
  {
    cout<<"*Setting config_ from request"<<endl;
  }

  //Set back the config
  config_.tf = req.itreq.tf - req.itreq.t0;
  config_.n = req.itreq.N;
  config_.nit = req.itreq.Niter;

  config_.x0  = req.itreq.x0.statevector[0];
  config_.y0  = req.itreq.x0.statevector[1];
  config_.th0 = req.itreq.x0.statevector[2];
  config_.v0  = req.itreq.x0.statevector[3];

  config_.xf  = req.itreq.xf.statevector[0];
  config_.yf  = req.itreq.xf.statevector[1];
  config_.thf = req.itreq.xf.statevector[2];
  config_.vf  = req.itreq.xf.statevector[3];

  dyn_write_ = true;
  setDDPParamsReconfig();
}

void
CallBackGcarDDP::setTrajResp(gcop_comm::Trajectory_req::Response &resp)
{
  if(config_.debug_on_all)
    cout<<"*Setting Traj response message"<<endl;

  if(traj_.N != ddp_pars_.n)
    initTraj();

  //Set xs
  for (int i = 0;i<ddp_pars_.n+1;i++)
  {
    double cth = ddp_xs_[i].first(0,0);
    double sth = ddp_xs_[i].first(1,0);
    double th = atan2(sth,cth);
    traj_.statemsg[i].statevector.resize(4);
    traj_.statemsg[i].statevector[0] = ddp_xs_[i].first(0,2);
    traj_.statemsg[i].statevector[1] = ddp_xs_[i].first(1,2);
    traj_.statemsg[i].statevector[2] = th;
    traj_.statemsg[i].statevector[3] = ddp_xs_[i].second;
  }

  //Set us
  for (int i = 0;i<ddp_pars_.n;i++)
  {
    traj_.ctrl[i].ctrlvec.resize(2);
    traj_.ctrl[i].ctrlvec[0] = ddp_us_[i](0);
    traj_.ctrl[i].ctrlvec[1] = ddp_us_[i](1);
  }
  traj_.time = ddp_ts_;

  resp.traj = traj_;
}

bool
CallBackGcarDDP::cbSrvDDP(gcop_comm::Trajectory_req::Request &req, gcop_comm::Trajectory_req::Response &resp)
{
  if(config_.debug_on_all)
    cout<<"*New service request received"<<endl;

  setDDPParamsRequest(req);
  initDDP();
  for(int i=0;i< ddp_pars_.nit;i++)
    ddp_algo_->Iterate();
  if(config_.debug_on_all)
    coutResults();
  setTrajResp(resp);

  if(config_.debug_on_all)
    cout<<"  about to return service request"<<endl;
}

void
CallBackGcarDDP::initTraj(void)
{
  traj_.N = ddp_pars_.n;
  traj_.statemsg.resize(ddp_pars_.n+1);
  traj_.ctrl.resize(ddp_pars_.n);
}

void
CallBackGcarDDP::initCtrl(void)
{
  if(config_.debug_on_all)
    std::cout<<"*initializing controls"<<endl;

  double tf = config_.tf;
  double thf = config_.thf;
  double th0 = config_.th0;
  double dx = config_.xf - config_.x0;
  double dy = config_.yf - config_.y0;
  double dth = atan2(sin(thf-th0), cos(thf-th0));
  double dist = sqrt(dx*dx + dy*dy);
  double u = config_.v0;
  double v = config_.vf;
  double acc = (4*dist - 2*(u+v)*tf)/(tf*tf);
  double torq = acc/ddp_sys_.r;
  int N  = config_.n;
  double tanangl = dth*ddp_sys_.l/(u*tf + acc*tf*tf/4);
  for (int i = 0; i < N/2; ++i) {
    ddp_us_[i] = Vector2d(torq, tanangl);
    ddp_us_[N/2+i] = Vector2d(-torq, tanangl);
  }
  if(config_.debug_on_all)
  {
    std::cout<<"  tf:"<<tf<<endl;
    std::cout<<"  dist:"<<dist<<std::endl;
    std::cout<<"  acc:"<<acc<<std::endl;
    std::cout<<"  u:"<<u<<std::endl;
    std::cout<<"  v:"<<v<<std::endl;
    std::cout<<"  th0:"<<th0<<std::endl;
    std::cout<<"  thf:"<<thf<<std::endl;
    std::cout<<"  dth:"<<dth<<std::endl;
    std::cout<<"  tanangl:"<<tanangl<<std::endl;
  }

}

void
CallBackGcarDDP::initDDP()
{
  if(config_.debug_on_all)
  {
    cout<<"*Initializing DDP"<<endl;
  }

  if(ddp_us_.size()!=ddp_pars_.n)
  {
    ddp_ts_.resize(ddp_pars_.n+1);
    ddp_xs_.resize(ddp_pars_.n+1);
    ddp_us_.resize(ddp_pars_.n);
  }
  //Set xs[0]
  ddp_xs_[0] = ddp_pars_.x0;

  //Set xf
  ddp_xf_ = ddp_pars_.xf;

  //Set ts
  double h = ddp_pars_.tf/ddp_pars_.n;
  for (int k = 0; k <=ddp_pars_.n; ++k)
    ddp_ts_[k] = k*h;

  //Set us
  //initCtrl();
  for (int i = 0; i < ddp_pars_.n/2; ++i)
  {
    ddp_us_[i] = Vector2d(0.01,.1);
    ddp_us_[ddp_pars_.n/2+i] = Vector2d(0.01,-.1);
  }

  //Set cost
  if(ddp_cost_==nullptr)
    ddp_cost_ = new GcarLqCost(ddp_sys_,ddp_pars_.tf, ddp_xf_);
  ddp_cost_->Q = ddp_pars_.Q;
  ddp_cost_->Qf = ddp_pars_.Qf;
  ddp_cost_->R = ddp_pars_.R;
  ddp_cost_->UpdateGains();

  //Set ddp algo
  if(ddp_algo_==nullptr)
    ddp_algo_ = new GcarDdp(ddp_sys_, *ddp_cost_, ddp_ts_, ddp_xs_, ddp_us_);
  else
    ddp_algo_->Update();
  ddp_algo_->debug = false; // turn off debug for speed
  ddp_algo_->mu = ddp_pars_.mu;
}

void
CallBackGcarDDP::coutCost()
{
  cout<<"*Cost params are:"<<endl;
  cout<<"Q:\n"<<ddp_cost_->Q<<endl;
  cout<<"Qf:\n"<<ddp_cost_->Qf<<endl;
  cout<<"R:\n"<<ddp_cost_->R<<endl;
}
void
CallBackGcarDDP::coutResults()
{
  double cth,sth,th;
  //cout time position at each node
  // and control inbetween
  cout<<"*iteration number:"<<ddp_pars_.it<<endl;
  for (int i = 0; i < ddp_xs_.size(); i++)
  {
    cth = ddp_xs_[i].first(0,0);
    sth = ddp_xs_[i].first(1,0);
    th = atan2(sth,cth);
    cout<<"Node "<<i<<": x:"<< ddp_xs_[i].first(0,2)<<"  y:"<<ddp_xs_[i].first(1,2)<<" th:"<< th<< endl;
  }
  //cout initial position
  cth = ddp_xf_.first(0,0);
  sth = ddp_xf_.first(1,0);
  th = atan2(sth,cth);
  cout<<"Node xf"<<": x:"<< ddp_xf_.first(0,2)<<"  y:"<<ddp_xf_.first(1,2)<<" th:"<< th<< endl;

}

void
CallBackGcarDDP::rvizInitMarker(void)
{
  int id=-1;
  //Marker for path
  id++;
  marker_path_.header.frame_id = "map";
  marker_path_.header.stamp = ros::Time();
  marker_path_.ns = "rampage";
  marker_path_.id = id;
  marker_path_.type = visualization_msgs::Marker::LINE_STRIP;
  marker_path_.action = visualization_msgs::Marker::ADD;
  marker_path_.scale.x = 0.4;
  marker_path_.color.a = 0.5; // Don't forget to set the alpha!
  marker_path_.color.r = 0.0;
  marker_path_.color.g = 1.0;
  marker_path_.color.b = 0.0;
  marker_path_.lifetime = ros::Duration(0);
}
void
CallBackGcarDDP::rvizEndMarker(void)
{
  marker_path_.action     = visualization_msgs::Marker::DELETE;
  pub_rviz_.publish( marker_path_ );
}
void
CallBackGcarDDP::rvizDispPath(void)
{
  marker_path_.points.clear();
  for (int i = 0; i < ddp_xs_.size(); i++)
  {
    geometry_msgs::Point node;
    node.x = ddp_xs_[i].first(0,2);node.y = ddp_xs_[i].first(1,2);node.z = 0.2;
    marker_path_.points.push_back(node);
  }
  pub_rviz_.publish( marker_path_ );
}

//------------------------------------------------------------------------
//--------------------------------MAIN -----------------------------------
//------------------------------------------------------------------------

int main(int argc, char** argv)
{
  ros::init(argc,argv,"gcar_ddp_planner",ros::init_options::NoSigintHandler);
  signal(SIGINT,mySigIntHandler);

  ros::NodeHandle nh;

  CallBackGcarDDP cbc;

  double a=10;

  ros::Rate loop_rate(100);
  while(!g_shutdown_requested)
  {
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;

}





