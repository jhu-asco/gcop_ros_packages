//dsl_ddp_planner.cpp
//Author: Subhransu Mishra
//Note: This ros node plans a d*lite path and then does ddp planning to next closest
//      waypoint on the d*lite path

//ROS
#include <ros/ros.h>
#include <ros/package.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>

//ROS & opencv
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

// ROS standard messages
#include <std_msgs/String.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>

//gcop_comm msgs
#include <gcop_comm/GcarCtrl.h>

//ROS dynamic reconfigure
#include <dynamic_reconfigure/server.h>
#include <gcop_ctrl/DslDdpPlannerConfig.h>

//Other includes
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

//D* Lite algorithm
#include <dsl/gridsearch.h>

//gcop include
#include <gcop/so3.h>
#include <gcop/lqcost.h>
#include <gcop/gcar.h>
#include <gcop/utils.h>
#include <gcop/se2.h>
#include <gcop/ddp.h>

//yaml
#include <yaml-cpp/yaml.h>

//local includes
#include <gcop_ros_utils/eigen_ros_conv.h>
#include <gcop_ros_utils/eig_splinterp.h>
#include <gcop_ros_utils/yaml_eig_conv.h>

// Eigen Includes
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/Splines>

//-------------------------------------------------------------------------
//-----------------------NAME SPACES ---------------------------------
//-------------------------------------------------------------------------
using namespace std;
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

class CallBackDslDdp
{
public:
  typedef Matrix<float, 4, 4> Matrix4f;
  typedef Matrix<double, 4, 4> Matrix4d;
  typedef Matrix<double, 5, 1> Vector5d;
  typedef Ddp<GcarState, 4, 2> GcarDdp;
  typedef Transform<double,2,Affine> Transform2d;

public:
  CallBackDslDdp();
  ~CallBackDslDdp();
public:
  ros::Rate loop_rate_main_;

private:
  ros::NodeHandle nh_, nh_p_;
  YAML::Node yaml_node_;

  string strtop_odom_, strtop_pose_start_, strtop_pose_goal_, strtop_og_;
  string strtop_diag_, strtop_marker_rviz_, strtop_og_dild_, strtop_ctrl_;
  string strfrm_world_, strfrm_robot_, strfrm_og_org_;

  ros::Subscriber sub_odom_, sub_pose_start_, sub_pose_goal_, sub_og_;
  ros::Publisher pub_diag_, pub_vis_, pub_og_dild_, pub_ctrl_;
  ros::Timer timer_vis_, timer_ddp_, timer_dsl_;

  tf::TransformListener tf_lr_;

  gcop_ctrl::DslDdpPlannerConfig config_;
  dynamic_reconfigure::Server<gcop_ctrl::DslDdpPlannerConfig> dyn_server_;
  visualization_msgs::Marker marker_path_dsl_,marker_wp_dsl_;
  visualization_msgs::Marker marker_path_dsl_intp_ ,marker_wp_dsl_intp_;
  VectorXd prop_path_pve_ddp_, prop_path_nve_ddp_, prop_wp_pve_ddp_, prop_wp_nve_ddp_ ;
  visualization_msgs::Marker marker_path_ddp_,marker_wp_ddp_;
  visualization_msgs::Marker marker_text_start_, marker_text_goal_;
  nav_msgs::OccupancyGrid occ_grid_;
  cv::Mat img_occ_grid_dilated_;

  // Frames and transformation
  Affine3d tfm_world2og_org_, tfm_world2og_ll_,tfm_og_org2og_ll_;
  Transform2d tfm_world2og_ll_2d_;



  // DSL vars
  dsl::GridSearch* p_gdsl_;
  double* map_dsl_;
  dsl::GridPath path_opt_;
  dsl::GridCost grid_cost_;
  bool dsl_cond_feas_s_, dsl_cond_feas_g_, dsl_done_;
  VectorXd pt_x_dsl_intp_, pt_y_dsl_intp_,a_dsl_intp_, t_dsl_intp_;//x,y,angle,t of the path
  Affine3d pose_dsl_start_, pose_dsl_goal_;
  Vector3d pt_dsl_start_, pt_dsl_goal_;//refers to grid point

  // DDP vars
  bool ddp_debug_on_;
  double ddp_mu_;
  int ddp_nseg_max_,ddp_nseg_min_, ddp_nit_max_;
  double ddp_tseg_ideal_;
  double ddp_tol_rel_, ddp_tol_abs_;
  Affine3d pose_ddp_start_, pose_ddp_goal_, pose_ddp_curr_;
  ros::Time time_ddp_start_, time_ddp_curr_;
  double vel_ddp_start_,     vel_ddp_goal_,  vel_ddp_curr_;
  Vector3d pt_ddp_start_, pt_ddp_goal_; //refers to grid point

  Gcar sys_gcar_;
  LqCost< GcarState, 4, 2> cost_lq_;
  vector<double> ddp_ts_;
  vector<GcarState> ddp_xs_;
  vector<Vector2d> ddp_us_;

private:

  void cbReconfig(gcop_ctrl::DslDdpPlannerConfig &config, uint32_t level);

  void cbOdom(const nav_msgs::OdometryConstPtr& msg_odom);
  void cbPoseStart(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg_pose_start);
  void cbPoseGoal(const geometry_msgs::PoseStampedConstPtr& msg_pose_goal);
  void cbOccGrid(const nav_msgs::OccupancyGridConstPtr& msg_occ_grid);

  void cbTimerVis(const ros::TimerEvent& event);
  void cbTimerDsl(const ros::TimerEvent& event);
  void cbTimerDdp(const ros::TimerEvent& event);
  void endMarker(void);

  void setupTopicsAndNames(void);
  void initSubsPubsAndTimers(void);
  void initRvizMarkers(void);
  void editColorMsg(std_msgs::ColorRGBA& rgba_msg, VectorXd& rgba_vec);
  void editRvizMarker(visualization_msgs::Marker& marker, VectorXd& prop);


  void dispPathDslRviz(void);
  void dispPathDslInterpdRviz();
  void dispPathDdpRviz(void);
  void dispStartRviz(void);
  void dispGoalRviz(void);


  void dslInit(void);
  void dslDelete(void);
  bool dslPlan(void);
  bool dslFeasible(void);
  void dslInterpolate(void);

  bool ddpFeasible(void);
  bool ddpInit(void);
  bool ddpPlan(void);


  void dilateObs(void);
  void setTfms(void);
  void imgToOccGrid(const cv::Mat& img,const geometry_msgs::Pose pose_org, const double res_m_per_pix,  nav_msgs::OccupancyGrid& occ_grid);

};

CallBackDslDdp::CallBackDslDdp():
                        nh_p_("~"),
                        loop_rate_main_(1000),
                        dsl_cond_feas_s_(false),
                        dsl_cond_feas_g_(false),
                        dsl_done_(false),
                        p_gdsl_(nullptr),
                        map_dsl_(nullptr),
                        sys_gcar_(),
                        cost_lq_(sys_gcar_, 1, GcarState(Matrix3d::Identity(), 0))
{
  cout<<"**************************************************************************"<<endl;
  cout<<"***************************DSL-DDP TRAJECTORY PLANNER*********************"<<endl;
  cout<<"*Entering constructor of cbc"<<endl;

  //Setup YAML reading and parsing
  string strfile_params;nh_p_.getParam("strfile_params",strfile_params);
  cout<<"loading yaml param file into yaml_node"<<endl;
  yaml_node_ = YAML::LoadFile(strfile_params);

  //setup dynamic reconfigure
  dynamic_reconfigure::Server<gcop_ctrl::DslDdpPlannerConfig>::CallbackType dyn_cb_f;
  dyn_cb_f = boost::bind(&CallBackDslDdp::cbReconfig, this, _1, _2);
  dyn_server_.setCallback(dyn_cb_f);

  // Setup topic names
  setupTopicsAndNames();
  cout<<"Setup topic names from yaml file done"<<endl;

  //Setup Subscriber, publishers and Timers
  initSubsPubsAndTimers();
  cout<<"Initialized publishers, subscriber and timers"<<endl;

  //Setup rviz markers
  initRvizMarkers();
  cout<<"Initialized Rviz Markers"<<endl;

  //Set gcar properties
  sys_gcar_.l = yaml_node_["gcar_l"].as<double>();
  sys_gcar_.r = yaml_node_["gcar_r"].as<double>();

  //init ddp planner
  ddpInit();

  cout<<"Waiting for start and goal position.\nSelect start through Publish Point button and select goal through 2D nav goal."<<endl;
}


CallBackDslDdp::~CallBackDslDdp()
{
  endMarker();
  dslDelete();
}

void
CallBackDslDdp::cbReconfig(gcop_ctrl::DslDdpPlannerConfig &config, uint32_t level)
{
  static bool first_time=true;

  //check for all change condition
  bool condn_dilate =     occ_grid_.info.width
      && (   config.dyn_dilation_obs_m != config_.dyn_dilation_obs_m
          || config.dyn_dilation_type != config_.dyn_dilation_type );

  if(!first_time)
  {
    if(condn_dilate)
    {
      config_.dyn_dilation_obs_m = config.dyn_dilation_obs_m;
      config_.dyn_dilation_type = config.dyn_dilation_type;
      dilateObs();
    }

    //loop rate setting
    if(config_.dyn_loop_rate_main != config.dyn_loop_rate_main)
      loop_rate_main_= ros::Rate(config.dyn_loop_rate_main);

    //dsl settings
    if(config.dyn_dsl_plan_once)
    {
      if(dslPlan() && config.dyn_dsl_disp_rviz)
      {
        dispPathDslRviz();
        dispPathDslInterpdRviz();
      }
      config.dyn_dsl_plan_once = false;
    }

    if(config_.dyn_dsl_loop_durn != config.dyn_dsl_loop_durn)
      timer_dsl_.setPeriod(ros::Duration(config.dyn_dsl_loop_durn));

    if(config_.dyn_dsl_plan_loop != config.dyn_dsl_plan_loop && config.dyn_dsl_plan_loop )
      timer_dsl_.start();

    if(config_.dyn_dsl_plan_loop != config.dyn_dsl_plan_loop && !config.dyn_dsl_plan_loop )
      timer_dsl_.stop();

    if(config_.dyn_dsl_avg_speed != config.dyn_dsl_avg_speed)
    {
      config_.dyn_dsl_avg_speed = config.dyn_dsl_avg_speed;
      dslInterpolate();
    }

    //ddp settings
    if(config.dyn_ddp_plan_once)
    {
      if(ddpPlan() && config_.dyn_ddp_disp_rviz)
        dispPathDdpRviz();
      config.dyn_ddp_plan_once = false;
    }

    if(config_.dyn_ddp_loop_durn != config.dyn_ddp_loop_durn)
      timer_ddp_.setPeriod(ros::Duration(config.dyn_ddp_loop_durn));

    if(config_.dyn_ddp_plan_loop != config.dyn_ddp_plan_loop && config.dyn_ddp_plan_loop )
      timer_ddp_.start();

    if(config_.dyn_ddp_plan_loop != config.dyn_ddp_plan_loop && !config.dyn_ddp_plan_loop )
      timer_ddp_.stop();

  }
  else
  {
    cout<<"First time in reconfig. Setting config from yaml"<<endl;

    //general settings
    config.dyn_debug_on           = yaml_node_["debug_on"].as<bool>();
    config.dyn_send_gcar_ctrl      = yaml_node_["send_gcar_ctrl"].as<bool>();
    config.dyn_loop_rate_main     = yaml_node_["loop_rate_main"].as<double>();
    loop_rate_main_= ros::Rate(config.dyn_loop_rate_main);

    //dilation settings
    config.dyn_dilation_type      = yaml_node_["dilation_type"].as<int>();
    config.dyn_dilation_obs_m     = yaml_node_["dilation_obs_m"].as<double>();
    config.dyn_dilation_min_m     = yaml_node_["dilation_min_m"].as<double>();
    config.dyn_dilation_max_m     = yaml_node_["dilation_max_m"].as<double>();

    //dsl settings
    config.dyn_dsl_avg_speed      = yaml_node_["dsl_avg_speed"].as<double>();
    config.dyn_dsl_interp_deg     = yaml_node_["dsl_interp_deg"].as<double>();
    config.dyn_dsl_interp_delt    = yaml_node_["dsl_interp_delt"].as<double>();
    config.dyn_dsl_preint_delt    = yaml_node_["dsl_preint_delt"].as<double>();
    config.dyn_dsl_from_curr_posn = yaml_node_["dsl_from_curr_posn"].as<bool>();
    config.dyn_dsl_loop_durn      = yaml_node_["dsl_loop_durn"].as<double>();
    config.dyn_dsl_plan_once      = false;
    config.dyn_dsl_plan_loop      = false;
    config.dyn_dsl_disp_rviz      = yaml_node_["dsl_disp_rviz"].as<bool>();

    //ddp settings
    config.dyn_ddp_from_curr_posn = yaml_node_["ddp_from_curr_posn"].as<bool>();
    config.dyn_ddp_t_away         = yaml_node_["ddp_t_away"].as<double>();
    config.dyn_ddp_loop_durn      = yaml_node_["ddp_loop_durn"].as<double>();
    config.dyn_ddp_plan_once=false;
    config.dyn_ddp_plan_loop=false;
    config.dyn_ddp_disp_rviz      = yaml_node_["ddp_disp_rviz"].as<bool>();

    first_time = false;
  }
  config_ = config;
}

void
CallBackDslDdp::setupTopicsAndNames(void)
{
  if(config_.dyn_debug_on)
    cout<<"setting up topic names"<<endl;

  // Input Topics
  string strtop_odom_str("strtop_odom");
  string strtop_pose_start_str("strtop_pose_start");
  string strtop_pose_goal_str("strtop_pose_goal");
  string strtop_og_str("strtop_og");

  // output Topics
  string strtop_diag_str("strtop_diag");
  string strtop_marker_rviz_str("strtop_marker_rviz");
  string strtop_og_dild_str("strtop_og_dild");
  string strtop_ctrl_str("strtop_ctrl");

  // Frame Names
  string strfrm_world_str("strfrm_world");
  string strfrm_robot_str("strfrm_robot");
  string strfrm_og_org_str("strfrm_og_org");

  //check and report missing params
  if(!yaml_node_[strtop_odom_str])
    cout<<"missing in yaml file:"<<strtop_odom_str<<endl;
  if(!yaml_node_[strtop_pose_start_str])
    cout<<"missing in yaml file:"<<strtop_pose_start_str<<endl;
  if(!yaml_node_[strtop_pose_goal_str])
    cout<<"missing in yaml file:"<<strtop_pose_goal_str<<endl;
  if(!yaml_node_[strtop_og_str])
    cout<<"missing in yaml file:"<<strtop_og_str<<endl;
  if(!yaml_node_[strtop_diag_str])
    cout<<"missing in yaml file:"<<strtop_diag_str<<endl;
  if(!yaml_node_[strtop_marker_rviz_str])
    cout<<"missing in yaml file:"<<strtop_marker_rviz_str<<endl;
  if(!yaml_node_[strtop_og_dild_str])
    cout<<"missing in yaml file:"<<strtop_og_dild_str<<endl;
  if(!yaml_node_[strtop_ctrl_str])
    cout<<"missing in yaml file:"<<strtop_ctrl_str<<endl;
  if(!yaml_node_[strfrm_world_str])
    cout<<"missing in yaml file:"<<strfrm_world_str<<endl;
  if(!yaml_node_[strfrm_robot_str])
    cout<<"missing in yaml file:"<<strfrm_robot_str<<endl;
  if(!yaml_node_[strfrm_og_org_str])
    cout<<"missing in yaml file:"<<strfrm_og_org_str<<endl;

  // Input topics
  strtop_odom_       = yaml_node_[strtop_odom_str].as<string>();
  strtop_pose_start_ = yaml_node_[strtop_pose_start_str].as<string>();
  strtop_pose_goal_  = yaml_node_[strtop_pose_goal_str].as<string>();
  strtop_og_          = yaml_node_[strtop_og_str].as<string>();

  // output topics
  strtop_diag_        = yaml_node_[strtop_diag_str].as<string>();
  strtop_marker_rviz_ = yaml_node_[strtop_marker_rviz_str].as<string>();
  strtop_og_dild_     = yaml_node_[strtop_og_dild_str].as<string>();
  strtop_ctrl_       = yaml_node_[strtop_ctrl_str].as<string>();

  // Frames
  strfrm_world_        = yaml_node_[strfrm_world_str].as<string>();
  strfrm_robot_      = yaml_node_[strfrm_robot_str].as<string>();
  strfrm_og_org_         = yaml_node_[strfrm_og_org_str].as<string>();

  if(config_.dyn_debug_on)
  {
    cout<<"Topics are(put here):"<<endl;
  }
}

void
CallBackDslDdp::initSubsPubsAndTimers(void)
{
  //Setup subscribers
  sub_odom_       = nh_.subscribe(strtop_odom_,1, &CallBackDslDdp::cbOdom,this);
  sub_og_   = nh_.subscribe(strtop_og_,  1, &CallBackDslDdp::cbOccGrid, this);
  cout<<"Subscribing to strtop_pose_start_:"<<strtop_pose_start_<<endl;
  sub_pose_start_ = nh_.subscribe(strtop_pose_start_, 1, &CallBackDslDdp::cbPoseStart, this);
  sub_pose_goal_  = nh_.subscribe(strtop_pose_goal_, 1, &CallBackDslDdp::cbPoseGoal, this);

  //Setup Publishers
  pub_diag_    = nh_.advertise<visualization_msgs::Marker>( strtop_diag_, 0 );
  pub_vis_     = nh_.advertise<visualization_msgs::Marker>( strtop_marker_rviz_, 0 );
  pub_og_dild_ = nh_.advertise<nav_msgs::OccupancyGrid>(strtop_og_dild_,0,true);
  pub_ctrl_    = nh_.advertise<gcop_comm::GcarCtrl>(strtop_ctrl_,0);

  //Setup timers
  timer_vis_ = nh_.createTimer(ros::Duration(0.1), &CallBackDslDdp::cbTimerVis, this);
  timer_vis_.stop();

  timer_dsl_ = nh_.createTimer(ros::Duration(config_.dyn_dsl_loop_durn), &CallBackDslDdp::cbTimerDsl, this);
  timer_dsl_.stop();

  timer_ddp_ = nh_.createTimer(ros::Duration(config_.dyn_ddp_loop_durn), &CallBackDslDdp::cbTimerDdp, this);
  timer_ddp_.stop();

}

void
CallBackDslDdp::cbTimerVis(const ros::TimerEvent& event)
{
  pub_vis_.publish( marker_path_dsl_ );
}

void
CallBackDslDdp::cbTimerDsl(const ros::TimerEvent& event)
{
  if(dslPlan() && config_.dyn_dsl_disp_rviz)
  {
    dispPathDslRviz();
    dispPathDslInterpdRviz();
  }
}

void
CallBackDslDdp::cbTimerDdp(const ros::TimerEvent& event)
{
  if(ddpPlan() && config_.dyn_ddp_disp_rviz)
    dispPathDdpRviz();
}

void
CallBackDslDdp::cbOdom(const nav_msgs::OdometryConstPtr& msg_odom)
{
  time_ddp_curr_ = msg_odom->header.stamp;
  poseMsg2Eig(pose_ddp_curr_,msg_odom->pose.pose);
  vel_ddp_curr_ = msg_odom->twist.twist.linear.x;
}

void
CallBackDslDdp::cbPoseStart(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg_pose_start)
{
  if(config_.dyn_debug_on)
    cout<<"Initial pose received from rviz"<<endl;

  float m_per_cell = occ_grid_.info.resolution;
  int w = occ_grid_.info.width;
  int h = occ_grid_.info.height;

  poseMsg2Eig(pose_dsl_start_,msg_pose_start->pose.pose);
  pt_dsl_start_ = (tfm_world2og_ll_.inverse()*pose_dsl_start_).translation()/m_per_cell;
  dispStartRviz();
  dsl_cond_feas_s_ = (pt_dsl_start_(0)>=0) && (pt_dsl_start_(1)>=0) && (pt_dsl_start_(0)<w) && (pt_dsl_start_(1)<h);
  if(dsl_cond_feas_s_)
    p_gdsl_->SetStart((int)pt_dsl_start_(0),(int)pt_dsl_start_(1));

  if(dsl_cond_feas_s_ && config_.dyn_debug_on)
  {
    cout<<"*Start position received at ("
        <<pose_dsl_start_.translation().transpose() <<") in world frame and ("
        <<pt_dsl_start_.transpose()                 <<") in OG coordinate"<<endl;
    cout<<"pose_dsl_start rotation:\n"<<pose_dsl_start_.rotation()<<endl;
  }

  if(!dsl_cond_feas_s_ && config_.dyn_debug_on)
  {
    cout<<"*Invalid Start position received at ("
        <<pose_dsl_start_.translation().transpose() <<") in world frame and ("
        <<pt_dsl_start_.transpose()                 <<") in OG coordinate"<<endl;
    cout<<"pose_dsl_start_ rotation:\n"<<pose_dsl_start_.rotation()<<endl;
  }

}

void
CallBackDslDdp::cbPoseGoal(const geometry_msgs::PoseStampedConstPtr& msg_pose_goal)
{
  if(config_.dyn_debug_on)
    cout<<"Initial pose received from rviz"<<endl;

  float m_per_cell = occ_grid_.info.resolution;
  int w = occ_grid_.info.width;
  int h = occ_grid_.info.height;

  poseMsg2Eig(pose_dsl_goal_,msg_pose_goal->pose);
  pt_dsl_goal_ = (tfm_world2og_ll_.inverse()*pose_dsl_goal_).translation()/m_per_cell;
  dispGoalRviz();
  dsl_cond_feas_g_ = (pt_dsl_goal_(0)>=0) && (pt_dsl_goal_(1)>=0) && (pt_dsl_goal_(0)<w) && (pt_dsl_goal_(1)<h);
  if(dsl_cond_feas_g_)
    p_gdsl_->SetGoal((int)pt_dsl_goal_(0),(int)pt_dsl_goal_(1));

  if(dsl_cond_feas_g_ && config_.dyn_debug_on)
  {
    cout<<"*goal position received at ("
        <<pose_dsl_goal_.translation().transpose() <<") in world frame and ("
        <<pt_dsl_goal_.transpose()                 <<") in OG coordinate"<<endl;
    cout<<"pose_dsl_goal rotation:\n"<<pose_dsl_goal_.rotation()<<endl;
  }

  if(!dsl_cond_feas_g_ && config_.dyn_debug_on)
  {
    cout<<"*Invalid goal position received at ("
        <<pose_dsl_goal_.translation().transpose() <<") in world frame and ("
        <<pt_dsl_goal_.transpose()                 <<") in OG coordinate"<<endl;
    cout<<"pose_dsl_goal rotation:\n"<<pose_dsl_goal_.rotation()<<endl;
  }
}

void
CallBackDslDdp::cbOccGrid(const nav_msgs::OccupancyGridConstPtr& msg_occ_grid)
{
  if(config_.dyn_debug_on)
  {
    cout<<"*Occupancy grid is received"<<endl;
  }

  occ_grid_ = *msg_occ_grid;
  int width = occ_grid_.info.width;
  int height = occ_grid_.info.height;
  for (int i = 0; i < width*height; ++i)
    occ_grid_.data[i] = (occ_grid_.data[i]<0?0:occ_grid_.data[i]);
  setTfms();
  dilateObs();
}

void
CallBackDslDdp::imgToOccGrid(const cv::Mat& img,const geometry_msgs::Pose pose_org, const double res_m_per_pix,  nav_msgs::OccupancyGrid& occ_grid)
{
  //confirm that the image is grayscale
  assert(img.channels()==1 && img.type()==0);

  //set occgrid header
  occ_grid.header.frame_id=strfrm_world_;
  occ_grid.header.stamp = ros::Time::now();

  //set occgrid info
  occ_grid.info.height = img.rows;
  occ_grid.info.width = img.cols;
  occ_grid.info.resolution =res_m_per_pix;//meter/pixel
  occ_grid.info.origin = pose_org;

  //set occgrid data
  occ_grid.data.reserve(img.rows*img.cols);
  occ_grid.data.assign(img.data, img.data+img.rows*img.cols) ;
}


void
CallBackDslDdp::setTfms(void)
{
  if(config_.dyn_debug_on)
  {
    cout<<"*Find the transformation between the world and the ll(lower left) corner of OG"<<endl;
  }
  //get the transformation between map to bottom left of the occupancy grid
  tf::StampedTransform tfms_map2og_org;
  tf_lr_.waitForTransform(strfrm_world_,strfrm_og_org_,ros::Time(0),ros::Duration(0.1));
  tf_lr_.lookupTransform(strfrm_world_,strfrm_og_org_, ros::Time(0), tfms_map2og_org);

  tf::transformTFToEigen(tfms_map2og_org,tfm_world2og_org_);
  poseMsg2Eig(tfm_og_org2og_ll_,occ_grid_.info.origin);
  tfm_world2og_ll_ = tfm_world2og_org_*tfm_og_org2og_ll_;

  double th = atan2(tfm_world2og_ll_.matrix()(1,0),tfm_world2og_ll_.matrix()(0,0));
  double x = tfm_world2og_ll_.matrix()(0,3);
  double y = tfm_world2og_ll_.matrix()(1,3);
  tfm_world2og_ll_2d_ = Translation2d(x,y)* Rotation2Dd(th);

}

void
CallBackDslDdp::dilateObs(void)
{


  int dyn_dilation_type;
  string str_type;
  switch(config_.dyn_dilation_type)
  {
    case 0:
      dyn_dilation_type=cv::MORPH_RECT;
      str_type = string("rectangle");
      break;
    case 1:
      dyn_dilation_type=cv::MORPH_CROSS;
      str_type = string("cross");
      break;
    case 2:
      dyn_dilation_type=cv::MORPH_ELLIPSE;
      str_type = string("ellipse");
      break;
  }

  //create cv::Mat for the occ_grid
  cv::Mat img_occ_grid = cv::Mat(occ_grid_.info.height,occ_grid_.info.width,CV_8UC1,(uint8_t*)occ_grid_.data.data());
  int dilation_size = config_.dyn_dilation_obs_m/occ_grid_.info.resolution;
  cv::Mat dilation_element = cv::getStructuringElement(dyn_dilation_type,
                                                       cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ) );

  cv::dilate(img_occ_grid  , img_occ_grid_dilated_, dilation_element );

  //display the dialated occ grid in rviz
  geometry_msgs::Pose pose_org; eig2PoseMsg(pose_org,tfm_world2og_ll_);
  nav_msgs::OccupancyGrid occ_grid_dilated;
  imgToOccGrid(img_occ_grid_dilated_,pose_org, occ_grid_.info.resolution,occ_grid_dilated);
  pub_og_dild_.publish(occ_grid_dilated);

  if(config_.dyn_debug_on)
  {
    cout<<"*Dilating obstacle with "<<str_type<<" type kernel of size "<<dilation_size<<"pixels"<<endl;
  }
  dslInit();

}

constexpr unsigned int str2int(const char* str, int h = 0)
{
  return !str[h] ? 5381 : (str2int(str, h+1)*33) ^ str[h];
}

void
CallBackDslDdp::endMarker(void)
{
  //remove visualization marker
  marker_path_dsl_.action      = visualization_msgs::Marker::DELETE;
  marker_path_dsl_intp_.action = visualization_msgs::Marker::DELETE;
  marker_path_ddp_.action      = visualization_msgs::Marker::DELETE;
  marker_text_start_.action    = visualization_msgs::Marker::DELETE;
  marker_text_goal_.action     = visualization_msgs::Marker::DELETE;
  marker_wp_dsl_.action        = visualization_msgs::Marker::DELETE;
  marker_wp_dsl_intp_.action   = visualization_msgs::Marker::DELETE;
  marker_wp_ddp_.action        = visualization_msgs::Marker::DELETE;
  pub_vis_.publish( marker_path_dsl_ );
  pub_vis_.publish( marker_path_dsl_intp_ );
  pub_vis_.publish( marker_path_ddp_ );
  pub_vis_.publish( marker_text_start_ );
  pub_vis_.publish( marker_text_goal_ );
  pub_vis_.publish( marker_wp_dsl_ );
  pub_vis_.publish( marker_wp_dsl_intp_ );
  pub_vis_.publish( marker_wp_ddp_ );
}

void
CallBackDslDdp::initRvizMarkers(void)
{
  if(config_.dyn_debug_on)
  {
    cout<<"*Initializing all rviz markers."<<endl;
  }
  VectorXd prop_path_n_wp;
  int id=-1;


  prop_path_pve_ddp_ = yaml_node_["prop_path_pve_ddp"].as<VectorXd>();
  prop_path_nve_ddp_ = yaml_node_["prop_path_nve_ddp"].as<VectorXd>();
  prop_wp_pve_ddp_ = yaml_node_["prop_wp_pve_ddp"].as<VectorXd>();
  prop_wp_nve_ddp_ = yaml_node_["prop_wp_nve_ddp"].as<VectorXd>();

  //Marker for dsl path
  id++;
  marker_path_dsl_.header.frame_id = strfrm_world_;
  marker_path_dsl_.header.stamp = ros::Time();
  marker_path_dsl_.ns = "dsl_ddp_planner";
  marker_path_dsl_.id = id;
  marker_path_dsl_.type = visualization_msgs::Marker::LINE_STRIP;
  marker_path_dsl_.action = visualization_msgs::Marker::ADD;
  marker_path_dsl_.lifetime = ros::Duration(0);
  prop_path_n_wp = yaml_node_["prop_path_dsl"].as<VectorXd>();
  editRvizMarker(marker_path_dsl_,prop_path_n_wp);

  //Marker for dsl path way points
  id++;
  marker_wp_dsl_.header.frame_id = strfrm_world_;
  marker_wp_dsl_.header.stamp = ros::Time();
  marker_wp_dsl_.ns = "dsl_ddp_planner";
  marker_wp_dsl_.id = id;
  marker_wp_dsl_.type = visualization_msgs::Marker::POINTS;
  marker_wp_dsl_.action = visualization_msgs::Marker::ADD;
  marker_wp_dsl_.lifetime = ros::Duration(0);
  prop_path_n_wp = yaml_node_["prop_wp_dsl"].as<VectorXd>();
  editRvizMarker(marker_wp_dsl_,prop_path_n_wp);

  //Marker for dsl path interpolated
  id++;
  marker_path_dsl_intp_.header.frame_id = strfrm_world_;
  marker_path_dsl_intp_.header.stamp = ros::Time();
  marker_path_dsl_intp_.ns = "dsl_ddp_planner";
  marker_path_dsl_intp_.id = id;
  marker_path_dsl_intp_.type = visualization_msgs::Marker::LINE_STRIP;
  marker_path_dsl_intp_.action = visualization_msgs::Marker::ADD;
  marker_path_dsl_intp_.lifetime = ros::Duration(0);
  prop_path_n_wp = yaml_node_["prop_path_dsl_intp"].as<VectorXd>();
  editRvizMarker(marker_path_dsl_intp_,prop_path_n_wp);

  //Marker for dsl path interpolated way points
  id++;
  marker_wp_dsl_intp_.header.frame_id = strfrm_world_;
  marker_wp_dsl_intp_.header.stamp = ros::Time();
  marker_wp_dsl_intp_.ns = "dsl_ddp_planner";
  marker_wp_dsl_intp_.id = id;
  marker_wp_dsl_intp_.type = visualization_msgs::Marker::POINTS;
  marker_wp_dsl_intp_.action = visualization_msgs::Marker::ADD;
  marker_wp_dsl_intp_.lifetime = ros::Duration(0);
  prop_path_n_wp = yaml_node_["prop_wp_dsl_intp"].as<VectorXd>();
  editRvizMarker(marker_wp_dsl_intp_,prop_path_n_wp);

  //Marker for pve ddp path
  id++;
  marker_path_ddp_.header.frame_id = strfrm_world_;
  marker_path_ddp_.header.stamp = ros::Time();
  marker_path_ddp_.ns = "dsl_ddp_planner";
  marker_path_ddp_.id = id;
  marker_path_ddp_.type = visualization_msgs::Marker::LINE_STRIP;
  marker_path_ddp_.action = visualization_msgs::Marker::ADD;
  marker_path_ddp_.lifetime = ros::Duration(0);
  editRvizMarker(marker_path_ddp_,prop_path_pve_ddp_);

  //Marker for ddp path way points
  id++;
  marker_wp_ddp_.header.frame_id = strfrm_world_;
  marker_wp_ddp_.header.stamp = ros::Time();
  marker_wp_ddp_.ns = "dsl_ddp_planner";
  marker_wp_ddp_.id = id;
  marker_wp_ddp_.type = visualization_msgs::Marker::POINTS;
  marker_wp_ddp_.action = visualization_msgs::Marker::ADD;
  marker_wp_ddp_.lifetime = ros::Duration(0);
  editRvizMarker(marker_wp_ddp_,prop_wp_pve_ddp_);

  //Marker for "start" text
  id++;
  marker_text_start_.header.frame_id = strfrm_world_;
  marker_text_start_.header.stamp = ros::Time();
  marker_text_start_.ns = "dsl_ddp_planner";
  marker_text_start_.id = id;
  marker_text_start_.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker_text_start_.action = visualization_msgs::Marker::ADD;
  marker_text_start_.text="S";
  marker_text_start_.scale.z = 4;
  marker_text_start_.color.r = 1.0;
  marker_text_start_.color.g = 0.0;
  marker_text_start_.color.b = 0.0;
  marker_text_start_.color.a = 1.0; // Don't forget to set the alpha!
  marker_text_start_.lifetime = ros::Duration(0);

  //Marker for "goal" text
  id++;
  marker_text_goal_.header.frame_id = strfrm_world_;
  marker_text_goal_.header.stamp = ros::Time();
  marker_text_goal_.ns = "dsl_ddp_planner";
  marker_text_goal_.id = id;
  marker_text_goal_.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker_text_goal_.action = visualization_msgs::Marker::ADD;
  marker_text_goal_.text="G";
  marker_text_goal_.scale.z = 4;
  marker_text_goal_.color.a = 1.0; // Don't forget to set the alpha!
  marker_text_goal_.color.r = 1.0;
  marker_text_goal_.color.g = 0.0;
  marker_text_goal_.color.b = 0.0;
  marker_text_goal_.lifetime = ros::Duration(0);
}
void
CallBackDslDdp::editColorMsg(std_msgs::ColorRGBA& rgba_msg, VectorXd& rgba_vec)
{
  rgba_msg.r = rgba_vec(0);
  rgba_msg.g = rgba_vec(1);
  rgba_msg.b = rgba_vec(2);
  rgba_msg.a = rgba_vec(3);
}

void
CallBackDslDdp::editRvizMarker(visualization_msgs::Marker& marker, VectorXd& prop)
{
  switch(prop.size())
  {
    case 3:
      marker.color.r = prop(0);// red
      marker.color.g = prop(1);// blue
      marker.color.b = prop(2);// green
      break;
    case 4:
      marker.color.r = prop(0);// red
      marker.color.g = prop(1);// blue
      marker.color.b = prop(2);// green
      marker.color.a = prop(3);//alpha
      break;
    case 5:
      marker.color.r = prop(0);// red
      marker.color.g = prop(1);// blue
      marker.color.b = prop(2);// green
      marker.color.a = prop(3);//alpha
      marker.scale.x = prop(4);//width
      break;
    case 6:
      marker.color.r = prop(0);// red
      marker.color.g = prop(1);// blue
      marker.color.b = prop(2);// green
      marker.color.a = prop(3);//alpha
      marker.scale.x = prop(4);//width
      marker.scale.y = prop(5);//height
      break;
    default:
      cout<<"Error setting marker properties"<<endl;
      break;
  }
}

void
CallBackDslDdp::dslDelete()
{
  delete[] map_dsl_;
  delete p_gdsl_;
}

void
CallBackDslDdp::dslInit()
{
  ros::Time t_start = ros::Time::now();
  dslDelete();
  map_dsl_ = new double[occ_grid_.info.width*occ_grid_.info.height];
  for (int i = 0; i < occ_grid_.info.width*occ_grid_.info.height; ++i)
    map_dsl_[i] = 1000*(double)img_occ_grid_dilated_.data[i];
  p_gdsl_ = new dsl::GridSearch(occ_grid_.info.width, occ_grid_.info.height, grid_cost_, map_dsl_);
  ros::Time t_end =  ros::Time::now();
  if(config_.dyn_debug_on)
  {
    cout<<"*Initialized DSL grid search object with map size:"<<occ_grid_.info.width<<" X "<<occ_grid_.info.height<<endl;
    cout<<"  delta t:"<<(t_end - t_start).toSec()<<" sec"<<endl;
  }
}

bool
CallBackDslDdp::dslFeasible(void)
{
  float w = occ_grid_.info.width;
  float h = occ_grid_.info.height;
  return w && h && dsl_cond_feas_s_ && dsl_cond_feas_g_;
}

bool
CallBackDslDdp::dslPlan()
{
  if(dslFeasible())
  {
    ros::Time t_start = ros::Time::now();
    dsl::GridPath path_init;
    p_gdsl_->Plan(path_init);
    p_gdsl_->OptPath(path_init, path_opt_);
    ros::Time t_end =  ros::Time::now();
    if(config_.dyn_debug_on)
    {
      cout<<"*Planned a path and optimized it and obtained a path(with "<<path_opt_.cells.size()<<"nodes)"<<endl;
      cout<<"  delta t:"<<(t_end - t_start).toSec()<<" sec"<<endl;
    }
    dslInterpolate();
    dsl_done_ = true;
  }
  else
  {
    if(config_.dyn_debug_on)
    {
      cout<<"*Planning with DSL not possible because it's infeasible"<<endl;
    }
    dsl_done_=false;
  }

  if(dslFeasible())
    return true;
  else
    return false;
}

void
CallBackDslDdp::dslInterpolate(void)
{

  float m_per_cell = occ_grid_.info.resolution;
  int n = path_opt_.cells.size();
  //std::vector<double> x(path_opt_.count), y(path_opt_.count),d(path_opt_.count), t(path_opt_.count);
  VectorXd pt_x_opt(n),pt_y_opt(n);    pt_x_opt.setZero(); pt_y_opt.setZero();
  VectorXd delx(n),dely(n); delx.setZero(); dely.setZero();
  VectorXd delt(n),t_opt(n);  delt.setZero(); t_opt.setZero();

  for (int i = 0; i < n; i++)
  {
    pt_x_opt(i) = path_opt_.cells[i][0];
    pt_y_opt(i) = path_opt_.cells[i][1];
  }
  delx.tail(n-1) = pt_x_opt.tail(n-1) - pt_x_opt.head(n-1);
  dely.tail(n-1) = pt_y_opt.tail(n-1) - pt_y_opt.head(n-1);

  //The time difference between 2 consecutive way points
  //is calculated as distance between the 2 way points divided by the top speed
  delt = ((delx.array().square() + dely.array().square()).array().sqrt())*m_per_cell/config_.dyn_dsl_avg_speed;
  for(int i=1;i<n;i++)
    t_opt(i) =t_opt(i-1) + delt(i);

  //Create linear interpolator for pre-interpolation from opt points
  SplineFunction lint_x(t_opt,pt_x_opt,1);
  SplineFunction lint_y(t_opt,pt_y_opt,1);

  //Pre-Interpolate
  bool update_dyn_server=false;
  if(config_.dyn_dsl_preint_delt > t_opt(n-1) )
  {
    config_.dyn_dsl_preint_delt = t_opt(n-1);
    update_dyn_server = true;
  }

  int n_segs_reg; n_segs_reg = round((double)(t_opt(n-1))/config_.dyn_dsl_preint_delt);
  int n_nodes_reg = n_segs_reg+1;//regularly spaced points
  double dt_reg= (double)(t_opt(n-1))/(double)n_segs_reg;


  //  Reserve space for regularly spaced nodes and the original nodes.
  //    n-2 because the 1st and last interpolated points are same as original
  vector<double> pt_x_dsl_preint_stl;vector<double> pt_y_dsl_preint_stl;vector<double> t_dsl_preint_stl;
  pt_x_dsl_preint_stl.reserve(n_nodes_reg + n-2); pt_x_dsl_preint_stl.push_back(pt_x_opt(0));
  pt_y_dsl_preint_stl.reserve(n_nodes_reg + n-2); pt_y_dsl_preint_stl.push_back(pt_y_opt(0));
  t_dsl_preint_stl.reserve(n_nodes_reg + n-2);    t_dsl_preint_stl.push_back(t_opt(0));

  int idx_opt=1; //index of original
  int idx_reg=1;
  double tol=1e-10;
  while( abs(t_opt(n-1) - t_dsl_preint_stl.back()) > tol)
  {
    //insert the original points if they are not already there
    double t_next = idx_reg*dt_reg;
    while( t_opt(idx_opt)<= t_next && idx_opt<n-1)
    {
      if(abs(t_opt(idx_opt) - t_next ) > 1e-8)
      {
        pt_x_dsl_preint_stl.push_back(pt_x_opt(idx_opt));
        pt_y_dsl_preint_stl.push_back(pt_y_opt(idx_opt));
        t_dsl_preint_stl.push_back(t_opt(idx_opt));
      }
      idx_opt++;
    }
    //insert the regularly spaced linear interpolation points
    pt_x_dsl_preint_stl.push_back(lint_x[t_next]);
    pt_y_dsl_preint_stl.push_back(lint_y[t_next]);
    t_dsl_preint_stl.push_back(t_next);
    idx_reg++;
  }

  //convert stl vectors to eigen vectors
  VectorXd pt_x_dsl_preint = Map<VectorXd>(pt_x_dsl_preint_stl.data(),pt_x_dsl_preint_stl.size());
  VectorXd pt_y_dsl_preint = Map<VectorXd>(pt_y_dsl_preint_stl.data(),pt_y_dsl_preint_stl.size());
  VectorXd t_dsl_preint    = Map<VectorXd>(t_dsl_preint_stl.data(),t_dsl_preint_stl.size());

  //Create spline interpolator from preint points
  SplineFunction intp_x(t_dsl_preint,pt_x_dsl_preint,config_.dyn_dsl_interp_deg);
  SplineFunction intp_y(t_dsl_preint,pt_y_dsl_preint,config_.dyn_dsl_interp_deg);

  //Interpolate
  if(config_.dyn_dsl_interp_delt > t_opt(n-1) )
  {
    config_.dyn_dsl_interp_delt = t_opt(n-1);
    update_dyn_server = true;
  }
  int n_segs; n_segs = round((double)(t_opt(n-1))/config_.dyn_dsl_interp_delt);
  int n_nodes = n_segs+1;
  double dt= (double)(t_opt(n-1))/(double)n_segs;

  pt_x_dsl_intp_.resize(n_nodes); pt_y_dsl_intp_.resize(n_nodes);a_dsl_intp_.resize(n_nodes); t_dsl_intp_.resize(n_nodes);
  pt_x_dsl_intp_(0) = pt_x_opt(0);     pt_y_dsl_intp_(0) = pt_y_opt(0);     t_dsl_intp_(0) = t_opt(0);

  for(int i=1;i<n_nodes;i++)
  {
    t_dsl_intp_(i) = i*dt;
    pt_x_dsl_intp_(i) = intp_x[i*dt];
    pt_y_dsl_intp_(i) = intp_y[i*dt];

    double dx = pt_x_dsl_intp_(i) -pt_x_dsl_intp_(i-1);
    double dy = pt_y_dsl_intp_(i) -pt_y_dsl_intp_(i-1);
    a_dsl_intp_(i-1) = atan2(dy,dx);
  }
  a_dsl_intp_(n_nodes-1) = a_dsl_intp_(n_nodes-2);

  if(update_dyn_server)
    dyn_server_.updateConfig(config_);
}

bool
CallBackDslDdp::ddpFeasible(void)
{
  return dsl_done_;
}

bool
CallBackDslDdp::ddpInit(void)
{
  if(config_.dyn_debug_on)
    cout<<"*Setting DDP params"<<endl;

  //Fetch and set all the parameter from yaml file
  cost_lq_.Q  = yaml_node_["ddp_Q"].as<Matrix4d>();
  cost_lq_.Qf = yaml_node_["ddp_Qf"].as<Matrix4d>();
  cost_lq_.R  = yaml_node_["ddp_R"].as<Matrix2d>();
  ddp_mu_ =yaml_node_["ddp_mu"].as<double>();
  ddp_debug_on_ =yaml_node_["ddp_debug_on"].as<bool>(); // turn off debug for speed
  ddp_nseg_min_= (yaml_node_["ddp_nseg_minmax"].as<Vector2d>())(0);
  ddp_nseg_max_= (yaml_node_["ddp_nseg_minmax"].as<Vector2d>())(1);
  ddp_tseg_ideal_ =yaml_node_["ddp_tseg_ideal"].as<double>();
  ddp_nit_max_=yaml_node_["ddp_nit_max"].as<int>();
  ddp_tol_abs_=yaml_node_["ddp_tol_abs"].as<double>();
  ddp_tol_rel_=yaml_node_["ddp_tol_rel"].as<double>();
  //Update internal gains of cost_lq
  cost_lq_.UpdateGains();
}

bool
CallBackDslDdp::ddpPlan(void)
{
  if(config_.dyn_debug_on)
    cout<<"*Entering local path planning"<<endl;

  if(!ddpFeasible())
    return false;
  float m_per_cell = occ_grid_.info.resolution;

  //DDP start
  //if dyn_ddp_from_curr_posn=true then start is current position else
  if(config_.dyn_ddp_from_curr_posn)
  {
    time_ddp_start_ = time_ddp_curr_;
    pose_ddp_start_ = pose_ddp_curr_;
    vel_ddp_start_  =  vel_ddp_curr_;
  }
  else
  {
    time_ddp_start_ = ros::Time::now();
    pose_ddp_start_ = pose_dsl_start_;
    vel_ddp_start_  = 0;
  }
  //Select ddp goal position by finding a point on the dsl way point
  //  that is t_away sec away from current position

  //  find nearest row(idx_nearest) on the [x_dsl_intp_ ,y_dsl_intp_] to posn_ddp_start_
  int n_nodes = pt_x_dsl_intp_.size();
  Vector3d pt_ddp_start = (tfm_world2og_ll_.inverse()*pose_ddp_start_).translation()/m_per_cell;
  VectorXd dist_sq =(pt_x_dsl_intp_ - VectorXd::Ones(n_nodes)*pt_ddp_start(0)).array().square()
                               +(pt_y_dsl_intp_ - VectorXd::Ones(n_nodes)*pt_ddp_start(1)).array().square();
  VectorXd::Index idx_min; dist_sq.minCoeff(&idx_min);
  int idx_nearest = (int)idx_min;

  //  set pose_ddp_goal_ to xy_dsl_intp_(idx_t_away) where idx_t_away is the index of dsl_intp path
  //    which is t_away sec ahead of dsl_intp path at idx_nearest
  vector<double> t_stl(t_dsl_intp_.size()); Map<VectorXd>(t_stl.data(),t_dsl_intp_.size()) = t_dsl_intp_;
  double t_away = config_.dyn_ddp_t_away;
  vector<double>::iterator it_t_away = upper_bound(t_stl.begin(),t_stl.end(),t_stl[idx_nearest]+t_away);
  int idx_t_away = it_t_away-t_stl.begin()==t_stl.size()? t_stl.size()-1: it_t_away-t_stl.begin();

  //  DDP goal
  pose_ddp_goal_ = tfm_world2og_ll_
                  *Translation3d(m_per_cell*pt_x_dsl_intp_(idx_t_away),m_per_cell*pt_y_dsl_intp_(idx_t_away),0)
                  *AngleAxisd(a_dsl_intp_(idx_t_away), Vector3d::UnitZ());

  //DDP path length in distance and time
  double tf = t_dsl_intp_(idx_t_away) - t_dsl_intp_(idx_nearest);
  double len_ddp_path = tf*config_.dyn_dsl_avg_speed;

  // Start and goal ddp state(GcarState. M3 is se2 elem and V1 is forward vel)
  Matrix3d se2_0; se2_0.setZero();
  se2_0.block<2,2>(0,0)= (pose_ddp_start_.matrix()).block<2,2>(0,0);
  se2_0.block<2,1>(0,2) = (pose_ddp_start_.matrix()).block<2,1>(0,3);
  se2_0(2,2)=1;
  GcarState x0(se2_0,vel_ddp_start_);


  Matrix3d se2_f; se2_f.setZero();
  se2_f.block<2,2>(0,0)= (pose_ddp_goal_.matrix()).block<2,2>(0,0);
  se2_f.block<2,1>(0,2) = (pose_ddp_goal_.matrix()).block<2,1>(0,3);
  se2_f(2,2)=1;
  GcarState xf(se2_f, config_.dyn_dsl_avg_speed);

  //Determine the number of segments for ddp based on tf, nseg_max, tseg_ideal
  //  and resize ts xs and us based on that
  int ddp_nseg = ceil(tf/ddp_tseg_ideal_);
  ddp_nseg = ddp_nseg>ddp_nseg_max_?ddp_nseg_max_:ddp_nseg;
  ddp_nseg = ddp_nseg<ddp_nseg_min_?ddp_nseg_min_:ddp_nseg;
  ddp_ts_.resize(ddp_nseg+1);
  ddp_xs_.resize(ddp_nseg+1);
  ddp_us_.resize(ddp_nseg);

  Vector3d rpy_start; SO3::Instance().g2q(rpy_start,pose_ddp_start_.linear());
  Vector3d rpy_goal; SO3::Instance().g2q(rpy_goal,pose_ddp_goal_.linear());
  if(config_.dyn_debug_on)
  {
    cout<<"  The ddp request is as follows"<<endl;
    cout<<"    Start x:"<<pose_ddp_start_.translation()(0)<<"\ty:"<<pose_ddp_start_.translation()(1)<<"\ta:"<<rpy_start(2)<<endl;
    cout<<"    Goal x:"<<pose_ddp_goal_.translation()(0)<<"\ty:"<<pose_ddp_goal_.translation()(1)<<"\ta:"<<rpy_goal(2)<<endl;
    cout<<"    tf:"<< tf<<" sec";
    cout<<"    path length:"<<len_ddp_path<<endl;
    cout<<"    nseg:"<<ddp_nseg<<endl;
  }

  //Set tf and xf for cost_lq_
  cost_lq_.tf = tf;
  cost_lq_.xf = &xf;

  //Initialize trajectory(ts, us and xs)
  double h=tf/ddp_nseg;
  for (int k = 0; k <ddp_nseg+1; ++k)
    ddp_ts_[k] = k*h;

  for (int k = 0; k <ddp_nseg; ++k)
    ddp_us_[k].setZero();// or put a function that sets the initial value of u

  ddp_xs_[0]=x0;

  //DDP update
  GcarDdp ddp_solver(sys_gcar_, cost_lq_, ddp_ts_, ddp_xs_, ddp_us_);
  ddp_solver.mu =ddp_mu_;
  ddp_solver.debug = ddp_debug_on_;
  //ddp_solver.Update()

  //Run an iteration loop until convergence
  int n_it(0);
  bool ddp_conv=false;
  double v_prev; v_prev=ddp_solver.V;

  while(!ddp_conv && !g_shutdown_requested)
  {
    if(config_.dyn_debug_on)
      cout<<"Iteration number:"<<n_it<<endl;

    ddp_solver.Iterate();

    n_it++;

    cout<<"tol_rel"<<abs(ddp_solver.V - v_prev)/abs(v_prev) <<endl;
    cout<<"tol_abs"<<abs(ddp_solver.V - v_prev) <<endl;
    if(                                     n_it  > ddp_nit_max_
        ||             abs(ddp_solver.V - v_prev) < ddp_tol_abs_
        || abs(ddp_solver.V - v_prev)/abs(v_prev) < ddp_tol_rel_)
      ddp_conv=true;
    v_prev=ddp_solver.V;
  }

  //create the GcarCtrl message and publish it
  if(config_.dyn_send_gcar_ctrl)
  {
    gcop_comm::GcarCtrl msg_ctrl;
    msg_ctrl.ts_eph.resize(ddp_us_.size());
    msg_ctrl.us_vel.resize(ddp_us_.size());
    msg_ctrl.us_phi.resize(ddp_us_.size());
    for (int i = 0; i < ddp_us_.size(); ++i)
    {
      msg_ctrl.ts_eph[i] = time_ddp_start_+ ros::Duration(ddp_ts_[i]);
      msg_ctrl.us_vel[i] = ddp_xs_[i].v;
      msg_ctrl.us_phi[i] = atan(ddp_us_[i](1));
    }
    pub_ctrl_.publish(msg_ctrl);
  }

  return true;
}

void
CallBackDslDdp::dispPathDdpRviz(void)
{
  if(config_.dyn_debug_on)
    cout<<"*Displaying ddp path"<<endl;

  float res = occ_grid_.info.resolution;

  marker_path_ddp_.points.resize(ddp_xs_.size());
  marker_path_ddp_.colors.resize(ddp_xs_.size());
  marker_wp_ddp_.points.resize(ddp_xs_.size());
  marker_wp_ddp_.colors.resize(ddp_xs_.size());

  for (int i = 0; i < ddp_xs_.size(); i++)
  {
    geometry_msgs::Point node;
    node.x    = (ddp_xs_[i].g)(0,2);
    node.y    = (ddp_xs_[i].g)(1,2);
    double th = atan2(ddp_xs_[i].g(1,0),ddp_xs_[i].g(0,0));
    double v = ddp_xs_[i].v;
    node.z = 0.2;
    marker_path_ddp_.points[i] =node;
    marker_wp_ddp_.points[i] =node;

    if(v>0)
    {
      editColorMsg(marker_path_ddp_.colors[i],prop_path_pve_ddp_);
      editColorMsg(marker_wp_ddp_.colors[i],prop_wp_pve_ddp_);
    }
    else
    {
      editColorMsg(marker_path_ddp_.colors[i],prop_path_nve_ddp_);
      editColorMsg(marker_wp_ddp_.colors[i],prop_wp_nve_ddp_);
    }
  }
  pub_vis_.publish( marker_path_ddp_ );
  pub_vis_.publish( marker_wp_ddp_ );
}

void
CallBackDslDdp::dispPathDslRviz()
{
  if(config_.dyn_debug_on)
    cout<<"*Displaying dsl path"<<endl;

  float m_per_cell = occ_grid_.info.resolution;
  marker_path_dsl_.points.resize(path_opt_.cells.size());
  marker_wp_dsl_.points.resize(path_opt_.cells.size());
  for (int i = 0; i < path_opt_.cells.size(); i++)
  {
    Vector3d posn_waypt_in_ll(m_per_cell*path_opt_.cells[i][0],m_per_cell*(path_opt_.cells[i][1]),0);
    Vector3d posn_waypt_in_world = tfm_world2og_ll_*posn_waypt_in_ll;
    geometry_msgs::Point node;
    node.x = posn_waypt_in_world(0);
    node.y = posn_waypt_in_world(1);
    node.z = 0.2;
    marker_path_dsl_.points[i] =node;
    marker_wp_dsl_.points[i] =node;
  }
  pub_vis_.publish( marker_path_dsl_ );
  pub_vis_.publish( marker_wp_dsl_ );
}

void
CallBackDslDdp::dispPathDslInterpdRviz()
{
  if(config_.dyn_debug_on)
    cout<<"*Displaying interpolated dsl path"<<endl;

  float m_per_cell = occ_grid_.info.resolution;
  marker_path_dsl_intp_.points.resize(pt_x_dsl_intp_.size());
  marker_wp_dsl_intp_.points.resize(pt_x_dsl_intp_.size());
  for (int i = 0; i < pt_x_dsl_intp_.size(); i++)
  {
    Vector3d posn_waypt_in_ll(m_per_cell*pt_x_dsl_intp_(i),m_per_cell*pt_y_dsl_intp_(i),0);
    Vector3d posn_waypt_in_world = tfm_world2og_ll_*posn_waypt_in_ll;
    geometry_msgs::Point node;
    node.x = posn_waypt_in_world(0);
    node.y = posn_waypt_in_world(1);
    node.z = 0.2;
    marker_path_dsl_intp_.points[i] =node;
    marker_wp_dsl_intp_.points[i] =node;
  }
  pub_vis_.publish( marker_path_dsl_intp_ );
  pub_vis_.publish( marker_wp_dsl_intp_ );
}

void
CallBackDslDdp::dispStartRviz()
{
  if(config_.dyn_debug_on)
    cout<<"*Displaying start markers"<<endl;

  marker_text_start_.pose.position.x =pose_dsl_start_.translation()(0);
  marker_text_start_.pose.position.y =pose_dsl_start_.translation()(1);
  marker_text_start_.pose.position.z =0.2;

  pub_vis_.publish( marker_text_start_ );
}

void
CallBackDslDdp::dispGoalRviz()
{
  if(config_.dyn_debug_on)
    cout<<"*Displaying goal markers"<<endl;

  marker_text_goal_.pose.position.x =pose_dsl_goal_.translation()(0);
  marker_text_goal_.pose.position.y =pose_dsl_goal_.translation()(1);
  marker_text_goal_.pose.position.z =0.2;

  pub_vis_.publish( marker_text_goal_ );
}


//------------------------------------------------------------------------
//--------------------------------MAIN -----------------------------------
//------------------------------------------------------------------------

int main(int argc, char** argv)
{
  ros::init(argc,argv,"dsl_ddp_planner",ros::init_options::NoSigintHandler);
  signal(SIGINT,mySigIntHandler);

  ros::NodeHandle nh;

  CallBackDslDdp cbc;

  while(!g_shutdown_requested)
  {
    ros::spinOnce();

    cbc.loop_rate_main_.sleep();
  }
  return 0;

}
