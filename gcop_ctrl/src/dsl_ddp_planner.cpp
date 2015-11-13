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
#include <gcop_comm/State.h>
#include <gcop_comm/CtrlTraj.h>
#include <gcop_comm/Trajectory_req.h>

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
#include <dsl/distedgecost.h>

//gcop include
#include <gcop/so3.h>

//yaml
#include <yaml-cpp/yaml.h>

//local includes
#include <eigen_ros_conv.h>
#include <eig_splinterp.h>
#include "yaml_eig_conv.h"

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

public:
  CallBackDslDdp();
  ~CallBackDslDdp();
private:
  ros::NodeHandle nh_, nh_p_;
  YAML::Node yaml_node_;

  string strtop_odom_, strtop_pose_start_, strtop_pose_goal_, strtop_og_;
  string strtop_diag_, strtop_marker_rviz_, strtop_og_dild_, strtop_ctrl_;
  string strfrm_world_, strfrm_robot_, strfrm_og_org_;

  ros::Subscriber sub_odom_, sub_pose_start_, sub_pose_goal_, sub_og_;
  ros::Publisher pub_diag_, pub_vis_, pub_og_dild_, pub_ctrl_;
  ros::Timer timer_vis_;
  ros::ServiceClient srvcl_traj_;

  tf::TransformBroadcaster tf_br_;
  tf::TransformListener tf_lr_;

  gcop_ctrl::DslDdpPlannerConfig config_;
  dynamic_reconfigure::Server<gcop_ctrl::DslDdpPlannerConfig> dyn_server_;
  visualization_msgs::Marker marker_path_dsl_,marker_path_dsl_intp_,marker_path_ddp_;
  visualization_msgs::Marker marker_start_, marker_goal_, marker_start_sphere_, marker_goal_sphere_;
  nav_msgs::OccupancyGrid occ_grid_;
  cv::Mat img_occ_grid_dilated_;


  Affine3d tfm_world2og_org_, tfm_world2og_ll_,tfm_og_org2og_ll_;
  Affine3d pose_dsl_start_, pose_dsl_goal_, pose_ddp_start_, pose_ddp_goal_, pose_curr_;
  Vector3d pt_dsl_start_, pt_dsl_goal_, pt_ddp_start_, pt_ddp_goal_; //refers to grid point

  dsl::GridSearch* p_gdsl_;
  double* map_dsl_;
  dsl::GridPath path_opt_;
  bool cond_feas_s_, cond_feas_g_;
  VectorXd x_dsl_intp_, y_dsl_intp_,a_dsl_intp_, t_dsl_intp_;//x,y,angle,t of the path

  gcop_comm::Trajectory_req traj_req_resp_;

private:

  void cbReconfig(gcop_ctrl::DslDdpPlannerConfig &config, uint32_t level);

  void cbOdom(const nav_msgs::OdometryConstPtr& msg_odom);
  void cbPoseStart(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg_pose_start);
  void cbPoseGoal(const geometry_msgs::PoseStampedConstPtr& msg_pose_goal);
  void cbOccGrid(const nav_msgs::OccupancyGridConstPtr& msg_occ_grid);

  void cbTimerVis(const ros::TimerEvent& event);
  void endMarker(void);

  void setupTopicsAndNames(void);
  void initSubsPubsAndTimers(void);
  void initRvizMarkers(void);

  void dispPathDslRviz(void);
  void dispPathDslInterpdRviz();
  void dispPathDdpRviz(void);
  void dispStartRviz(void);
  void dispGoalRviz(void);


  void dslInit(void);
  void dslDelete(void);
  void dslPlan(void);
  bool dslFeasible(void);
  void dslInterpolate(void);

  bool ddpPlan(void);


  void dilateObs(void);
  void setTfms(void);
  void imgToOccGrid(const cv::Mat& img,const geometry_msgs::Pose pose_org, const double res_m_per_pix,  nav_msgs::OccupancyGrid& occ_grid);

};

CallBackDslDdp::CallBackDslDdp():
            nh_p_("~"),
            cond_feas_s_(false),
            cond_feas_g_(false),
            p_gdsl_(nullptr),
            map_dsl_(nullptr)
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

  //Setup service clients
  srvcl_traj_  = nh_.serviceClient<gcop_comm::Trajectory_req>("req_ddp_lcl");

  //init ddp planner
  traj_req_resp_.request.itreq.x0.statevector.resize(4);
  traj_req_resp_.request.itreq.xf.statevector.resize(4);

  cout<<"**************************************************************************"<<endl;
  cout<<"****************************DSL TRAJECTORY PLANNER************************"<<endl;
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
  bool condn_dilate = occ_grid_.info.width && (config.dyn_obs_dilation_m != config_.dyn_obs_dilation_m || config.dyn_dilation_type != config_.dyn_dilation_type );

  config_ = config;
  if(!first_time)
  {
    if(condn_dilate)
    {
      dilateObs();
    }

    if(config.dyn_dsl_plan_once)
    {
      dslPlan();
      dispPathDslRviz();
      dispPathDslInterpdRviz();
      config.dyn_dsl_plan_once = false;
    }

    if(config.dyn_ddp_plan_once)
    {
      if(ddpPlan())
        dispPathDdpRviz();
      config.dyn_ddp_plan_once = false;
    }

  }
  else
    first_time = false;
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
  pub_ctrl_    = nh_.advertise<gcop_comm::Ctrl>(strtop_ctrl_,0);

  //Setup timers
  timer_vis_ = nh_.createTimer(ros::Duration(0.1), &CallBackDslDdp::cbTimerVis, this);
  timer_vis_.stop();
}


void
CallBackDslDdp::cbTimerVis(const ros::TimerEvent& event)
{
  pub_vis_.publish( marker_path_dsl_ );
}

void
CallBackDslDdp::cbOdom(const nav_msgs::OdometryConstPtr& msg_odom)
{
  poseMsg2Eig(pose_curr_,msg_odom->pose.pose);
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
  cond_feas_s_ = (pt_dsl_start_(0)>=0) && (pt_dsl_start_(1)>=0) && (pt_dsl_start_(0)<w) && (pt_dsl_start_(1)<h);
  if(cond_feas_s_)
    p_gdsl_->SetStart((int)pt_dsl_start_(0),(int)pt_dsl_start_(1));

  if(cond_feas_s_ && config_.dyn_debug_on)
  {
    cout<<"*Start position received at ("
        <<pose_dsl_start_.translation().transpose() <<") in world frame and ("
        <<pt_dsl_start_.transpose()                 <<") in OG coordinate"<<endl;
    cout<<"pose_dsl_start rotation:\n"<<pose_dsl_start_.rotation()<<endl;
  }

  if(!cond_feas_s_ && config_.dyn_debug_on)
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
  cond_feas_g_ = (pt_dsl_goal_(0)>=0) && (pt_dsl_goal_(1)>=0) && (pt_dsl_goal_(0)<w) && (pt_dsl_goal_(1)<h);
  if(cond_feas_g_)
    p_gdsl_->SetGoal((int)pt_dsl_goal_(0),(int)pt_dsl_goal_(1));

  if(cond_feas_g_ && config_.dyn_debug_on)
  {
    cout<<"*goal position received at ("
        <<pose_dsl_goal_.translation().transpose() <<") in world frame and ("
        <<pt_dsl_goal_.transpose()                 <<") in OG coordinate"<<endl;
    cout<<"pose_dsl_goal rotation:\n"<<pose_dsl_goal_.rotation()<<endl;
  }

  if(!cond_feas_g_ && config_.dyn_debug_on)
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
  int dilation_size = config_.dyn_obs_dilation_m/occ_grid_.info.resolution;
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
  marker_path_dsl_.action     = visualization_msgs::Marker::DELETE;
  marker_path_ddp_.action = visualization_msgs::Marker::DELETE;
  marker_start_.action    = visualization_msgs::Marker::DELETE;
  marker_goal_.action     = visualization_msgs::Marker::DELETE;
  pub_vis_.publish( marker_path_dsl_ );
  pub_vis_.publish( marker_path_ddp_ );
  pub_vis_.publish( marker_start_ );
  pub_vis_.publish( marker_goal_ );
}

void
CallBackDslDdp::initRvizMarkers(void)
{
  if(config_.dyn_debug_on)
  {
    cout<<"*Initializing all rviz markers."<<endl;
  }
  int id=-1;
  //Marker for dsl path
  id++;
  marker_path_dsl_.header.frame_id = strfrm_world_;
  marker_path_dsl_.header.stamp = ros::Time();
  marker_path_dsl_.ns = "dsl_ddp_planner";
  marker_path_dsl_.id = id;
  marker_path_dsl_.type = visualization_msgs::Marker::LINE_STRIP;
  marker_path_dsl_.action = visualization_msgs::Marker::ADD;
  marker_path_dsl_.scale.x = 0.4;
  marker_path_dsl_.color.a = 0.5; // Don't forget to set the alpha!
  marker_path_dsl_.color.r = 0.0;
  marker_path_dsl_.color.g = 1.0;
  marker_path_dsl_.color.b = 0.0;
  marker_path_dsl_.lifetime = ros::Duration(0);

  //Marker for dsl path
  id++;
  marker_path_dsl_intp_.header.frame_id = strfrm_world_;
  marker_path_dsl_intp_.header.stamp = ros::Time();
  marker_path_dsl_intp_.ns = "dsl_ddp_planner";
  marker_path_dsl_intp_.id = id;
  marker_path_dsl_intp_.type = visualization_msgs::Marker::LINE_STRIP;
  marker_path_dsl_intp_.action = visualization_msgs::Marker::ADD;
  marker_path_dsl_intp_.scale.x = 0.4;
  marker_path_dsl_intp_.color.a = 0.5; // Don't forget to set the alpha!
  marker_path_dsl_intp_.color.r = 1.0;
  marker_path_dsl_intp_.color.g = 0.0;
  marker_path_dsl_intp_.color.b = 0.0;
  marker_path_dsl_intp_.lifetime = ros::Duration(0);

  //Marker for ddp path
  id++;
  marker_path_ddp_.header.frame_id = strfrm_world_;
  marker_path_ddp_.header.stamp = ros::Time();
  marker_path_ddp_.ns = "dsl_ddp_planner";
  marker_path_ddp_.id = id;
  marker_path_ddp_.type = visualization_msgs::Marker::LINE_STRIP;
  marker_path_ddp_.action = visualization_msgs::Marker::ADD;
  marker_path_ddp_.scale.x = 0.4;
  marker_path_ddp_.color.a = 0.5; // Don't forget to set the alpha!
  marker_path_ddp_.color.r = 1.0;
  marker_path_ddp_.color.g = 0.0;
  marker_path_ddp_.color.b = 0.0;
  marker_path_ddp_.lifetime = ros::Duration(0);

  //Marker for "start" text
  id++;
  marker_start_.header.frame_id = strfrm_world_;
  marker_start_.header.stamp = ros::Time();
  marker_start_.ns = "dsl_ddp_planner";
  marker_start_.id = id;
  marker_start_.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker_start_.action = visualization_msgs::Marker::ADD;
  marker_start_.text="S";
  marker_start_.scale.z = 4;
  marker_start_.color.a = 1.0; // Don't forget to set the alpha!
  marker_start_.color.r = 1.0;
  marker_start_.color.g = 0.0;
  marker_start_.color.b = 0.0;
  marker_start_.lifetime = ros::Duration(0);

  //Marker for "goal" text
  id++;
  marker_goal_.header.frame_id = strfrm_world_;
  marker_goal_.header.stamp = ros::Time();
  marker_goal_.ns = "dsl_ddp_planner";
  marker_goal_.id = id;
  marker_goal_.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker_goal_.action = visualization_msgs::Marker::ADD;
  marker_goal_.text="G";
  marker_goal_.scale.z = 4;
  marker_goal_.color.a = 1.0; // Don't forget to set the alpha!
  marker_goal_.color.r = 1.0;
  marker_goal_.color.g = 0.0;
  marker_goal_.color.b = 0.0;
  marker_goal_.lifetime = ros::Duration(0);

  //Marker for "start" sphere
  id++;
  marker_start_sphere_.header.frame_id = strfrm_world_;
  marker_start_sphere_.header.stamp = ros::Time();
  marker_start_sphere_.ns = "dsl_ddp_planner";
  marker_start_sphere_.id = id;
  marker_start_sphere_.type = visualization_msgs::Marker::SPHERE;
  marker_start_sphere_.action = visualization_msgs::Marker::ADD;
  marker_start_sphere_.text="start";
  marker_start_sphere_.scale.z = 10;
  marker_start_sphere_.color.a = 1.0; // Don't forget to set the alpha!
  marker_start_sphere_.color.r = 1.0;
  marker_start_sphere_.color.g = 0.0;
  marker_start_sphere_.color.b = 0.0;
  marker_start_sphere_.lifetime = ros::Duration(0);

  //Marker for "goal" sphere
  id++;
  marker_goal_sphere_.header.frame_id = strfrm_world_;
  marker_goal_sphere_.header.stamp = ros::Time();
  marker_goal_sphere_.ns = "dsl_ddp_planner";
  marker_goal_sphere_.id = id;
  marker_goal_sphere_.type = visualization_msgs::Marker::SPHERE;
  marker_goal_sphere_.action = visualization_msgs::Marker::ADD;
  marker_goal_sphere_.text="goal";
  marker_goal_sphere_.scale.z = 10;
  marker_goal_sphere_.color.a = 1.0; // Don't forget to set the alpha!
  marker_goal_sphere_.color.r = 1.0;
  marker_goal_sphere_.color.g = 0.0;
  marker_goal_sphere_.color.b = 0.0;
  marker_goal_sphere_.lifetime = ros::Duration(0);
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
  p_gdsl_ = new dsl::GridSearch(occ_grid_.info.width, occ_grid_.info.height, new dsl::DistEdgeCost(), map_dsl_);
  ros::Time t_end =  ros::Time::now();
  if(config_.dyn_debug_on)
  {
    cout<<"*Initialized DSL grid search object with map size:"<<occ_grid_.info.width<<" X "<<occ_grid_.info.height<<endl;
    cout<<"  delta t:"<<(t_end - t_start).toSec()<<" sec"<<endl;
  }
}

bool
CallBackDslDdp::dslFeasible()
{
  float w = occ_grid_.info.width;
  float h = occ_grid_.info.height;
  return w && h && cond_feas_s_ && cond_feas_g_;
}
void
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
      cout<<"*Planned a path and optimized it and obtained a path(with "<<path_opt_.count<<"nodes)"<<endl;
      cout<<"  delta t:"<<(t_end - t_start).toSec()<<" sec"<<endl;
    }
    dslInterpolate();
  }
  else
  {
    if(config_.dyn_debug_on)
    {
      cout<<"*Planning with DSL not possible because it's infeasible"<<endl;
    }
  }
}

void
CallBackDslDdp::dslInterpolate(void)
{

  float m_per_cell = occ_grid_.info.resolution;
  int n = path_opt_.count;
  //std::vector<double> x(path_opt_.count), y(path_opt_.count),d(path_opt_.count), t(path_opt_.count);
  VectorXd x(n),y(n);    x.setZero(); y.setZero();
  VectorXd delx(n),dely(n); delx.setZero(); dely.setZero();
  VectorXd delt(n),t(n);  delt.setZero(); t.setZero();

  for (int i = 0; i < n; i++)
  {
    x(i) = path_opt_.pos[2*i];
    y(i) = path_opt_.pos[2*i+1];
  }
  delx.tail(n-1) = x.tail(n-1) - x.head(n-1);
  dely.tail(n-1) = y.tail(n-1) - y.head(n-1);

  //The time difference between 2 consecutive way points
  //is calculated as distance between the 2 way points divided by the top speed
  delt = ((delx.array().square() + dely.array().square()).array().sqrt())*m_per_cell/config_.dyn_max_speed;
  for(int i=1;i<n;i++)
    t(i) =t(i-1) + delt(i);

  //Create spline interpolator
  SplineFunction intp_x(t,x,config_.dyn_dsl_interp_deg);
  SplineFunction intp_y(t,y,config_.dyn_dsl_interp_deg);

  //Interpolate
  int n_segs; n_segs = floor((double)(t(n-1))/config_.dyn_dsl_interp_delt);
  int n_nodes = n_segs+1;
  double dt= (double)(t(n-1))/(double)n_segs;

  x_dsl_intp_.resize(n_nodes); y_dsl_intp_.resize(n_nodes);a_dsl_intp_.resize(n_nodes); t_dsl_intp_.resize(n_nodes);
  x_dsl_intp_(0) = x(0);     y_dsl_intp_(0) = y(0);     t_dsl_intp_(0) = t(0);

  for(int i=1;i<n_nodes;i++)
  {
    t_dsl_intp_(i) = i*dt;
    x_dsl_intp_(i) = intp_x[i*dt];
    y_dsl_intp_(i) = intp_y[i*dt];

    double dx = x_dsl_intp_(i) -x_dsl_intp_(i-1);
    double dy = y_dsl_intp_(i) -y_dsl_intp_(i-1);
    a_dsl_intp_(i-1) = atan2(dy,dx);
  }
  a_dsl_intp_(n_nodes-1) = a_dsl_intp_(n_nodes-2);
}

bool
CallBackDslDdp::ddpPlan(void)
{
  if(config_.dyn_debug_on)
    cout<<"*Entering local path planning"<<endl;

  float m_per_cell = occ_grid_.info.resolution;

  //Select the ddp start and goal position
  //if dyn_ddp_from_curr_posn=true then start is current position else
  if(config_.dyn_ddp_from_curr_posn)
    pose_ddp_start_ = pose_curr_;
  else
    pose_ddp_start_ = pose_dsl_start_;

  //Select ddp goal position by finding a point on the dsl way point
  //  that is t_away sec away from current position

  // find nearest row(idx_nearest) on the [x_dsl_intp_ ,y_dsl_intp_] to posn_ddp_start_
  int n_nodes = x_dsl_intp_.size();
  Vector3d pt_ddp_start = (tfm_world2og_ll_.inverse()*pose_ddp_start_).translation()/m_per_cell;
  VectorXd dist_sq =(x_dsl_intp_ - VectorXd::Ones(n_nodes)*pt_ddp_start(0)).array().square()
                   +(y_dsl_intp_ - VectorXd::Ones(n_nodes)*pt_ddp_start(1)).array().square();
  VectorXd::Index idx_min; dist_sq.minCoeff(&idx_min);
  int idx_nearest = (int)idx_min;

  //set pose_ddp_goal_ to t_away sec ahead from t_dsl_intp_ at idx_close
  vector<double> t_stl(t_dsl_intp_.size()); Map<VectorXd>(t_stl.data(),t_dsl_intp_.size()) = t_dsl_intp_;
  double t_away = config_.dyn_ddp_t_away;
  vector<double>::iterator it_t_away = upper_bound(t_stl.begin(),t_stl.end(),t_stl[idx_nearest]+t_away);
  int idx_t_away = it_t_away-t_stl.begin()==t_stl.size()? t_stl.size()-1: it_t_away-t_stl.begin();

  pose_ddp_goal_ = tfm_world2og_ll_
                   *Translation3d(m_per_cell*x_dsl_intp_(idx_t_away),m_per_cell*y_dsl_intp_(idx_t_away),0)
                   *AngleAxisd(a_dsl_intp_(idx_t_away), Vector3d::UnitZ());

  //Prepare the service request
  //Number of trajectory segments
  traj_req_resp_.request.itreq.N = 32;
  //Number of iterations
  traj_req_resp_.request.itreq.Niter = 32;
  //set t0
  traj_req_resp_.request.itreq.t0 = 0;
  //set tf
  traj_req_resp_.request.itreq.tf = path_opt_.len * m_per_cell/config_.dyn_max_speed;
  //set x0
  Vector3d rpy_start; SO3::Instance().g2q(rpy_start,pose_ddp_start_.linear());
  Vector3d ypr_start = pose_ddp_start_.rotation().eulerAngles(2,1,0);
  traj_req_resp_.request.itreq.x0.statevector[0] = pose_ddp_start_.translation()(0);
  traj_req_resp_.request.itreq.x0.statevector[1] = pose_ddp_start_.translation()(1);
  traj_req_resp_.request.itreq.x0.statevector[2] = rpy_start(2);
  traj_req_resp_.request.itreq.x0.statevector[3] = 0;
  //set xf
  Vector3d rpy_goal; SO3::Instance().g2q(rpy_goal,pose_ddp_goal_.linear());
  Vector3d ypr_goal = pose_ddp_goal_.rotation().eulerAngles(2,1,0);
  traj_req_resp_.request.itreq.xf.statevector[0] = pose_ddp_goal_.translation()(0);
  traj_req_resp_.request.itreq.xf.statevector[1] = pose_ddp_goal_.translation()(1);
  traj_req_resp_.request.itreq.xf.statevector[2] = rpy_goal(2);
  traj_req_resp_.request.itreq.xf.statevector[3] = 0;

  cout<<"the start ypr:"<<ypr_start.transpose()<<endl;
  cout<<"the start rpy:"<<rpy_start.transpose()<<endl;

  cout<<"the goal ypr:"<<ypr_goal.transpose()<<endl;
  cout<<"the goal rpy:"<<rpy_goal.transpose()<<endl;

  bool req_success = srvcl_traj_.call(traj_req_resp_);

  if(req_success && config_.dyn_enable_motors)
  {
    cout<<"ctrl[0]"<<traj_req_resp_.response.traj.ctrl[0]<<endl;
    cout<<"ctrl[1]"<<traj_req_resp_.response.traj.ctrl[1]<<endl;
    double ctrl_vel_si(0);
    double ctrl_str_si(0);
    gcop_comm::Ctrl ctrl;
    ctrl.ctrlvec.push_back(ctrl_vel_si);
    ctrl.ctrlvec.push_back(ctrl_str_si);
  }


  if (req_success && config_.dyn_debug_on)
  {
    cout<<"*Response received and number of nodes is:"<<traj_req_resp_.response.traj.statemsg.size()<<endl;
    cout<<"  requested x0:"<<pose_ddp_start_.translation()(0)<<" , "<<pose_ddp_start_.translation()(1)
                           <<"  xf:"<<pose_ddp_goal_.translation()(0)<<" , "<<pose_ddp_goal_.translation()(1)
                           <<"  t0:"<<traj_req_resp_.request.itreq.t0<<"  tf:"<< traj_req_resp_.request.itreq.tf<<endl;
    cout<<"  Total path length:"<<path_opt_.len * m_per_cell<<endl;
  }
  else
  {
    cout<<"*Failed to call service optimization_requests"<<endl;
  }

  return req_success;
}

void
CallBackDslDdp::dispPathDdpRviz(void)
{
  if(config_.dyn_debug_on)
    cout<<"*Displaying ddp path"<<endl;


  float res = occ_grid_.info.resolution;
  marker_path_ddp_.points.clear();
  int n = traj_req_resp_.response.traj.statemsg.size();
  for (int i = 0; i < traj_req_resp_.response.traj.statemsg.size(); i++)
  {
    geometry_msgs::Point node;
    node.x    = traj_req_resp_.response.traj.statemsg[i].statevector[0];
    node.y    = traj_req_resp_.response.traj.statemsg[i].statevector[1];
    double th = traj_req_resp_.response.traj.statemsg[i].statevector[2];
    double v =  traj_req_resp_.response.traj.statemsg[i].statevector[3];
    node.z = 0.2;
    marker_path_ddp_.points.push_back(node);
    cout<<"  x"<<i<<":"<<node.x<<" , "<<node.y<<" , "<<th<<" , "<<v<<endl;
  }
  pub_vis_.publish( marker_path_ddp_ );
}

void
CallBackDslDdp::dispPathDslRviz()
{
  if(config_.dyn_debug_on)
    cout<<"*Displaying dsl path"<<endl;

  float m_per_cell = occ_grid_.info.resolution;
  marker_path_dsl_.points.clear();
  for (int i = 0; i < path_opt_.count; i++)
  {
    Vector3d posn_waypt_in_ll(m_per_cell*path_opt_.pos[2*i],m_per_cell*(path_opt_.pos[2*i+1]),0);
    Vector3d posn_waypt_in_world = tfm_world2og_ll_*posn_waypt_in_ll;
    geometry_msgs::Point node;
    node.x = posn_waypt_in_world(0);
    node.y = posn_waypt_in_world(1);
    node.z = 0.2;
    marker_path_dsl_.points.push_back(node);
  }
  pub_vis_.publish( marker_path_dsl_ );
}

void
CallBackDslDdp::dispPathDslInterpdRviz()
{
  if(config_.dyn_debug_on)
    cout<<"*Displaying interpolated dsl path"<<endl;

  float m_per_cell = occ_grid_.info.resolution;
  marker_path_dsl_intp_.points.clear();
  for (int i = 0; i < x_dsl_intp_.size(); i++)
  {
    Vector3d posn_waypt_in_ll(m_per_cell*x_dsl_intp_(i),m_per_cell*y_dsl_intp_(i),0);
    Vector3d posn_waypt_in_world = tfm_world2og_ll_*posn_waypt_in_ll;
    geometry_msgs::Point node;
    node.x = posn_waypt_in_world(0);
    node.y = posn_waypt_in_world(1);
    node.z = 0.2;
    marker_path_dsl_intp_.points.push_back(node);
  }
  pub_vis_.publish( marker_path_dsl_intp_ );
}

void
CallBackDslDdp::dispStartRviz()
{
  if(config_.dyn_debug_on)
    cout<<"*Displaying start markers"<<endl;

  marker_start_.pose.position.x =pose_dsl_start_.translation()(0);
  marker_start_.pose.position.y =pose_dsl_start_.translation()(1);
  marker_start_.pose.position.z =0.2;
  marker_start_sphere_.pose.position.x =pose_dsl_start_.translation()(0);
  marker_start_sphere_.pose.position.y =pose_dsl_start_.translation()(1);
  marker_start_sphere_.pose.position.z =0.2;

  pub_vis_.publish( marker_start_ );
  pub_vis_.publish( marker_start_sphere_ );
}

void
CallBackDslDdp::dispGoalRviz()
{
  if(config_.dyn_debug_on)
    cout<<"*Displaying goal markers"<<endl;

  marker_goal_.pose.position.x =pose_dsl_goal_.translation()(0);
  marker_goal_.pose.position.y =pose_dsl_goal_.translation()(1);
  marker_goal_.pose.position.z =0.2;

  marker_goal_sphere_.pose.position.x =pose_dsl_goal_.translation()(0);
  marker_goal_sphere_.pose.position.y =pose_dsl_goal_.translation()(1);
  marker_goal_sphere_.pose.position.z =0.2;

  pub_vis_.publish( marker_goal_ );
  pub_vis_.publish( marker_goal_sphere_ );
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

  double a=10;

  ros::Rate loop_rate(100);
  while(!g_shutdown_requested)
  {
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;

}
