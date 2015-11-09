//rampage_traj_plot_rviz.cpp
//Author: Subhransu Mishra
//Note: This ros node is used for displaying the gps, visual odometry
//      wheel odometry, satellite images and other log data 
//TODO: 1) it a bezier curve/ spline to the trajectory
//      2)

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

// ROS rampage messages
//#include "rampage_msgs/UavCmds.h"
#include "rampage_msgs/GpsSimple.h"
#include "rampage_msgs/WheelOdometry.h"

//gcop_comm msgs
#include <gcop_comm/State.h>
#include <gcop_comm/CtrlTraj.h>
#include <gcop_comm/Trajectory_req.h>

//ROS dynamic reconfigure
#include <dynamic_reconfigure/server.h>
#include <gcop_ctrl/DstarDdpPlannerConfig.h>

//Other includes
#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>

//D* Lite algorithm
#include <dsl/gridsearch.h>
#include <dsl/distedgecost.h>

//local includes
#include <yaml-cpp/yaml.h>
#include "yaml_eig_conv.h"

// Eigen Includes
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

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

class CallBackDstarDdp
{
public:
  typedef Matrix<float, 4, 4> Matrix4f;
  typedef Matrix<double, 4, 4> Matrix4d;

public:
  CallBackDstarDdp();
  ~CallBackDstarDdp();
private:
  ros::NodeHandle nh_, nh_p_;
  YAML::Node yaml_node_;

  string strtop_odom_, strtop_pose_start_, strtop_pose_goal_, strtop_og_;
  string strtop_diag_, strtop_marker_rviz_, strtop_og_dild_, strtop_traj_;
  string strfrm_map_, strfrm_robot_, strfrm_og_;

  ros::Subscriber sub_odom_, sub_pose_start_, sub_pose_goal_, sub_og_;
  ros::Publisher pub_diag_, pub_vis_, pub_og_dild_, pub_traj_;
  ros::Timer timer_vis_;
  ros::ServiceClient srvcl_traj_;

  tf::TransformBroadcaster tf_br_;
  tf::TransformListener tf_lr_;

  gcop_ctrl::DstarDdpPlannerConfig config_;
  dynamic_reconfigure::Server<gcop_ctrl::DstarDdpPlannerConfig> dyn_server_;
  visualization_msgs::Marker marker_path_,marker_path_lcl_;
  visualization_msgs::Marker marker_start_, marker_goal_, marker_start_sphere_, marker_goal_sphere_;
  nav_msgs::OccupancyGrid occ_grid_;
  cv::Mat img_occ_grid_dilated_;


  tf::Transform tfm_start_, tfm_goal_;
  tf::Point pt_start_, pt_goal_;
  tf::Point pt_start_lcl_, pt_goal_lcl_;
  tf::Transform tfm_map_to_occ_grid_bl_;
  dsl::GridSearch* p_gdsl_;
  double* map_dsl_;
  dsl::GridPath path_opt_;
  bool cond_feas_s_, cond_feas_g_;


  gcop_comm::Trajectory_req traj_req_resp_;

private:

  void cbReconfig(gcop_ctrl::DstarDdpPlannerConfig &config, uint32_t level);

  void cbOdom(const nav_msgs::OdometryConstPtr& msg_odom);
  void cbPoseStart(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg_pose_start);
  void cbPoseGoal(const geometry_msgs::PoseStampedConstPtr& msg_pose_goal);
  void cbOccGrid(const nav_msgs::OccupancyGridConstPtr& msg_occ_grid);

  void cbTimerVis(const ros::TimerEvent& event);
  void endMarker(void);

  void setupTopicsAndNames(void);
  void initSubsPubsAndTimers(void);
  void initRvizMarkers(void);

  void dispPathRviz(void);
  void dispPathLclRviz(void);
  void dispStartRviz(void);
  void dispGoalRviz(void);


  void dslInit(void);
  void dslDelete(void);
  void dslPlan(void);
  bool dslFeasible(void);

  bool lclPlan(void);


  void dilateObs(void);
  void setTfms(void);
  void imgToOccGrid(const cv::Mat& img,const geometry_msgs::Pose pose_org, const double res_m_per_pix,  nav_msgs::OccupancyGrid& occ_grid);

};

CallBackDstarDdp::CallBackDstarDdp():
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
    dynamic_reconfigure::Server<gcop_ctrl::DstarDdpPlannerConfig>::CallbackType dyn_cb_f;
    dyn_cb_f = boost::bind(&CallBackDstarDdp::cbReconfig, this, _1, _2);
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


CallBackDstarDdp::~CallBackDstarDdp()
{
  endMarker();
  dslDelete();
}

void
CallBackDstarDdp::setupTopicsAndNames(void)
{
  if(config_.dyn_debug_on)
      cout<<"setting up topic names"<<endl;

    // input topics
    strtop_odom_       = yaml_node_["strtop_odom"].as<string>();
    strtop_pose_start_ = yaml_node_["strtop_pose_start"].as<string>();
    strtop_pose_goal_  = yaml_node_["strtop_pose_goal"].as<string>();
    strtop_og_          = yaml_node_["strtop_og"].as<string>();

    // output topics
    strtop_diag_        = yaml_node_["strtop_diag"].as<string>();
    strtop_marker_rviz_ = yaml_node_["strtop_marker_rivz"].as<string>();
    strtop_og_dild_     = yaml_node_["strtop_og_dild"].as<string>();
    strtop_traj_       = yaml_node_["strtop_traj"].as<string>();

    strfrm_map_        = yaml_node_["strfrm_map"].as<string>();
    strfrm_robot_      = yaml_node_["strfrm_robot"].as<string>();
    strfrm_og_         = yaml_node_["strfrm_og"].as<string>();

    if(config_.dyn_debug_on)
    {
      cout<<"Topics are(put here):"<<endl;
    }
}

void
CallBackDstarDdp::initSubsPubsAndTimers(void)
{
  //Setup subscribers
  sub_odom_       = nh_.subscribe(strtop_odom_,1, &CallBackDstarDdp::cbOdom,this);
  sub_og_   = nh_.subscribe(strtop_og_,  1, &CallBackDstarDdp::cbOccGrid, this);
  sub_pose_start_ = nh_.subscribe(strtop_pose_start_, 1, &CallBackDstarDdp::cbPoseStart, this);
  sub_pose_goal_  = nh_.subscribe(strtop_pose_goal_, 1, &CallBackDstarDdp::cbPoseGoal, this);

  //Setup Publishers
  pub_diag_    = nh_.advertise<visualization_msgs::Marker>( strtop_diag_, 0 );
  pub_vis_     = nh_.advertise<visualization_msgs::Marker>( strtop_marker_rviz_, 0 );
  pub_og_dild_ = nh_.advertise<nav_msgs::OccupancyGrid>(strtop_og_dild_,0,true);
  pub_traj_    = nh_.advertise<gcop_comm::CtrlTraj>(strtop_traj_,0);

  //Setup timers
  timer_vis_ = nh_.createTimer(ros::Duration(0.1), &CallBackDstarDdp::cbTimerVis, this);
  timer_vis_.stop();
}

void
CallBackDstarDdp::cbReconfig(gcop_ctrl::DstarDdpPlannerConfig &config, uint32_t level)
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

    if(config.dyn_plan_dsl_global)
    {
      dslPlan();
      dispPathRviz();
      config.dyn_plan_dsl_global = false;
    }

    if(config.dyn_plan_ddp_local)
    {
      if(lclPlan())
        dispPathLclRviz();
      config.dyn_plan_ddp_local = false;
    }

  }
  else
    first_time = false;
}

void
CallBackDstarDdp::cbTimerVis(const ros::TimerEvent& event)
{
  pub_vis_.publish( marker_path_ );
}

void
CallBackDstarDdp::cbOdom(const nav_msgs::OdometryConstPtr& msg_odom)
{

}

void
CallBackDstarDdp::cbPoseStart(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg_pose_start)
{
  tf::poseMsgToTF(msg_pose_start->pose.pose, tfm_start_);
  pt_start_.setValue(msg_pose_start->pose.pose.position.x,msg_pose_start->pose.pose.position.y,msg_pose_start->pose.pose.position.z);
  dispStartRviz();

  pt_start_lcl_ = tfm_map_to_occ_grid_bl_.inverse()*pt_start_;
  float res = occ_grid_.info.resolution;
  int w = occ_grid_.info.width;
  int h = occ_grid_.info.height;
  int dsl_start_x = (int)(pt_start_lcl_.x()/res);
  int dsl_start_y = (int)(pt_start_lcl_.y()/res);
  cond_feas_s_ = (dsl_start_x>=0) && (dsl_start_y>=0) && (dsl_start_x<w) && (dsl_start_y<h);
  if(cond_feas_s_)
  {
    p_gdsl_->SetStart(dsl_start_x, dsl_start_y);

    if(config_.dyn_debug_on)
    {
      cout<<"*Start position received at ("<<pt_start_.getX()<<" , "<<pt_start_.getY()
          <<") in map frame and ("       <<dsl_start_x           <<" , "<<dsl_start_y
          <<") in image coordinate"<<endl;
    }
  }
  else
  {
    if(config_.dyn_debug_on)
    {
      cout<<"*Invalid Start position received at ("<<pt_start_.getX()<<" , "<<pt_start_.getY()
          <<") in map frame and ("       <<dsl_start_x           <<" , "<<dsl_start_y
          <<") in image coordinate"<<endl;
    }
  }
}

void
CallBackDstarDdp::cbPoseGoal(const geometry_msgs::PoseStampedConstPtr& msg_pose_goal)
{
  tf::poseMsgToTF(msg_pose_goal->pose, tfm_goal_);
  pt_goal_.setValue(msg_pose_goal->pose.position.x,msg_pose_goal->pose.position.y,msg_pose_goal->pose.position.z);
  dispGoalRviz();

  pt_goal_lcl_ = tfm_map_to_occ_grid_bl_.inverse()*pt_goal_;
  float res = occ_grid_.info.resolution;
  int w = occ_grid_.info.width;
  int h = occ_grid_.info.height;
  int dsl_goal_x = (int)(pt_goal_lcl_.x()/res);
  int dsl_goal_y = (int)(pt_goal_lcl_.y()/res);

  cond_feas_g_ = (dsl_goal_x>=0) && (dsl_goal_y>=0) && (dsl_goal_x<w) && (dsl_goal_y<h);
  if(cond_feas_g_)
  {
    p_gdsl_->SetGoal(dsl_goal_x, dsl_goal_y);
    if(config_.dyn_debug_on)
    {
      cout<<"*Goal position received at ("<<pt_goal_.getX()<<" , "<<pt_goal_.getY()
               <<") in map frame and ("       <<dsl_goal_x           <<" , "<<dsl_goal_y
               <<") in image coordinate"<<endl;
    }
  }
  else
  {
    if(config_.dyn_debug_on)
    {
      cout<<"*Invalid Goal position received at ("<<pt_start_.getX()<<" , "<<pt_goal_.getY()
                     <<") in map frame and ("       <<dsl_goal_x           <<" , "<<dsl_goal_y
                     <<") in image coordinate"<<endl;
    }
  }
}

void
CallBackDstarDdp::cbOccGrid(const nav_msgs::OccupancyGridConstPtr& msg_occ_grid)
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
CallBackDstarDdp::imgToOccGrid(const cv::Mat& img,const geometry_msgs::Pose pose_org, const double res_m_per_pix,  nav_msgs::OccupancyGrid& occ_grid)
{
  //confirm that the image is grayscale
  assert(img.channels()==1 && img.type()==0);

  //set occgrid header
  occ_grid.header.frame_id="/map";
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
CallBackDstarDdp::setTfms(void)
{
  if(config_.dyn_debug_on)
  {
    cout<<"*Setting transformation for map to occ grid bottom left corner."<<endl;
  }
  //get the transformation between map to bottom left of the occupancy grid
  geometry_msgs::Pose pose_org =  occ_grid_.info.origin;
  tf::StampedTransform tfms_map_to_map1;
  tf_lr_.waitForTransform("map","map1",ros::Time(0),ros::Duration(0.1));
  tf_lr_.lookupTransform("/map", "/map1", ros::Time(0), tfms_map_to_map1);
  tf::Transform tfm_map1_to_bot_left; tf::poseMsgToTF(pose_org,tfm_map1_to_bot_left);
  tfm_map_to_occ_grid_bl_ = tfms_map_to_map1*tfm_map1_to_bot_left;
}

void
CallBackDstarDdp::dilateObs(void)
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
  geometry_msgs::Pose pose_org;tf::poseTFToMsg(tfm_map_to_occ_grid_bl_,pose_org);
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
CallBackDstarDdp::endMarker(void)
{
//remove visualization marker
marker_path_.action     = visualization_msgs::Marker::DELETE;
marker_path_lcl_.action = visualization_msgs::Marker::DELETE;
marker_start_.action    = visualization_msgs::Marker::DELETE;
marker_goal_.action     = visualization_msgs::Marker::DELETE;
pub_vis_.publish( marker_path_ );
pub_vis_.publish( marker_path_lcl_ );
pub_vis_.publish( marker_start_ );
pub_vis_.publish( marker_goal_ );
}

void
CallBackDstarDdp::initRvizMarkers(void)
{
  if(config_.dyn_debug_on)
  {
    cout<<"*Initializing all rviz markers."<<endl;
  }
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

  //Marker for local path
  id++;
  marker_path_lcl_.header.frame_id = "map";
  marker_path_lcl_.header.stamp = ros::Time();
  marker_path_lcl_.ns = "rampage";
  marker_path_lcl_.id = id;
  marker_path_lcl_.type = visualization_msgs::Marker::LINE_STRIP;
  marker_path_lcl_.action = visualization_msgs::Marker::ADD;
  marker_path_lcl_.scale.x = 0.4;
  marker_path_lcl_.color.a = 0.5; // Don't forget to set the alpha!
  marker_path_lcl_.color.r = 1.0;
  marker_path_lcl_.color.g = 0.0;
  marker_path_lcl_.color.b = 0.0;
  marker_path_lcl_.lifetime = ros::Duration(0);

  //Marker for "start" text
  id++;
  marker_start_.header.frame_id = "map";
  marker_start_.header.stamp = ros::Time();
  marker_start_.ns = "rampage";
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
  marker_goal_.header.frame_id = "map";
  marker_goal_.header.stamp = ros::Time();
  marker_goal_.ns = "rampage";
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
   marker_start_sphere_.header.frame_id = "map";
   marker_start_sphere_.header.stamp = ros::Time();
   marker_start_sphere_.ns = "rampage";
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
   marker_goal_sphere_.header.frame_id = "map";
   marker_goal_sphere_.header.stamp = ros::Time();
   marker_goal_sphere_.ns = "rampage";
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
CallBackDstarDdp::dslDelete()
{
  delete[] map_dsl_;
  delete p_gdsl_;
}

void
CallBackDstarDdp::dslInit()
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
CallBackDstarDdp::dslFeasible()
{
  float w = occ_grid_.info.width;
  float h = occ_grid_.info.height;
  return w && h && cond_feas_s_ && cond_feas_g_;
}
void
CallBackDstarDdp::dslPlan()
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
  }
  else
  {
    if(config_.dyn_debug_on)
    {
      cout<<"*Planning with DSL not possible because it's infeasible"<<endl;
    }
  }
}

bool
CallBackDstarDdp::lclPlan(void)
{
  if(config_.dyn_debug_on)
  {
    cout<<"*Entering local path planning"<<endl;
    cout<<"  number of points along the path:"<<path_opt_.count<<endl;
  }
  float res = occ_grid_.info.resolution;//m/cell

  int i=0;//first point of D* path
  tf::Point pt_lcl_0(res*path_opt_.pos[2*i],res*(path_opt_.pos[2*i+1]),0);
  tf::Point pt_0 = tfm_map_to_occ_grid_bl_*pt_lcl_0;

  i=path_opt_.count-1; //lst point of D* path
  tf::Point pt_lcl_f(res*path_opt_.pos[2*i],res*(path_opt_.pos[2*i+1]),0);
  tf::Point pt_f = tfm_map_to_occ_grid_bl_*pt_lcl_f;

  //Number of trajectory segments
  traj_req_resp_.request.itreq.N = 32;

  //Number of iterations
  traj_req_resp_.request.itreq.Niter = 32;

  //set t0
  traj_req_resp_.request.itreq.t0 = 0;

  //set tf
  traj_req_resp_.request.itreq.tf = path_opt_.len * res/config_.dyn_max_speed;

  //set x0
  traj_req_resp_.request.itreq.x0.statevector[0] = tfm_start_.getOrigin().getX();
  traj_req_resp_.request.itreq.x0.statevector[1] = tfm_start_.getOrigin().getY();
  traj_req_resp_.request.itreq.x0.statevector[2] = tfm_start_.getRotation().getAngle()*sgn(tfm_start_.getRotation().getZ());
  traj_req_resp_.request.itreq.x0.statevector[3] = 0;

  //set xf
  traj_req_resp_.request.itreq.xf.statevector[0] = tfm_goal_.getOrigin().getX();
  traj_req_resp_.request.itreq.xf.statevector[1] = tfm_goal_.getOrigin().getY();
  traj_req_resp_.request.itreq.xf.statevector[2] = tfm_goal_.getRotation().getAngle()*sgn(tfm_goal_.getRotation().getZ());
  traj_req_resp_.request.itreq.xf.statevector[3] = 0;


  bool req_success = srvcl_traj_.call(traj_req_resp_);
  if (req_success)
  {
    cout<<"*Response received and number of nodes is:"<<traj_req_resp_.response.traj.statemsg.size()<<endl;
    cout<<"  requested x0:"<<tfm_start_.getOrigin().getX()<<" , "<<tfm_start_.getOrigin().getY()
                       <<"  xf:"<<tfm_goal_.getOrigin().getX()<<" , "<<tfm_goal_.getOrigin().getY()
                       <<"  t0:"<<traj_req_resp_.request.itreq.t0<<"  tf:"<< traj_req_resp_.request.itreq.tf<<endl;
    cout<<"  Total path length:"<<path_opt_.len * res<<endl;
  }
  else
  {
    cout<<"*Failed to call service optimization_requests"<<endl;
  }

  return req_success;
}
void
CallBackDstarDdp::dispPathLclRviz(void)
{
  cout<<"*Displaying local path"<<endl;
  float res = occ_grid_.info.resolution;
  marker_path_lcl_.points.clear();
  int n = traj_req_resp_.response.traj.statemsg.size();
  for (int i = 0; i < traj_req_resp_.response.traj.statemsg.size(); i++)
  {
    geometry_msgs::Point node;
    node.x    = traj_req_resp_.response.traj.statemsg[i].statevector[0];
    node.y    = traj_req_resp_.response.traj.statemsg[i].statevector[1];
    double th = traj_req_resp_.response.traj.statemsg[i].statevector[2];
    double v =  traj_req_resp_.response.traj.statemsg[i].statevector[3];
    node.z = 0.2;
    marker_path_lcl_.points.push_back(node);
    cout<<"  x"<<i<<":"<<node.x<<" , "<<node.y<<" , "<<th<<" , "<<v<<endl;
  }
  pub_vis_.publish( marker_path_lcl_ );
}

void
CallBackDstarDdp::dispPathRviz()
{
  float res = occ_grid_.info.resolution;
  marker_path_.points.clear();
  for (int i = 0; i < path_opt_.count; i++)
  {
    tf::Point pt_lcl(res*path_opt_.pos[2*i],res*(path_opt_.pos[2*i+1]),0);
    tf::Point pt = tfm_map_to_occ_grid_bl_*pt_lcl;
    geometry_msgs::Point node;
    node.x = pt.getX();node.y = pt.getY();node.z = 0.2;
    marker_path_.points.push_back(node);
  }
  pub_vis_.publish( marker_path_ );
}

void
CallBackDstarDdp::dispStartRviz()
{
  marker_start_.pose.position.x =pt_start_.getX();
  marker_start_.pose.position.y =pt_start_.getY();
  marker_start_.pose.position.z =0.2;

  marker_start_sphere_.pose.position.x =pt_start_.getX();
  marker_start_sphere_.pose.position.y =pt_start_.getY();
  marker_start_sphere_.pose.position.z =0.2;
  pub_vis_.publish( marker_start_ );
  pub_vis_.publish( marker_start_sphere_ );
}

void
CallBackDstarDdp::dispGoalRviz()
{
  marker_goal_.pose.position.x =pt_goal_.getX();
  marker_goal_.pose.position.y =pt_goal_.getY();
  marker_goal_.pose.position.z =0.2;

  marker_goal_sphere_.pose.position.x =pt_goal_.getX();
  marker_goal_sphere_.pose.position.y =pt_goal_.getY();
  marker_goal_sphere_.pose.position.z =0.2;
  pub_vis_.publish( marker_goal_ );
  pub_vis_.publish( marker_goal_sphere_ );
}


//------------------------------------------------------------------------
//--------------------------------MAIN -----------------------------------
//------------------------------------------------------------------------

int main(int argc, char** argv)
{
  ros::init(argc,argv,"rampage_traj_plot_rviz",ros::init_options::NoSigintHandler);
  signal(SIGINT,mySigIntHandler);

  ros::NodeHandle nh;

  CallBackDstarDdp cbc;

  double a=10;

  ros::Rate loop_rate(100);
  while(!g_shutdown_requested)
  {
    ros::spinOnce();
    loop_rate.sleep();
  }
  return 0;

}
