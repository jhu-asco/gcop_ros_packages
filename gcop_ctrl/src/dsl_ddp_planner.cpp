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

//ROS dynamic reconfigure
#include <dynamic_reconfigure/server.h>
#include <gcop_ctrl/DslDdpPlannerConfig.h>

//yaml
#include <yaml-cpp/yaml.h>

//gcop ros utils
#include <gcop_ros_utils/eigen_ros_conv.h>
#include <gcop_ros_utils/eig_splinterp.h>
#include <gcop_ros_utils/yaml_eig_conv.h>

// Eigen Includes
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/Splines>

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
#include <dsl/grid2d.h>
#include <dsl/gridcost.h>
#include <dsl/grid2dconnectivity.h>
#include <dsl/cargrid.h>
#include <dsl/carcost.h>
#include <dsl/carconnectivity.h>
#include <dsl/utils.h>

//gcop include
#include <gcop/so3.h>
#include <gcop/lqcost.h>
#include <gcop/gcar.h>
#include <gcop/utils.h>
#include <gcop/se2.h>
#include <gcop/ddp.h>



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


//to be used in switch case statement involving strings
constexpr unsigned int str2int(const char* str, int h = 0)
{
  return !str[h] ? 5381 : (str2int(str, h+1)*33) ^ str[h];
}



void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs)
{
    blobs.clear();

    // Fill the label_image with the blobs
    // 0  - background
    // 1  - unlabelled foreground
    // 2+ - labelled foreground

    cv::Mat label_image;
    binary.convertTo(label_image, CV_32SC1);

    int label_count = 2; // starts at 2 because 0,1 are used already

    for(int y=0; y < label_image.rows; y++) {
        int *row = (int*)label_image.ptr(y);
        for(int x=0; x < label_image.cols; x++) {
            if(row[x] != 1) {
                continue;
            }

            cv::Rect rect;
            cv::floodFill(label_image, cv::Point(x,y), label_count, &rect, 0, 0, 4);

            std::vector <cv::Point2i> blob;

            for(int i=rect.y; i < (rect.y+rect.height); i++) {
                int *row2 = (int*)label_image.ptr(i);
                for(int j=rect.x; j < (rect.x+rect.width); j++) {
                    if(row2[j] != label_count) {
                        continue;
                    }

                    blob.push_back(cv::Point2i(j,i));
                }
            }

            blobs.push_back(blob);

            label_count++;
        }
    }
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

private:

  string indStr(void);
  string indStr(int count);

  void cbReconfig(gcop_ctrl::DslDdpPlannerConfig &config, uint32_t level);

  void cbOdom(const nav_msgs::OdometryConstPtr& msg_odom);
  void cbPoseStart(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg_pose_start);
  void cbPoseGoal(const geometry_msgs::PoseStampedConstPtr& msg_pose_goal);
  void cbOccGrid(const nav_msgs::OccupancyGridConstPtr& msg_occ_grid);

  void cbTimerVis(const ros::TimerEvent& event);
  void cbTimerDsl(const ros::TimerEvent& event);
  void cbTimerDdp(const ros::TimerEvent& event);


  void setupTopicsAndNames(void);
  void initSubsPubsAndTimers(void);

  void rvizColorMsgEdit(std_msgs::ColorRGBA& rgba_msg, VectorXd& rgba_vec);
  void rvizMarkersEdit(visualization_msgs::Marker& marker, VectorXd& prop);
  void rvizMarkersInit(void);
  void rvizShowStart(void);
  void rvizShowGoal(void);
  void rvizShowPathDsl(void);
  void rvizShowPathDdp(void);
  void rvizShowPathDslInterpd(void);
  void rvizRemoveCar(void);
  void rvizRemoveStart(void);
  void rvizRemoveGoal(void);
  void rvizRemovePathDdp(void);
  void rvizRemovePathDsl(void);
  void rvizRemovePathDslInterpd(void);

  void dslInit(void);
  void dslDelete(void);
  bool dslPlan(void);
  bool dslFeasible(void);
  void dslInterpolate(void);

  bool ddpFeasible(void);
  bool ddpInit(void);
  bool ddpPlan(void);

  void setTfmsWorld2OgLL(void);

  void occGridProcessAndPub(void);
  void occGridDilateAndFilterUnseen(const nav_msgs::OccupancyGrid& og_original, nav_msgs::OccupancyGrid& og_dild_fild);
  void occGridResize(const nav_msgs::OccupancyGrid& og_dild_fild, nav_msgs::OccupancyGrid& og_final);
  void occGridFromImg(const cv::Mat& img,const geometry_msgs::Pose pose_org,
                      const double res_m_per_pix,  nav_msgs::OccupancyGrid& occ_grid);
public:
  ros::Rate loop_rate_main_;

private:
  ros::NodeHandle nh_, nh_p_;
  YAML::Node yaml_node_;
  gcop_ctrl::DslDdpPlannerConfig config_;
  dynamic_reconfigure::Server<gcop_ctrl::DslDdpPlannerConfig> dyn_server_;
  int ind_count_;
  string str_ind_;

  string strtop_odom_, strtop_pose_start_, strtop_pose_goal_, strtop_og_;
  string strtop_diag_, strtop_marker_rviz_, strtop_og_dild_, strtop_ctrl_;
  string strfrm_world_, strfrm_robot_, strfrm_og_org_;

  ros::Subscriber sub_odom_, sub_pose_start_, sub_pose_goal_, sub_og_;
  ros::Publisher pub_diag_, pub_vis_, pub_og_final_, pub_ctrl_;
  ros::Timer timer_vis_, timer_ddp_, timer_dsl_;

  tf::TransformListener tf_lr_;

  visualization_msgs::Marker marker_path_dsl_,marker_wp_dsl_;
  visualization_msgs::Marker marker_path_dsl_intp_ ,marker_wp_dsl_intp_;
  VectorXd prop_path_pve_ddp_, prop_path_nve_ddp_, prop_wp_pve_ddp_, prop_wp_nve_ddp_ ;
  visualization_msgs::Marker marker_path_ddp_,marker_wp_ddp_;
  visualization_msgs::Marker marker_text_start_, marker_text_goal_;
  visualization_msgs::Marker marker_car_;
  nav_msgs::OccupancyGrid og_original_, og_final_;
  cv::Mat img_og_final_;

  // Frames and transformation
  Affine3d tfm_world2og_org_, tfm_world2og_ll_,tfm_og_org2og_ll_;
  Transform2d tfm_world2og_ll_2d_;

  // DSL vars
  double* p_dsl_map_;
  //   dsl with 2d planner
  dsl::GridCost<2>*        p_dsl_cost2d_;
  dsl::Grid2d*             p_dsl_grid2d_;
  dsl::Grid2dConnectivity* p_dsl_conn2d_;
  dsl::GridSearch<2>*      p_dsl_search2d_;
  //   dsl with geometric car
  dsl::CarCost*                 p_dsl_costcar_;
  dsl::CarGrid*                 p_dsl_gridcar_;
  dsl::CarConnectivity*         p_dsl_conncar_;
  dsl::GridSearch<3, Matrix3d>* p_dsl_searchcar_;
  vector<Vector3d>              pathaxy_ll_opt_;
  bool dsl_cond_feas_s_, dsl_cond_feas_g_, dsl_done_, dsl_use_geom_car_;
  VectorXd x_ll_intp_, y_ll_intp_,a_ll_intp_, t_dsl_intp_;//x,y,angle,t of the path
  Affine3d pose_world_dsl_start_, pose_world_dsl_goal_;
  Vector3d se2_dsl_start_, se2_dsl_goal_;//refers to grid point

  // DDP vars
  bool ddp_debug_on_;
  double ddp_mu_;
  int ddp_nseg_max_,ddp_nseg_min_, ddp_nit_max_;
  double ddp_tseg_ideal_;
  double ddp_tol_rel_, ddp_tol_abs_, ddp_tol_goal_m_;
  Affine3d pose_ddp_start_, pose_ddp_goal_, pose_ddp_curr_;
  ros::Time time_ddp_start_, time_ddp_curr_;
  double vel_ddp_start_,     vel_ddp_goal_,  vel_ddp_curr_;
  Vector3d pt_ddp_start_, pt_ddp_goal_; //refers to grid point

  Gcar sys_gcar_;
  LqCost< GcarState, 4, 2> cost_lq_;
  vector<double> ddp_ts_;
  vector<GcarState> ddp_xs_;
  vector<Vector2d> ddp_us_;

};

CallBackDslDdp::CallBackDslDdp():
                        nh_p_("~"),
                        loop_rate_main_(1000),
                        ind_count_(-1),
                        dsl_cond_feas_s_(false),
                        dsl_cond_feas_g_(false),
                        dsl_done_(false),
                        p_dsl_map_(nullptr),
                        p_dsl_cost2d_(nullptr),
                        p_dsl_grid2d_(nullptr),
                        p_dsl_conn2d_(nullptr),
                        p_dsl_search2d_(nullptr),
                        p_dsl_costcar_(nullptr),
                        p_dsl_gridcar_(nullptr),
                        p_dsl_conncar_(nullptr),
                        p_dsl_searchcar_(nullptr),
                        sys_gcar_(),
                        cost_lq_(sys_gcar_, 1, GcarState(Matrix3d::Identity(), 0))
{
  ind_count_++;
  cout<<"**************************************************************************"<<endl;
  cout<<"***************************DSL-DDP TRAJECTORY PLANNER*********************"<<endl;
  cout<<"*Entering constructor of cbc"<<endl;

  //Setup YAML reading and parsing
  string strfile_params;nh_p_.getParam("strfile_params",strfile_params);
  cout<<indStr(1)+"loading yaml param file into yaml_node"<<endl;
  yaml_node_ = YAML::LoadFile(strfile_params);
  str_ind_ = yaml_node_["str_ind"].as<string>();

  //setup dynamic reconfigure
  dynamic_reconfigure::Server<gcop_ctrl::DslDdpPlannerConfig>::CallbackType dyn_cb_f;
  dyn_cb_f = boost::bind(&CallBackDslDdp::cbReconfig, this, _1, _2);
  dyn_server_.setCallback(dyn_cb_f);

  // Setup topic names
  setupTopicsAndNames();

  //Setup Subscriber, publishers and Timers
  initSubsPubsAndTimers();

  //Setup rviz markers
  rvizMarkersInit();

  //Set gcar properties
  sys_gcar_.l = yaml_node_["gcar_l"].as<double>();
  sys_gcar_.r = yaml_node_["gcar_r"].as<double>();

  //init ddp planner
  ddpInit();

  cout<<indStr(1)+"Waiting for start and goal position."<<endl;
  cout<<indStr(1)+"Select Start through Publish Point button and select goal through 2D nav goal."<<endl;

  ind_count_--;
}


CallBackDslDdp::~CallBackDslDdp()
{
  rvizRemovePathDdp();
  rvizRemovePathDsl();
  rvizRemovePathDslInterpd();
  rvizRemoveCar();
  rvizRemoveStart();
  rvizRemoveGoal();
  dslDelete();
  cv::destroyAllWindows();
}
string
CallBackDslDdp::indStr()
{
  string ind;
  for(int i=0; i<ind_count_; i++)
    ind = ind+ str_ind_;
  return ind;
}

string
CallBackDslDdp::indStr(int count_extra)
{
  string ind;
  for(int i=0; i<ind_count_+count_extra; i++)
    ind = ind+ str_ind_;
  return ind;
}

void
CallBackDslDdp::cbReconfig(gcop_ctrl::DslDdpPlannerConfig &config, uint32_t level)
{
  ind_count_++;
  static bool first_time=true;

  if(!first_time)
  {
    if(config_.dyn_debug_on)
      cout<<indStr(0)+"*Dynamic reconfigure called"<<endl;

    //Change in dilation
    bool condn_dilate =     og_final_.info.width
        && (   config.dyn_dilation_obs_m != config_.dyn_dilation_obs_m
            || config.dyn_dilation_type != config_.dyn_dilation_type );
    if(condn_dilate)
    {
      config_.dyn_dilation_obs_m = config.dyn_dilation_obs_m;
      config_.dyn_dilation_type = config.dyn_dilation_type;
      occGridProcessAndPub();
    }

    //loop rate setting
    if(config_.dyn_loop_rate_main != config.dyn_loop_rate_main)
      loop_rate_main_= ros::Rate(config.dyn_loop_rate_main);

    //dsl settings
    if(config.dyn_dsl_plan_once)
    {
      if(dslPlan() && config.dyn_dsl_disp_rviz)
      {
        rvizShowPathDsl();
        rvizShowPathDslInterpd();
      }
      else
      {
        rvizRemovePathDsl();
        rvizRemovePathDslInterpd();
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
        rvizShowPathDdp();
      else
        rvizRemovePathDdp();

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
    cout<<indStr(0)+"*First call from dynamic reconfigure. Setting config from yaml"<<endl;

    //general settings
    config.dyn_debug_on           = yaml_node_["debug_on"].as<bool>();
    config.dyn_debug_verbose_on   = yaml_node_["debug_verbose_on"].as<bool>();
    config.dyn_send_gcar_ctrl     = yaml_node_["send_gcar_ctrl"].as<bool>();
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
  ind_count_--;
}

void
CallBackDslDdp::setupTopicsAndNames(void)
{
  ind_count_++;

  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*setting up topic names"<<endl;

  // Input topics
  strtop_odom_       = yaml_node_["strtop_odom"].as<string>();
  strtop_pose_start_ = yaml_node_["strtop_pose_start"].as<string>();
  strtop_pose_goal_  = yaml_node_["strtop_pose_goal"].as<string>();
  strtop_og_          = yaml_node_["strtop_og"].as<string>();

  // output topics
  strtop_diag_        = yaml_node_["strtop_diag"].as<string>();
  strtop_marker_rviz_ = yaml_node_["strtop_marker_rviz"].as<string>();
  strtop_og_dild_     = yaml_node_["strtop_og_dild"].as<string>();
  strtop_ctrl_       = yaml_node_["strtop_ctrl"].as<string>();

  // Frames
  strfrm_world_        = yaml_node_["strfrm_world"].as<string>();
  strfrm_robot_      = yaml_node_["strfrm_robot"].as<string>();
  strfrm_og_org_         = yaml_node_["strfrm_og_org"].as<string>();

  ind_count_--;
}

void
CallBackDslDdp::initSubsPubsAndTimers(void)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Initializing subs pubs and timers"<<endl;

  //Setup subscribers
  sub_odom_       = nh_.subscribe(strtop_odom_,1, &CallBackDslDdp::cbOdom,this);
  sub_og_   = nh_.subscribe(strtop_og_,  1, &CallBackDslDdp::cbOccGrid, this);
  sub_pose_start_ = nh_.subscribe(strtop_pose_start_, 1, &CallBackDslDdp::cbPoseStart, this);
  sub_pose_goal_  = nh_.subscribe(strtop_pose_goal_, 1, &CallBackDslDdp::cbPoseGoal, this);

  //Setup Publishers
  pub_diag_    = nh_.advertise<visualization_msgs::Marker>( strtop_diag_, 0 );
  pub_vis_     = nh_.advertise<visualization_msgs::Marker>( strtop_marker_rviz_, 0 );
  pub_og_final_ = nh_.advertise<nav_msgs::OccupancyGrid>(strtop_og_dild_,0,true);
  pub_ctrl_    = nh_.advertise<gcop_comm::GcarCtrl>(strtop_ctrl_,0);

  //Setup timers
  timer_vis_ = nh_.createTimer(ros::Duration(0.1), &CallBackDslDdp::cbTimerVis, this);
  timer_vis_.stop();

  timer_dsl_ = nh_.createTimer(ros::Duration(config_.dyn_dsl_loop_durn), &CallBackDslDdp::cbTimerDsl, this);
  timer_dsl_.stop();

  timer_ddp_ = nh_.createTimer(ros::Duration(config_.dyn_ddp_loop_durn), &CallBackDslDdp::cbTimerDdp, this);
  timer_ddp_.stop();
  ind_count_--;
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
    rvizShowPathDsl();
    rvizShowPathDslInterpd();
  }
  else
  {
    rvizRemovePathDsl();
    rvizRemovePathDslInterpd();
  }
}

void
CallBackDslDdp::cbTimerDdp(const ros::TimerEvent& event)
{
  if(ddpPlan() && config_.dyn_ddp_disp_rviz)
    rvizShowPathDdp();
  else
    rvizRemovePathDdp();
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
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Initial pose received from rviz"<<endl;

  float wpix = og_final_.info.resolution;//width of pixel in meter
  float hpix = og_final_.info.resolution;//height of pixel in meter
  int w = og_final_.info.width;
  int h = og_final_.info.height;

  poseMsg2Eig(pose_world_dsl_start_,msg_pose_start->pose.pose);
  Affine3d pose_ll_dsl_start = tfm_world2og_ll_.inverse()*pose_world_dsl_start_;
  Vector3d rpy_start; SO3::Instance().g2q(rpy_start,pose_ll_dsl_start.rotation().matrix());
  Vector3d axy_ll_dsl_start = Vector3d(rpy_start(2),pose_ll_dsl_start.translation()(0),pose_ll_dsl_start.translation()(1));

  rvizShowStart();
  dsl_cond_feas_s_ = (axy_ll_dsl_start(1)>=0) && (axy_ll_dsl_start(2)>=0) && (axy_ll_dsl_start(1)<w*wpix) && (axy_ll_dsl_start(2)<h*hpix);
  if(dsl_cond_feas_s_){
    if(dsl_use_geom_car_){
      axy_ll_dsl_start(0) = axy_ll_dsl_start(0) > p_dsl_gridcar_->xub(0)? axy_ll_dsl_start(0)-2*M_PI:axy_ll_dsl_start(0);
      axy_ll_dsl_start(0) = axy_ll_dsl_start(0) < p_dsl_gridcar_->xlb(0)? axy_ll_dsl_start(0)+2*M_PI:axy_ll_dsl_start(0);
      dsl_cond_feas_s_ = p_dsl_searchcar_->SetStart(axy_ll_dsl_start);
    }else
      dsl_cond_feas_s_ = p_dsl_search2d_->SetStart(Vector2d(axy_ll_dsl_start(1),axy_ll_dsl_start(2)));
  }


  IOFormat iof(StreamPrecision,0,", ",",\n",indStr(3)+"","","","");

  if(dsl_cond_feas_s_ && config_.dyn_debug_on)
  {
    cout<<indStr(1)+"The start position is valid"<<endl;
    if(config_.dyn_debug_verbose_on)
    {
      cout<<indStr(2)+"In world frame:\n"
          <<pose_world_dsl_start_.affine().format(iof)<<endl;
      cout<<indStr(2)+"In ll frame axy:\n"
          <<axy_ll_dsl_start.transpose().format(iof)<<endl;
    }
  }

  if(!dsl_cond_feas_s_ && config_.dyn_debug_on)
  {
    cout<<indStr(1)+"The start position is invalid"<<endl;
    if(config_.dyn_debug_verbose_on)
    {
      cout<<indStr(2)+"In world frame:\n"
          <<pose_world_dsl_start_.affine().format(iof)<<endl;
      cout<<indStr(2)+"In ll frame axy:\n"
          <<axy_ll_dsl_start.transpose().format(iof)<<endl;
    }
  }

  ind_count_--;
}


void
CallBackDslDdp::cbPoseGoal(const geometry_msgs::PoseStampedConstPtr& msg_pose_goal)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Goal pose received from rviz"<<endl;

  float wpix = og_final_.info.resolution;//width of pixel in meters
  float hpix = og_final_.info.resolution;//height of pixel in meters
  int w = og_final_.info.width;
  int h = og_final_.info.height;

  poseMsg2Eig(pose_world_dsl_goal_,msg_pose_goal->pose);
  Affine3d pose_ll_dsl_goal = tfm_world2og_ll_.inverse()*pose_world_dsl_goal_;
  Vector3d rpy_goal; SO3::Instance().g2q(rpy_goal,pose_ll_dsl_goal.rotation().matrix());
  Vector3d axy_ll_dsl_goal = Vector3d(rpy_goal(2),pose_ll_dsl_goal.translation()(0),pose_ll_dsl_goal.translation()(1));

  rvizShowGoal();
  dsl_cond_feas_g_ = (axy_ll_dsl_goal(1)>=0) && (axy_ll_dsl_goal(2)>=0) && (axy_ll_dsl_goal(1)<w*wpix) && (axy_ll_dsl_goal(2)<h*hpix);

  if(dsl_cond_feas_g_){
    if(dsl_use_geom_car_){
      axy_ll_dsl_goal(0) = axy_ll_dsl_goal(0) > p_dsl_gridcar_->xub(0)? axy_ll_dsl_goal(0)-2*M_PI:axy_ll_dsl_goal(0);
      axy_ll_dsl_goal(0) = axy_ll_dsl_goal(0) < p_dsl_gridcar_->xlb(0)? axy_ll_dsl_goal(0)+2*M_PI:axy_ll_dsl_goal(0);
      dsl_cond_feas_g_ = p_dsl_searchcar_->SetGoal(axy_ll_dsl_goal);
    }else
      dsl_cond_feas_g_ = p_dsl_search2d_->SetGoal(Vector2d(axy_ll_dsl_goal(1),axy_ll_dsl_goal(2)));

  }

  IOFormat iof(StreamPrecision,0,", ",",\n",indStr(3)+"","","","");
  if(dsl_cond_feas_g_ && config_.dyn_debug_on)
  {
    cout<<indStr(1)+"The goal position is valid"<<endl;
    if(config_.dyn_debug_verbose_on)
    {
      cout<<indStr(2)+"In world frame:\n"
          <<pose_world_dsl_goal_.affine().format(iof)<<endl;
      cout<<indStr(2)+"In ll frame axy:\n"
          <<axy_ll_dsl_goal.transpose().format(iof)<<endl;
    }
  }

  if(!dsl_cond_feas_g_ && config_.dyn_debug_on)
  {
    cout<<indStr(1)+"The goal position is invalid."<<endl;
    if(config_.dyn_debug_verbose_on)
    {
      cout<<indStr(2)+"In world frame:\n"
          <<pose_world_dsl_goal_.affine().format(iof)<<endl;
      cout<<indStr(2)+"In ll frame axy:\n"
          <<axy_ll_dsl_goal.transpose().format(iof)<<endl;
    }
  }
  ind_count_--;
}

void
CallBackDslDdp::cbOccGrid(const nav_msgs::OccupancyGridConstPtr& msg_occ_grid)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Occupancy grid is received"<<endl;

  //Copy OG to be used elsewhere
  og_original_ = *msg_occ_grid;

  //Find transformation between origin of og and world
  //  done in og subscriber because when a og arrives then you know that the launch file
  //  with transformations are also running
  setTfmsWorld2OgLL();

  // Process Occ Grid and publish
  occGridProcessAndPub();

  //Init dsl
  dslInit();

  ind_count_--;
}

void
CallBackDslDdp::setTfmsWorld2OgLL(void)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Find the transformation between the world and the ll(lower left) corner of OG"<<endl;

  //get the transformation between map to bottom left of the occupancy grid
  tf::StampedTransform tfms_world2og_org;
  tf_lr_.waitForTransform(strfrm_world_,strfrm_og_org_,ros::Time(0),ros::Duration(1.0));
  tf_lr_.lookupTransform(strfrm_world_,strfrm_og_org_, ros::Time(0), tfms_world2og_org);

  tf::transformTFToEigen(tfms_world2og_org,tfm_world2og_org_);
  poseMsg2Eig(tfm_og_org2og_ll_,og_original_.info.origin);
  tfm_world2og_ll_ = tfm_world2og_org_*tfm_og_org2og_ll_;

  double th = atan2(tfm_world2og_ll_.matrix()(1,0),tfm_world2og_ll_.matrix()(0,0));
  double x = tfm_world2og_ll_.matrix()(0,3);
  double y = tfm_world2og_ll_.matrix()(1,3);
  tfm_world2og_ll_2d_ = Translation2d(x,y)* Rotation2Dd(th);
  ind_count_--;
}
void
CallBackDslDdp::occGridProcessAndPub(void)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Processing the occ grid received and publishing the processed grid to rviz"<<endl;

  //Process occupancy grid starting with og_original_
  nav_msgs::OccupancyGrid og_dild_fild;
  occGridDilateAndFilterUnseen(og_original_, og_dild_fild);//dilate the resized occupancy grid
  occGridResize(og_dild_fild,og_final_);

  //publish the occupancy grid
  pub_og_final_.publish(og_final_);
  ind_count_--;

}
void
CallBackDslDdp::occGridDilateAndFilterUnseen(const nav_msgs::OccupancyGrid& og_original, nav_msgs::OccupancyGrid& og_dild_fild)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Dilating obstacles and filtering unseen"<<endl;

  double width = og_original.info.width;
  double height = og_original.info.height;

  cv::Mat img_og_original = cv::Mat(og_original.info.height,og_original.info.width,CV_8UC1,(uint8_t*)og_original.data.data());
  cv::Size size_img = img_og_original.size();

  //Find obstacle dilation parameters
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
  double og_cell_m_resized = yaml_node_["og_cell_m_resized"].as<double>();
  double og_cell_m_original = og_original_.info.resolution;
  int dilation_size_obs = config_.dyn_dilation_obs_m/og_original_.info.resolution;
  int dilation_size_scaling =  ceil((og_cell_m_original/og_cell_m_resized-1)/2) ;
  int dilation_size = dilation_size_scaling>dilation_size_obs ? dilation_size_scaling:dilation_size_obs;

  cv::Mat dilation_element = cv::getStructuringElement(dyn_dilation_type,
                                                       cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ) );

  //Dilate only the obstacles(not the unseen cells)
  cv::Mat img_wout_unseen;  cv::threshold(img_og_original,img_wout_unseen,101,100,cv::THRESH_TOZERO_INV);
  cv::Mat img_og_dild; cv::dilate(img_wout_unseen, img_og_dild, dilation_element );

  //Find parameter that is used to decide if a set of connected unseen cells are to be set free or not
  double cell_area = og_cell_m_original * og_cell_m_original;
  double og_tol_unseen_sqm = yaml_node_["og_tol_unseen_sqm"].as<double>();
  int og_tol_unseen_n = og_tol_unseen_sqm/cell_area;

  //Find connected unseen cells as blobs
  cv::Mat img_og_unseen;  cv::threshold(img_og_original,img_og_unseen,101,1.0,cv::THRESH_BINARY );
  std::vector < std::vector<cv::Point2i> > blobs;
  FindBlobs(img_og_unseen, blobs);

  //Create an image with all zeros(so free) then set the cells in large unseen clusters to 100(i.e. obstacles)
  cv::Mat img_og_unseen_fild(height,width,CV_8UC1,cv::Scalar(0));

  for(int n=0; n<blobs.size(); n++)
    if(blobs[n].size()>og_tol_unseen_n)
      for(int m=0; m<blobs[n].size(); m++)
          img_og_unseen_fild.at<uchar>(blobs[n][m].y,blobs[n][m].x)=100;

   //Combine the dilated obstacles with the larger unseen cell clusters set as obstacle
  cv::Mat img_og_dild_fild = cv::max(img_og_unseen_fild,img_og_dild);

  //display the final occ grid in rviz
  geometry_msgs::Pose pose_org; eig2PoseMsg(pose_org,tfm_world2og_ll_);
  occGridFromImg(img_og_dild_fild,pose_org, og_cell_m_original,og_dild_fild);

  ind_count_--;
}

void
CallBackDslDdp::occGridResize(const nav_msgs::OccupancyGrid& og_dild_fild, nav_msgs::OccupancyGrid& og_final)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Resizing the occ grid"<<endl;

  //make cv::Mat image from OG
  cv::Mat img_og_dild_fild = cv::Mat(og_dild_fild.info.height,og_dild_fild.info.width,CV_8UC1,(uint8_t*)og_dild_fild.data.data());
  cv::Size size_img = img_og_dild_fild.size();

  //get desired resolution(in meter/pix)
  double og_cell_m_resized = yaml_node_["og_cell_m_resized"].as<double>();
  double og_cell_m_original = og_dild_fild.info.resolution;

  //Row and cols of resized image
  double width_m = og_original_.info.width   * og_cell_m_original;
  double height_m = og_original_.info.height * og_cell_m_original;
  int cols_resized = floor(width_m/og_cell_m_resized);
  int rows_resized = floor(height_m/og_cell_m_resized);

  //create resized image
  img_og_final_ = cv::Mat(rows_resized,cols_resized,img_og_dild_fild.type());
  cv::Mat map_x(rows_resized,cols_resized, CV_32FC1 );
  cv::Mat map_y(rows_resized,cols_resized, CV_32FC1 );

  // remap the resized image
  for(int r=0;r<rows_resized; r++)
  {
    for(int c=0;c<cols_resized; c++)
    {
      map_x.at<float>(r,c) = (c +0.5)*og_cell_m_resized/og_cell_m_original;
      map_y.at<float>(r,c) = (r +0.5)*og_cell_m_resized/og_cell_m_original;
    }
  }
  cv::remap( img_og_dild_fild, img_og_final_, map_x, map_y, cv::INTER_NEAREST, cv::BORDER_CONSTANT);

  //create occ_grid_final
  geometry_msgs::Pose pose_org; eig2PoseMsg(pose_org,tfm_world2og_ll_);
  nav_msgs::OccupancyGrid occ_grid_dilated;
  occGridFromImg(img_og_final_,pose_org, og_cell_m_resized,og_final);

  ind_count_--;
}


void
CallBackDslDdp::occGridFromImg(const cv::Mat& img,const geometry_msgs::Pose pose_org, const double res_m_per_pix,  nav_msgs::OccupancyGrid& occ_grid)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Creating occ grid from image"<<endl;

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
  ind_count_--;
}

void
CallBackDslDdp::dslDelete()
{
  delete p_dsl_map_;

  delete p_dsl_grid2d_;
  delete p_dsl_conn2d_;
  delete p_dsl_cost2d_;
  delete p_dsl_search2d_;

  delete p_dsl_costcar_;
  delete p_dsl_gridcar_;
  delete p_dsl_conncar_;
  delete p_dsl_searchcar_;
}

void
CallBackDslDdp::dslInit()
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Initializing dsl now that a processed occ grid is ready"<<endl;

  //get required params from yaml_node_
  bool expand_at_start = yaml_node_["dsl_expand_at_start"].as<bool>();
  double cell_width = og_final_.info.resolution;
  double cell_height = og_final_.info.resolution;
  int map_width = og_final_.info.width;
  int map_height = og_final_.info.height;
  double max_cost = 0.5; //we just want something lower than 100(obstacle value)
  double dsl_cost_scale=1;
  bool dsl_use_geom_car_ = yaml_node_["dsl_use_geom_car"].as<bool>();
  int  dsl_prim_w_div = yaml_node_["dsl_prim_w_div"].as<int>();
  double dsl_maxtanphi = yaml_node_["dsl_maxtanphi"].as<double>();

  double l,b,ox,oy;
  if(dsl_use_geom_car_){
    Vector4d geom = yaml_node_["dsl_car_dim_and_org"].as<Vector4d>();
    l = geom(0); b = geom(1); ox = geom(2); oy = geom(3);
  }
  double bp = yaml_node_["dsl_backward_penalty"].as<double>();
  double onlyfwd = yaml_node_["dsl_onlyfwd"].as<bool>();



  ros::Time t_start = ros::Time::now();
  dslDelete();
  p_dsl_map_ = new double[og_final_.info.width*og_final_.info.height];
  for (int i = 0; i < og_final_.info.width*og_final_.info.height; ++i)
    p_dsl_map_[i] = 1000*(double)img_og_final_.data[i];

  dsl::save_map((const char*) img_og_final_.data, map_width, map_height, "/home/subhransu/projects/ws_catkin_lab/src/gcop_ros_packages/gcop_ctrl/map_final.ppm");


  if(dsl_use_geom_car_){
    p_dsl_gridcar_   =  new dsl::CarGrid(l,b,ox,oy,map_width, map_height,  p_dsl_map_, cell_width, cell_height, M_PI/16, dsl_cost_scale, 0.5);
    p_dsl_conncar_   =  new dsl::CarConnectivity(*p_dsl_gridcar_,bp, onlyfwd, dsl_prim_w_div,dsl_maxtanphi);
    p_dsl_costcar_   =  new dsl::CarCost();
    p_dsl_searchcar_ =  new dsl::GridSearch<3, Matrix3d>(*p_dsl_gridcar_, *p_dsl_conncar_, *p_dsl_costcar_, expand_at_start);
  }else{
    p_dsl_grid2d_   = new dsl::Grid2d(map_width, map_height, p_dsl_map_, cell_width, cell_height, dsl_cost_scale, 0.5);
    p_dsl_conn2d_   = new dsl::Grid2dConnectivity(*p_dsl_grid2d_);
    p_dsl_cost2d_   = new dsl::GridCost<2>();
    p_dsl_search2d_ = new dsl::GridSearch<2>(*p_dsl_grid2d_, *p_dsl_conn2d_, *p_dsl_cost2d_, expand_at_start);
  }
  ros::Time t_end =  ros::Time::now();

  if(config_.dyn_debug_verbose_on)
  {
    cout<<indStr(1)+"DSL grid initialized with map size:"<<og_final_.info.width<<" X "<<og_final_.info.height<<endl;
    cout<<indStr(1)+"delta t:"<<(t_end - t_start).toSec()<<" sec"<<endl;
  }
  ind_count_--;
}

bool
CallBackDslDdp::dslFeasible(void)
{
  float w = og_final_.info.width;
  float h = og_final_.info.height;
  return w && h && dsl_cond_feas_s_ && dsl_cond_feas_g_;
}

bool
CallBackDslDdp::dslPlan()
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Planning dsl path"<<endl;

  vector<Vector3d> pathaxy_ll_init;
  if(dslFeasible())
  {
    ros::Time t_start = ros::Time::now();
    if(dsl_use_geom_car_){
      dsl::SE2Path pathcar_ll_opt;
      if(!(p_dsl_searchcar_->Plan(pathcar_ll_opt))){
        if(config_.dyn_debug_on)
          cout<<indStr(1)+"Planning failed! No path from start to goal."<<endl;
        return false;
      }
      //convert the se2 path in to a simpler axy path
      pathaxy_ll_opt_.clear();
      for (vector<dsl::SE2Cell>::iterator it = pathcar_ll_opt.cells.begin(); it !=  pathcar_ll_opt.cells.end(); ++it)
          pathaxy_ll_opt_.push_back(it->c);
      pathaxy_ll_init = pathaxy_ll_opt_;
    }else{
      dsl::GridPath<2> path2d_ll_init,path2d_ll_opt;
      if(!(p_dsl_search2d_->Plan(path2d_ll_init))){
        if(config_.dyn_debug_on)
          cout<<indStr(1)+"Planning failed! No path from start to goal"<<endl;
        return false;
      }
      p_dsl_search2d_->OptPath(path2d_ll_init, path2d_ll_opt);
      pathaxy_ll_opt_.clear();
      for (vector<dsl::Cell<2>>::iterator it = path2d_ll_opt.cells.begin(); it !=  path2d_ll_opt.cells.end(); ++it)
          pathaxy_ll_opt_.push_back(Vector3d(0,it->c(0),it->c(1)));
      for (vector<dsl::Cell<2>>::iterator it = path2d_ll_init.cells.begin(); it !=  path2d_ll_init.cells.end(); ++it)
          pathaxy_ll_init.push_back(Vector3d(0,it->c(0),it->c(1)));
    }
    ros::Time t_end =  ros::Time::now();

    if(config_.dyn_debug_on)
    {
      cout<<indStr(1)+"Planned a path and optimized it and obtained a path(with "<<pathaxy_ll_opt_.size()<<"nodes)"<<endl;
      cout<<indStr(1)+"delta t:"<<(t_end - t_start).toSec()<<" sec"<<endl;
    }
    if(config_.dyn_debug_verbose_on)
    {
      VectorXd pt_x(pathaxy_ll_init.size());
      VectorXd pt_y(pathaxy_ll_init.size());
      for (int i = 0; i < pathaxy_ll_init.size(); i++)
      {
        pt_x(i) = pathaxy_ll_init[i](1);
        pt_y(i) = pathaxy_ll_init[i](2);
      }
      cout<<indStr(1)+"The number of initial dsl path points: "<<pt_x.size()<<endl;
      cout<<indStr(1)+"The dsl initial path points are: "<<endl;
      cout<<indStr(2)+"x:"<<pt_x.transpose()<<endl;
      cout<<indStr(2)+"y:"<<pt_y.transpose()<<endl;
    }

    dslInterpolate();
    dsl_done_ = true;
  }
  else
  {
    if(config_.dyn_debug_on)
    {
      cout<<indStr(1)+"Planning with DSL not possible because it's infeasible"<<endl;
    }
    dsl_done_=false;
  }

  ind_count_--;

  if(dslFeasible())
    return true;
  else
    return false;
}

void
CallBackDslDdp::dslInterpolate(void){
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Interpolating dsl path with a more regularly spaced(in time) path "<<endl;

  float m_per_cell = og_final_.info.resolution;
  int n = pathaxy_ll_opt_.size();

  VectorXd a_ll_opt(n),x_ll_opt(n),y_ll_opt(n); a_ll_opt.setZero(); x_ll_opt.setZero(); y_ll_opt.setZero();
  VectorXd dela(n), delx(n),dely(n); dela.setZero(); delx.setZero(); dely.setZero();
  VectorXd delt(n),t_opt(n);  delt.setZero(); t_opt.setZero();

  for (int i = 0; i < n; i++){
    a_ll_opt(i) = pathaxy_ll_opt_[i](0);
    x_ll_opt(i) = pathaxy_ll_opt_[i](1);
    y_ll_opt(i) = pathaxy_ll_opt_[i](2);
  }
  dela.tail(n-1) = a_ll_opt.tail(n-1) - a_ll_opt.head(n-1);
  delx.tail(n-1) = x_ll_opt.tail(n-1) - x_ll_opt.head(n-1);
  dely.tail(n-1) = y_ll_opt.tail(n-1) - y_ll_opt.head(n-1);

  //The time difference between 2 consecutive way points
  //is calculated as distance between the 2 way points divided by the avg speed of dsl path
  //TODO do this for the se2 path case as well
  delt = ((delx.array().square() + dely.array().square()).array().sqrt())/config_.dyn_dsl_avg_speed;
  for(int i=1;i<n;i++)
    t_opt(i) =t_opt(i-1) + delt(i);

  if(config_.dyn_debug_verbose_on){
    cout<<indStr(1)+"The number of optimal dsl path points: "<<x_ll_opt.size()<<endl;
    cout<<indStr(1)+"The length of optimal dsl path : "<<config_.dyn_dsl_avg_speed*t_opt.tail<1>()<<endl;
    cout<<indStr(1)+"The dsl optimal path points in ll frame are: "<<endl;
    cout<<indStr(2)+"a:"<<a_ll_opt.transpose()<<endl;
    cout<<indStr(2)+"x:"<<x_ll_opt.transpose()<<endl;
    cout<<indStr(2)+"y:"<<y_ll_opt.transpose()<<endl;
    cout<<indStr(2)+"t:"<<   t_opt.transpose()<<endl;
  }

  //Pre-Interpolate.
  //  Required because otherwise the smooth path might deviate far away from the line connecting
  //    2 adjacent vertices along the original path. So we will make the spline pass through these
  //    pre interpolated points

  //  Create linear interpolator for pre-interpolation from opt points
  //  TODO: a simple spline interpolation wouldn't work for se2
  //SplineFunction lint_a(t_opt,a_ll_opt,1);
  SplineFunction lint_x(t_opt,x_ll_opt,1);
  SplineFunction lint_y(t_opt,y_ll_opt,1);


  //  Change the input preint_delt to the value that is permissible
  bool update_dyn_server=false;
  if(config_.dyn_dsl_preint_delt > t_opt(n-1) ){
    config_.dyn_dsl_preint_delt = t_opt(n-1);
    update_dyn_server = true;
  }

  int n_segs_reg; n_segs_reg = round((double)(t_opt(n-1))/config_.dyn_dsl_preint_delt);
  int n_nodes_reg = n_segs_reg+1;//regularly spaced points
  double dt_reg= (double)(t_opt(n-1))/(double)n_segs_reg;


  //  Reserve space for regularly spaced nodes and the original nodes.
  //    n-2 because the 1st and last interpolated points are same as original
  vector<double> pt_x_dsl_preint_stl;vector<double> pt_y_dsl_preint_stl;vector<double> t_dsl_preint_stl;
  pt_x_dsl_preint_stl.reserve(n_nodes_reg + n-2); pt_x_dsl_preint_stl.push_back(x_ll_opt(0));
  pt_y_dsl_preint_stl.reserve(n_nodes_reg + n-2); pt_y_dsl_preint_stl.push_back(y_ll_opt(0));
  t_dsl_preint_stl.reserve(n_nodes_reg + n-2);    t_dsl_preint_stl.push_back(t_opt(0));

  //  Push the pre-interpolatd points and the original points into stl vectors
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
        pt_x_dsl_preint_stl.push_back(x_ll_opt(idx_opt));
        pt_y_dsl_preint_stl.push_back(y_ll_opt(idx_opt));
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

  //  convert above stl vectors to eigen vectors
  //    use Eigen::Map to do so
  VectorXd pt_x_dsl_preint = Map<VectorXd>(pt_x_dsl_preint_stl.data(),pt_x_dsl_preint_stl.size());
  VectorXd pt_y_dsl_preint = Map<VectorXd>(pt_y_dsl_preint_stl.data(),pt_y_dsl_preint_stl.size());
  VectorXd    t_dsl_preint = Map<VectorXd>(t_dsl_preint_stl.data(),t_dsl_preint_stl.size());

  if(config_.dyn_debug_verbose_on)
  {
    cout<<indStr(1)+"The number of pre-interpolation points: "<<pt_x_dsl_preint.size()<<endl;

    cout<<indStr(1)+"The pre-interpolated points in ll frame are: "<<endl;
    cout<<indStr(2)+"x:"<<pt_x_dsl_preint.transpose()<<endl;
    cout<<indStr(2)+"y:"<<pt_y_dsl_preint.transpose()<<endl;
    cout<<indStr(2)+"t:"<<   t_dsl_preint.transpose()<<endl;
  }

  //Main Interpolation

  //  Create spline interpolator from preint points
  SplineFunction intp_x(t_dsl_preint,pt_x_dsl_preint,config_.dyn_dsl_interp_deg);
  SplineFunction intp_y(t_dsl_preint,pt_y_dsl_preint,config_.dyn_dsl_interp_deg);

  //  Interpolate
  if(config_.dyn_dsl_interp_delt > t_opt(n-1) )
  {
    config_.dyn_dsl_interp_delt = t_opt(n-1);
    update_dyn_server = true;
  }
  int n_segs; n_segs = round((double)(t_opt(n-1))/config_.dyn_dsl_interp_delt);
  int n_nodes = n_segs+1;
  double dt= (double)(t_opt(n-1))/(double)n_segs;

  x_ll_intp_.resize(n_nodes); y_ll_intp_.resize(n_nodes);a_ll_intp_.resize(n_nodes); t_dsl_intp_.resize(n_nodes);
  x_ll_intp_(0) = x_ll_opt(0);     y_ll_intp_(0) = y_ll_opt(0);     t_dsl_intp_(0) = t_opt(0);

  for(int i=1;i<n_nodes;i++)
  {
    t_dsl_intp_(i) = i*dt;
    x_ll_intp_(i) = intp_x[i*dt];
    y_ll_intp_(i) = intp_y[i*dt];

    double dx = x_ll_intp_(i) -x_ll_intp_(i-1);
    double dy = y_ll_intp_(i) -y_ll_intp_(i-1);
    a_ll_intp_(i-1) = atan2(dy,dx);
  }
  a_ll_intp_(n_nodes-1) = a_ll_intp_(n_nodes-2);

  if(config_.dyn_debug_verbose_on)
  {
    cout<<"  The number of final interpolation points: "<<x_ll_intp_.size()<<endl;

    cout<<indStr(1)+"The final interpolated points are: "<<endl;
    cout<<indStr(2)+"x:"<<x_ll_intp_.transpose()<<endl;
    cout<<indStr(2)+"y:"<<y_ll_intp_.transpose()<<endl;
    cout<<indStr(2)+"a:"<<   a_ll_intp_.transpose()<<endl;
    cout<<indStr(2)+"t:"<<   t_dsl_intp_.transpose()<<endl;
  }

  //Update the values in the reconfigure node.
  if(update_dyn_server)
    dyn_server_.updateConfig(config_);

  ind_count_--;
}

bool
CallBackDslDdp::ddpFeasible(void)
{
  return dsl_done_;
}

bool
CallBackDslDdp::ddpInit(void)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Setting DDP params"<<endl;

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
  ddp_tol_goal_m_=yaml_node_["ddp_tol_goal_m"].as<double>();

  //Update internal gains of cost_lq
  cost_lq_.UpdateGains();
  ind_count_--;
}

bool
CallBackDslDdp::ddpPlan(void)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Entering local path planning"<<endl;

  if(!ddpFeasible())
    return false;
  float m_per_cell = og_final_.info.resolution;

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
    pose_ddp_start_ = pose_world_dsl_start_;
    vel_ddp_start_  = 0;
  }
  //Select ddp goal position by finding a point on the dsl way point
  //  that is t_away sec away from current position

  //  find nearest row(idx_nearest) on the [x_ll_intp_ ,y_ll_intp_] to posn_ddp_start_
  int n_nodes = x_ll_intp_.size();
  Vector3d pos_ll_ddp_start = (tfm_world2og_ll_.inverse()*pose_ddp_start_).translation();
  VectorXd dist_sq =(x_ll_intp_ - VectorXd::Ones(n_nodes)*pos_ll_ddp_start(0)).array().square()
                               +(y_ll_intp_ - VectorXd::Ones(n_nodes)*pos_ll_ddp_start(1)).array().square();
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
                  *Translation3d(x_ll_intp_(idx_t_away),y_ll_intp_(idx_t_away),0)
                  *AngleAxisd(a_ll_intp_(idx_t_away), Vector3d::UnitZ());

  //DDP path length in distance and time
  double tf = t_dsl_intp_(idx_t_away) - t_dsl_intp_(idx_nearest);
  double len_ddp_path = tf*config_.dyn_dsl_avg_speed;

  //Stop DDP planning if start position is close to goal position
  if(len_ddp_path<ddp_tol_goal_m_)
    return false;

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
    cout<<indStr(1)+"The ddp request is as follows"<<endl;
    cout<<indStr(2)+"Start x:"<<pose_ddp_start_.translation()(0)<<"\ty:"<<pose_ddp_start_.translation()(1)<<"\ta:"<<rpy_start(2)<<endl;
    cout<<indStr(2)+"Goal x:"<<pose_ddp_goal_.translation()(0)<<"\ty:"<<pose_ddp_goal_.translation()(1)<<"\ta:"<<rpy_goal(2)<<endl;
    cout<<indStr(2)+"tf:"<< tf<<" sec";
    cout<<indStr(2)+"path length:"<<len_ddp_path<<endl;
    cout<<indStr(2)+"nseg:"<<ddp_nseg<<endl;
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
      cout<<indStr(0)+"  Iteration number:"<<n_it<<endl;

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

  ind_count_--;
  return true;
}

void
CallBackDslDdp::rvizRemoveCar(void)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Removing car marker from rviz"<<endl;

  //remove start test related visualization marker
  marker_car_.action    = visualization_msgs::Marker::DELETE;
  pub_vis_.publish( marker_car_ );
  ind_count_--;
}

void
CallBackDslDdp::rvizRemoveStart(void)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Removing start marker from rviz"<<endl;

  //remove start test related visualization marker
  marker_text_start_.action    = visualization_msgs::Marker::DELETE;
  pub_vis_.publish( marker_text_start_ );
  ind_count_--;
}

void
CallBackDslDdp::rvizRemoveGoal(void)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Removing goal marker from rviz"<<endl;

  //remove goal test related visualization marker
  marker_text_goal_.action    = visualization_msgs::Marker::DELETE;
  pub_vis_.publish( marker_text_goal_ );
  ind_count_--;
}

void
CallBackDslDdp::rvizRemovePathDdp(void)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Removing ddp path from rviz"<<endl;

  //remove ddp related visualization marker
  marker_path_ddp_.action      = visualization_msgs::Marker::DELETE;
  marker_wp_ddp_.action        = visualization_msgs::Marker::DELETE;
  pub_vis_.publish( marker_path_ddp_ );
  pub_vis_.publish( marker_wp_ddp_ );
  ind_count_--;
}

void
CallBackDslDdp::rvizRemovePathDsl(void)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Removing dsl path from rviz"<<endl;

  //remove visualization marker
  marker_path_dsl_.action      = visualization_msgs::Marker::DELETE;
  marker_wp_dsl_.action        = visualization_msgs::Marker::DELETE;

  pub_vis_.publish( marker_path_dsl_ );
  pub_vis_.publish( marker_wp_dsl_ );
  ind_count_--;
}

void
CallBackDslDdp::rvizRemovePathDslInterpd(void)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Removing interpolated dsl path from rviz"<<endl;

  //remove visualization marker
  marker_path_dsl_intp_.action = visualization_msgs::Marker::DELETE;
  marker_wp_dsl_intp_.action   = visualization_msgs::Marker::DELETE;
  pub_vis_.publish( marker_path_dsl_intp_ );
  pub_vis_.publish( marker_wp_dsl_intp_ );
  ind_count_--;
}


void
CallBackDslDdp::rvizColorMsgEdit(std_msgs::ColorRGBA& rgba_msg, VectorXd& rgba_vec)
{
  rgba_msg.r = rgba_vec(0);
  rgba_msg.g = rgba_vec(1);
  rgba_msg.b = rgba_vec(2);
  rgba_msg.a = rgba_vec(3);
}

void
CallBackDslDdp::rvizMarkersEdit(visualization_msgs::Marker& marker, VectorXd& prop)
{
  ind_count_++;
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
      cout<<indStr(0)+"*Error setting marker properties"<<endl;
      break;
  }
  ind_count_--;
}


void
CallBackDslDdp::rvizMarkersInit(void)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Initializing all rviz markers."<<endl;

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
  rvizMarkersEdit(marker_path_dsl_,prop_path_n_wp);

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
  rvizMarkersEdit(marker_wp_dsl_,prop_path_n_wp);

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
  rvizMarkersEdit(marker_path_dsl_intp_,prop_path_n_wp);

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
  rvizMarkersEdit(marker_wp_dsl_intp_,prop_path_n_wp);

  //Marker for pve ddp path
  id++;
  marker_path_ddp_.header.frame_id = strfrm_world_;
  marker_path_ddp_.header.stamp = ros::Time();
  marker_path_ddp_.ns = "dsl_ddp_planner";
  marker_path_ddp_.id = id;
  marker_path_ddp_.type = visualization_msgs::Marker::LINE_STRIP;
  marker_path_ddp_.action = visualization_msgs::Marker::ADD;
  marker_path_ddp_.lifetime = ros::Duration(0);
  rvizMarkersEdit(marker_path_ddp_,prop_path_pve_ddp_);

  //Marker for ddp path way points
  id++;
  marker_wp_ddp_.header.frame_id = strfrm_world_;
  marker_wp_ddp_.header.stamp = ros::Time();
  marker_wp_ddp_.ns = "dsl_ddp_planner";
  marker_wp_ddp_.id = id;
  marker_wp_ddp_.type = visualization_msgs::Marker::POINTS;
  marker_wp_ddp_.action = visualization_msgs::Marker::ADD;
  marker_wp_ddp_.lifetime = ros::Duration(0);
  rvizMarkersEdit(marker_wp_ddp_,prop_wp_pve_ddp_);

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

  //Marker for car
  id++;
  marker_car_.header.frame_id = strfrm_world_;
  marker_car_.header.stamp = ros::Time();
  marker_car_.ns = "dsl_ddp_planner";
  marker_car_.id = id;
  marker_car_.type = visualization_msgs::Marker::CUBE_LIST;
  marker_car_.action = visualization_msgs::Marker::ADD;
  marker_car_.scale.x = 0.7;
  marker_car_.scale.y = 0.43;
  marker_car_.scale.z = 0.4;
  marker_car_.color.a = 0.2; // Don't forget to set the alpha!
  marker_car_.color.r = 1.0;
  marker_car_.color.g = 0.0;
  marker_car_.color.b = 0.0;
  marker_car_.lifetime = ros::Duration(0);

  ind_count_--;
}

void
CallBackDslDdp::rvizShowStart()
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Sending start marker to rviz"<<endl;

  marker_text_start_.pose.position.x =pose_world_dsl_start_.translation()(0);
  marker_text_start_.pose.position.y =pose_world_dsl_start_.translation()(1);
  marker_text_start_.pose.position.z =0.2;

  pub_vis_.publish( marker_text_start_ );
  ind_count_--;
}

void
CallBackDslDdp::rvizShowGoal()
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Sending goal marker to rviz"<<endl;

  marker_text_goal_.pose.position.x =pose_world_dsl_goal_.translation()(0);
  marker_text_goal_.pose.position.y =pose_world_dsl_goal_.translation()(1);
  marker_text_goal_.pose.position.z =0.2;

  pub_vis_.publish( marker_text_goal_ );
  ind_count_--;
}
void
CallBackDslDdp::rvizShowPathDsl()
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Sending DSL path to rviz"<<endl;

  marker_path_dsl_.action      = visualization_msgs::Marker::ADD;
  marker_wp_dsl_.action        = visualization_msgs::Marker::ADD;
  marker_path_dsl_.points.resize(pathaxy_ll_opt_.size());
  for (int i = 0; i < pathaxy_ll_opt_.size(); i++)
  {
    Vector3d posn_waypt_in_ll(pathaxy_ll_opt_[i](1),pathaxy_ll_opt_[i](2),0);
    Vector3d posn_waypt_in_world = tfm_world2og_ll_*posn_waypt_in_ll;
    geometry_msgs::Point node;
    node.x = posn_waypt_in_world(0);
    node.y = posn_waypt_in_world(1);
    node.z = 0.2;
    marker_path_dsl_.points[i] =node;
  }
  marker_wp_dsl_.points = marker_path_dsl_.points;
  pub_vis_.publish( marker_path_dsl_ );
  pub_vis_.publish( marker_wp_dsl_ );
  ind_count_--;
}

void
CallBackDslDdp::rvizShowPathDslInterpd(void)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Sending interpolated DSL path to rviz"<<endl;

  marker_path_dsl_intp_.action = visualization_msgs::Marker::ADD;
  marker_wp_dsl_intp_.action   = visualization_msgs::Marker::ADD;
  marker_car_.action   = visualization_msgs::Marker::ADD;

  marker_path_dsl_intp_.points.resize(x_ll_intp_.size());
  for (int i = 0; i < x_ll_intp_.size(); i++)
  {
    Vector3d posn_waypt_in_ll(x_ll_intp_(i),y_ll_intp_(i),0);
    Vector3d posn_waypt_in_world = tfm_world2og_ll_*posn_waypt_in_ll;
    geometry_msgs::Point node;
    node.x = posn_waypt_in_world(0);
    node.y = posn_waypt_in_world(1);
    node.z = 0.2;
    marker_path_dsl_intp_.points[i] =node;

    //marker_car_.points         = marker_path_dsl_intp_.points;

  }

  marker_wp_dsl_intp_.points = marker_path_dsl_intp_.points;

  pub_vis_.publish( marker_path_dsl_intp_ );
  pub_vis_.publish( marker_wp_dsl_intp_ );
  //pub_vis_.publish( marker_car_ );

  ind_count_--;
}

void
CallBackDslDdp::rvizShowPathDdp(void)
{
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Sending DDP  path to rviz"<<endl;

  float res = og_final_.info.resolution;
  marker_path_ddp_.action      = visualization_msgs::Marker::ADD;
  marker_wp_ddp_.action        = visualization_msgs::Marker::ADD;

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
      rvizColorMsgEdit(marker_path_ddp_.colors[i],prop_path_pve_ddp_);
      rvizColorMsgEdit(marker_wp_ddp_.colors[i],prop_wp_pve_ddp_);
    }
    else
    {
      rvizColorMsgEdit(marker_path_ddp_.colors[i],prop_path_nve_ddp_);
      rvizColorMsgEdit(marker_wp_ddp_.colors[i],prop_wp_nve_ddp_);
    }
  }
  pub_vis_.publish( marker_path_ddp_ );
  pub_vis_.publish( marker_wp_ddp_ );
  ind_count_--;
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
