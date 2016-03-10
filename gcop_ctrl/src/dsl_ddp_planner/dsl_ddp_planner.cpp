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
#include <gcop_ros_utils/pose_twist.h>
#include <gcop_ros_utils/pose_twist2d.h>

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
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <nav_msgs/Path.h>

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
#include <iterator>

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
#include <gcop/constraintcost.h>
#include <gcop/multicost.h>
#include <gcop/diskconstraint.h>

#include <gcop_ctrl/obs_detection_lidar/miniball.hpp>

//-------------------------------------------------------------------------
//-----------------------NAME SPACES ---------------------------------
//-------------------------------------------------------------------------
using namespace std;
using namespace Eigen;
using namespace gcop;


//-------------------------------------------------------------------------
//-----------------------GLOBAL VARIABLES ---------------------------------
//-------------------------------------------------------------------------
sig_atomic_t g_shutdown_requested=0;

//-------------------------------------------------------------------------
//---------------------------_TYPEDEFS ------------------------------------
//-------------------------------------------------------------------------
typedef std::vector<std::vector<double> >::const_iterator PointIterator;
typedef std::vector<double>::const_iterator CoordIterator;
typedef Miniball::Miniball <Miniball::CoordAccessor<PointIterator, CoordIterator> > MB;


//------------------------------------------------------------------------
//-----------------------FUNCTION DEFINITIONS ----------------------------
//------------------------------------------------------------------------


void mySigIntHandler(int signal){
  g_shutdown_requested=1;
}

void timer_start(struct timeval *time){
  gettimeofday(time,(struct timezone*)0);
}

long timer_us(struct timeval *time){
  struct timeval now;
  gettimeofday(&now,(struct timezone*)0);
  return 1000000*(now.tv_sec - time->tv_sec) + now.tv_usec - time->tv_usec;
}

template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}


//to be used in switch case statement involving strings
constexpr unsigned int str2int(const char* str, int h = 0){
  return !str[h] ? 5381 : (str2int(str, h+1)*33) ^ str[h];
}


void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs){
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

int GcarStateToVector2d(Vector2d &p, const GcarState &x){
  p[0] = x.g(0,2);   // x.g is the car SE(2) pose matrix
  p[1] = x.g(1,2);
  return 1;  // index where position coordinates start (the car model coordinates are (theta,x,y,v)
}


template<class T>
std::vector<T> selectVecElems(const std::vector<T>& vec, std::vector<std::size_t> indxs){
  std::vector<T> ret;
  for (std::size_t ind: indxs)
    ret.push_back(vec.at(ind));
  return ret;
}

/**
 * It finds a maximum of K clusters which are closest to origin. Each cluster has a maximum radius
 * @param clusters The returned clusters
 * @param ptpol pt in polar coordinates of the obstacle locations
 * @param robs radius of individual obstacle point
 * @param K K nearest neighbor
 * @param rcmax maximum radius of the cluster
 */
void findKClustersWithSizeConstraint(vector<vector<size_t>>& clusters, const vector<Vector2d>& pts_cart,double robs,int K, double rcmax){
  if(!pts_cart.size())
    return;

  //  Make a vector of ranges and convert ptpol to cartesian
  vector<int> clid(pts_cart.size(),-1);//-1 means cluster numbers have not been assigned
  vector<double> ranges(pts_cart.size());
  vector<size_t> idxs_noclid(pts_cart.size());//indexes of all the obstacles that don't have a cluster id
  for(size_t i=0;i<pts_cart.size();i++){
    ranges[i] = pts_cart[i].norm();
    idxs_noclid[i] = i;
  }

  int k=0; //current cluster id
  while(k<K && idxs_noclid.size()>0)
  {
    //vector of obstacle ranges with no clid(cluster id)
    vector<double> ranges_noclid = selectVecElems(ranges,idxs_noclid);

    vector<bool> nbrs_noclid(idxs_noclid.size());
    size_t n_nbrs=0;

    //  find the closest obstacle to center in the list of obstacles that don't belong to any clusters
    size_t idx_innoclid_closest = std::distance(ranges_noclid.begin(),min_element(ranges_noclid.begin(),ranges_noclid.end()));
    Vector2d pt_polar_closest = pts_cart[idxs_noclid[idx_innoclid_closest]];

    //find all obstacles with no clid which lie in a circle of radius rcmax and centered at
    //  pt_polar_closest + pt_polar(rcmax/2,0) and assign them the cluster id k
    Vector2d pt_polar_cent = pt_polar_closest + Vector2d(rcmax,0);
    Vector2d pt_cart_cent = Vector2d(pt_polar_cent(0)*cos(pt_polar_cent(1))
                                     ,pt_polar_cent(0)*sin(pt_polar_cent(1)));
    vector<size_t> cluster;
    for( size_t i_in_noclid=0; i_in_noclid < idxs_noclid.size(); i_in_noclid++){
      Vector2d pt_cart = pts_cart[idxs_noclid[i_in_noclid]];
      double dist = (pt_cart -pt_cart_cent).norm();

      if(dist<=rcmax+1e-10){
        clid[idxs_noclid[i_in_noclid]] = k;
        cluster.push_back(idxs_noclid[i_in_noclid]);
      }
    }
    clusters.push_back(cluster);

    //update idxs_noclid and k for the next loop
    idxs_noclid.clear();
    for(size_t i=0;i<pts_cart.size();i++){
      if(clid[i]==-1)
        idxs_noclid.push_back(i);
    }
    k++;
  }
}



void findCircumCircle(vector<Vector2d>& pts_cart_center_circle, vector<double>& radiuses_circle,
                      const vector<vector<size_t>>&clusters,const vector<Vector2d>& pts_cart){

  pts_cart_center_circle.resize(clusters.size());
  radiuses_circle.resize(clusters.size());
  for(int i=0; i< clusters.size(); i++){
    vector<Vector2d> pts_cart_cl = selectVecElems(pts_cart,clusters[i]);
    vector<vector<double>> lp;
    for(size_t i=0;i<pts_cart_cl.size();i++){
      vector<double> p; p.push_back(pts_cart_cl[i](0)); p.push_back(pts_cart_cl[i](1));
      lp.push_back(p);
    }
    int d=2; int n = pts_cart_cl.size();
    MB mb (d, lp.begin(), lp.end());

    const double* center = mb.center();
    pts_cart_center_circle[i] = Vector2d(center[0],center[1]);
    radiuses_circle[i] = sqrt(mb.squared_radius());
  }

}


struct ObsDetectionCfg{
  double search_radius_max;
  double search_radius_min;      //! The laser points
  double search_angle_fwd;       //! restrict the search angle of the lidar data
  int cluster_count_max;         //! max number of obstacles to be returned
  double cluster_radius_max;     //! the max cluster radius
  double map_cell_size;         //! cell size of the map
  shared_ptr<sensor_msgs::LaserScan> p_laserscan_msg;//! Pointer to the laserscan message
  shared_ptr<vector<Affine2d>> p_lidar2nodes;        //! Pointer to vector of transformation for lidar frame to nodei frame
};

//------------------------------------------------------------------------
//-----------------------------CALLBACK CLASS ----------------------------
//------------------------------------------------------------------------

class CallBackDslDdp{
public:
  typedef Matrix<float, 4, 4> Matrix4f;
  typedef Matrix<double, 4, 4> Matrix4d;
  typedef Matrix<double, 5, 1> Vector5d;
  typedef Ddp<GcarState, 4, 2> GcarDdp;
  typedef DiskConstraint<GcarState, 4, 2> GcarDiskConstraint;
  typedef boost::shared_ptr<GcarDiskConstraint> GcarDiskConstraint_ptr;
  typedef boost::shared_ptr<SplineFunction> SplineFunction_ptr;
  typedef ConstraintCost<GcarState, 4, 2, Dynamic, 1> DiskConstraintCost;
  typedef boost::shared_ptr<DiskConstraintCost> DiskConstraintCost_ptr;
  typedef Transform<double,2,Affine> Transform2d;
  typedef sensor_msgs::LaserScan::_ranges_type::const_iterator RangesConstIt;

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
  void cbLidar(const sensor_msgs::LaserScanConstPtr& pmsg_lidar);
  void cbClickedPoint(const geometry_msgs::PointStampedConstPtr& pmsg_cp );
  void cbTimerVis(const ros::TimerEvent& event);
  void cbTimerDsl(const ros::TimerEvent& event);
  void cbTimerDdp(const ros::TimerEvent& event);


  void setupTopicsAndNames(void);
  void initSubsPubsAndTimers(void);

  void rvizColorMsgEdit(std_msgs::ColorRGBA& rgba_msg, VectorXd& rgba_vec);
  void rvizMarkersEdit(visualization_msgs::Marker& marker, VectorXd& prop);
  void rvizMarkersInit(void);
  void rvizShowObstacles(void);
  void rvizShowPath(ros::Publisher& pub_posearr, const vector<GcarState>& path_gcar,
                    visualization_msgs::Marker& marker_path, visualization_msgs::Marker& marker_wp,
                    VectorXd& marker_prop_path, VectorXd& marker_prop_wp);
  void rvizRemovePath(visualization_msgs::Marker& marker_path, visualization_msgs::Marker& marker_wp);
  void rvizShowText(ros::Publisher& pub_ps, Affine3d& pose_w2b, visualization_msgs::Marker& marker_text, VectorXd& prop_text);
  void rvizRemoveText(visualization_msgs::Marker& marker_text);
  //void rvizRemoveObstacles(void);

  void dslInit(void);
  void dslDelete(void);
  bool dslPlan(void);
  bool dslFeasible(void);
  bool dslSetStartIfFeasible(void);
  bool dslSetGoalIfFeasible(void);
  bool dslGoalFeasible(void);
  void dslInterpolate(void);

  bool ddpFeasible(void);
  bool ddpInit(void);
  bool ddpPlan(void);
  void obsDetect(vector<Vector3d>& centers_encirc, vector<double>& rads_encirc, const ObsDetectionCfg& obs_cfg);
  /**
   * Finds n obstacles that are closest to the entire trajectory
   * @param centers_encirc center of the circle that encircles a set of pointsS
   * @param rads_encirc radius of the circle that encirles a set of points
   * @param cfg configuration for obstacle detection
   */
  void obsFindCloseToTraj(vector<Vector3d>& centers_encirc_inlidar, vector<double>& rads_encirc, const ObsDetectionCfg& cfg);

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
  string ind_str_;

  string strtop_odom_, strtop_pose_start_, strtop_pose_goal_, strtop_og_, strtop_lidar_;
  string strtop_diag_, strtop_marker_rviz_, strtop_og_dild_, strtop_ctrl_;
  string strtop_rviz_pose_start_, strtop_rviz_pose_goal_;
  string strtop_rviz_posearr_ddp_,strtop_rviz_posearr_dsl_, strtop_rviz_posearr_dsl_intp_;
  string strtop_clicked_point_;
  string strfrm_world_, strfrm_robot_, strfrm_og_org_;

  ros::Subscriber sub_odom_, sub_pose_start_, sub_pose_goal_, sub_og_, sub_lidar_;
  ros::Subscriber sub_clicked_point_;
  ros::Publisher pub_diag_, pub_vis_, pub_og_final_, pub_ctrl_;
  ros::Publisher pub_rviz_pose_start_, pub_rviz_pose_goal_;
  ros::Publisher pub_rviz_posearr_ddp_,pub_rviz_posearr_dsl_, pub_rviz_posearr_dsl_intp_;

  ros::Timer timer_vis_, timer_ddp_, timer_dsl_;

  tf::TransformListener tf_lr_;

  //Rviz display markers and their properties
  VectorXd marker_prop_path_dsl_;
  VectorXd marker_prop_wp_dsl_;
  VectorXd marker_prop_path_dsl_intp_;
  VectorXd marker_prop_wp_dsl_intp_;
  VectorXd marker_prop_path_ddp_;
  VectorXd marker_prop_wp_ddp_;
  VectorXd marker_prop_text_start_;
  VectorXd marker_prop_text_goal_;
  VectorXd marker_prop_obs_;

  visualization_msgs::Marker marker_path_dsl_,marker_wp_dsl_;
  visualization_msgs::Marker marker_path_dsl_intp_ ,marker_wp_dsl_intp_;
  visualization_msgs::Marker marker_path_ddp_,marker_wp_ddp_;
  visualization_msgs::Marker marker_text_start_, marker_text_goal_;
  visualization_msgs::Marker marker_obs_;


  nav_msgs::OccupancyGrid og_original_, og_final_;
  cv::Mat img_og_final_;
  int marker_id_;

  // Frames and transformation
  Affine3d tfm_world2og_org_, tfm_world2og_ll_,tfm_og_org2og_ll_;
  Transform2d tfm_world2og_ll_2d_;
  Affine3d pose_w2b_rviz_start_, pose_w2b_rviz_goal_;

  //Paths
  vector<GcarState>& path_gcar_w2b_ddp_;
  vector<GcarState>  path_gcar_w2b_dsl_;
  vector<GcarState>  path_gcar_w2b_dsl_intp_;

  // Velocity poses and time of the object
  Affine3d pose_aff3d_w2b_curr_;
  double vel_binb_curr_;
  ros::Time time_curr_;

  // DSL vars
  double* p_dsl_map_;
  //   dsl with geometric car
  dsl::CarCost*                 p_dsl_costcar_;
  dsl::CarGrid*                 p_dsl_gridcar_;
  dsl::CarConnectivity*         p_dsl_conncar_;
  dsl::GridSearch<3, Matrix3d>* p_dsl_searchcar_;
  vector<Vector3d>              path_axy_ll2b_opt_;
  bool dsl_cond_feas_s_, dsl_cond_feas_g_, dsl_done_;
  VectorXd x_ll2b_intp_, y_ll2b_intp_,a_ll2b_intp_, t_dsl_intp_;//x,y,angle,t of the path, ll2b: body in lower left of og
  Affine3d pose_w2b_dsl_start_, pose_w2b_dsl_goal_;//w2b: body in world
  SplineFunction_ptr p_spline_x_ll2b_, p_spline_y_ll2b_;
  dsl::CarPrimitiveCfg          prim_cfg_;

  // DDP vars
  bool ddp_debug_on_, ddp_force_cold_start_;
  double ddp_hot_start_delt_;
  double ddp_mu_;
  int ddp_nseg_max_,ddp_nseg_min_, ddp_nit_max_, ddp_init_type_;
  double ddp_tseg_ideal_;
  double ddp_tol_rel_, ddp_tol_abs_, ddp_tol_goal_m_;
  Affine3d pose_aff3d_w2b_ddp_start_, pose_aff3d_w2b_ddp_goal_;
  ros::Time time_ddp_start_;
  double vel_binb_ddp_start_,     vel_ddp_goal_;
  Vector3d pt_ddp_start_, pt_ddp_goal_; //refers to grid point
  vector<Vector3d> centers_encirc_oinw_prev_;
  vector<double> rads_encirc_prev_;
  Matrix4d ddp_Q_per_t_;
  Matrix2d ddp_R_per_t_;
  Vector2d ddp_disk_penl_minmax_;

  Gcar sys_gcar_;
  LqCost< GcarState, 4, 2> ddp_cost_lq_;
  double ddp_disk_coln_rad_;
  vector<double> ddp_ts_, ddp_ts_prev_;
  vector<GcarState> ddp_xs_;
  vector<Vector2d> ddp_us_, ddp_us_prev_;

  //obstacle properties
  vector<geometry_msgs::PointStamped> clicked_points_;
  ObsDetectionCfg obs_cfg_;
  double obs_clicked_rad_;
  sensor_msgs::LaserScan msg_lidar_;
  vector<Disk> disks_;

};

CallBackDslDdp::CallBackDslDdp():
  nh_p_("~"),
  loop_rate_main_(1000),
  ind_count_(-1),
  dsl_cond_feas_s_(false),
  dsl_cond_feas_g_(false),
  dsl_done_(false),
  p_dsl_map_(nullptr),
  p_dsl_costcar_(nullptr),
  p_dsl_gridcar_(nullptr),
  p_dsl_conncar_(nullptr),
  p_dsl_searchcar_(nullptr),
  path_gcar_w2b_ddp_(ddp_xs_),
  sys_gcar_(),
  ddp_cost_lq_(sys_gcar_, 1, GcarState(Matrix3d::Identity(), 0)){
ind_count_++;
  cout<<"**************************************************************************"<<endl;
  cout<<"***************************DSL-DDP TRAJECTORY PLANNER*********************"<<endl;
  cout<<"*Entering constructor of cbc"<<endl;

  //Setup YAML reading and parsing
  string strfile_params;nh_p_.getParam("strfile_params",strfile_params);
  yaml_node_ = YAML::LoadFile(strfile_params);
  ind_str_ = yaml_node_["str_ind"].as<string>();
  cout<<indStr(1)+"loaded yaml param file into yaml_node"<<endl;

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
  sys_gcar_.U.lb[1] = tan(yaml_node_["gcar_minphi"].as<double>());
  sys_gcar_.U.ub[1] = tan(yaml_node_["gcar_maxphi"].as<double>());

  //init ddp planner
  ddpInit();

  cout<<indStr(1)+"Waiting for start and goal position."<<endl;
  cout<<indStr(1)+"Select Start through Publish Point button and select goal through 2D nav goal."<<endl;

  ind_count_--;
}


CallBackDslDdp::~CallBackDslDdp(){
  rvizRemovePath(marker_path_ddp_, marker_wp_ddp_);
  rvizRemovePath(marker_path_dsl_, marker_wp_dsl_);
  rvizRemovePath(marker_path_dsl_intp_, marker_wp_dsl_intp_);
  rvizRemoveText(marker_text_start_);
  rvizRemoveText(marker_text_goal_);
  //rvizRemoveObstacles();

  dslDelete();
  cv::destroyAllWindows();
}
string CallBackDslDdp::indStr(){
  string ind;
  for(int i=0; i<ind_count_; i++)
    ind = ind+ ind_str_;
  return ind;
}

string CallBackDslDdp::indStr(int count_extra){
  string ind;
  for(int i=0; i<ind_count_+count_extra; i++)
    ind = ind+ ind_str_;
  return ind;
}

void CallBackDslDdp::cbReconfig(gcop_ctrl::DslDdpPlannerConfig &config, uint32_t level){
  ind_count_++;
  static bool first_time=true;

  if(!first_time){
    if(config_.dyn_debug_on)
      cout<<indStr(0)+"*Dynamic reconfigure called"<<endl;

    //Change in dilation
    bool condn_dilate =     og_final_.info.width
        && (   config.dyn_dilation_obs_m != config_.dyn_dilation_obs_m
            || config.dyn_dilation_type != config_.dyn_dilation_type );
    if(condn_dilate){
      config_.dyn_dilation_obs_m = config.dyn_dilation_obs_m;
      config_.dyn_dilation_type = config.dyn_dilation_type;
      occGridProcessAndPub();
    }

    //loop rate setting
    if(config_.dyn_loop_rate_main != config.dyn_loop_rate_main)
      loop_rate_main_= ros::Rate(config.dyn_loop_rate_main);

    //dsl settings
    if(config.dyn_dsl_plan_once){
      if(dslPlan() && config.dyn_dsl_disp_rviz){
        rvizShowPath(pub_rviz_posearr_dsl_, path_gcar_w2b_dsl_,
                     marker_path_dsl_, marker_wp_dsl_,
                     marker_prop_path_dsl_, marker_prop_wp_dsl_);
        rvizShowPath(pub_rviz_posearr_dsl_intp_, path_gcar_w2b_dsl_intp_,
                     marker_path_dsl_intp_, marker_wp_dsl_intp_,
                     marker_prop_path_dsl_intp_, marker_prop_wp_dsl_intp_);
      }else{
        rvizRemovePath(marker_path_dsl_, marker_wp_dsl_);
        rvizRemovePath(marker_path_dsl_intp_, marker_wp_dsl_intp_);
      }
      config.dyn_dsl_plan_once = false;
    }

    if(config.dyn_reset_clicked_points){
      clicked_points_.clear();
      config.dyn_reset_clicked_points = false;
    }

    if(config_.dyn_dsl_loop_durn != config.dyn_dsl_loop_durn)
      timer_dsl_.setPeriod(ros::Duration(config.dyn_dsl_loop_durn));

    if(config_.dyn_dsl_plan_loop != config.dyn_dsl_plan_loop && config.dyn_dsl_plan_loop )
      timer_dsl_.start();

    if(config_.dyn_dsl_plan_loop != config.dyn_dsl_plan_loop && !config.dyn_dsl_plan_loop )
      timer_dsl_.stop();

    //ddp settings
    ddp_Q_per_t_.setZero();
    ddp_Q_per_t_.diagonal() << config.dyn_ddp_Q_per_t_00
        ,config.dyn_ddp_Q_per_t_11
        ,config.dyn_ddp_Q_per_t_22
        ,config.dyn_ddp_Q_per_t_33;

    ddp_R_per_t_.setZero();
    ddp_R_per_t_.diagonal() << config.dyn_ddp_R_per_t_00
        ,config.dyn_ddp_R_per_t_11;

    ddp_disk_penl_minmax_(1) = config.dyn_ddp_disk_penl_max;

    if(config.dyn_ddp_plan_once){
      if(ddpPlan() && config_.dyn_ddp_disp_rviz){
        //rvizShowPathDdp();
        rvizShowPath(pub_rviz_posearr_ddp_, path_gcar_w2b_ddp_, marker_path_ddp_, marker_wp_ddp_, marker_prop_path_ddp_, marker_prop_wp_ddp_);
        rvizShowObstacles();
      }else{
        rvizRemovePath(marker_path_ddp_, marker_wp_ddp_);
        //rvizRemovePathDdp();
        //rvizRemoveObstacles();
      }

      config.dyn_ddp_plan_once = false;
    }

    if(config_.dyn_ddp_hot_start != config.dyn_ddp_hot_start && config.dyn_ddp_hot_start )
      ddp_force_cold_start_ = true;

    if(config_.dyn_ddp_loop_durn != config.dyn_ddp_loop_durn)
      timer_ddp_.setPeriod(ros::Duration(config.dyn_ddp_loop_durn));

    if(config_.dyn_ddp_plan_loop != config.dyn_ddp_plan_loop && config.dyn_ddp_plan_loop )
      timer_ddp_.start();

    if(config_.dyn_ddp_plan_loop != config.dyn_ddp_plan_loop && !config.dyn_ddp_plan_loop )
      timer_ddp_.stop();

  }else{
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
    config.dyn_ddp_hot_start      = yaml_node_["ddp_hot_start"].as<bool>();

    Matrix2d R_per_t              = yaml_node_["ddp_R_per_t"].as<Matrix2d>();
    config.dyn_ddp_R_per_t_00     = R_per_t(0,0);
    config.dyn_ddp_R_per_t_11     = R_per_t(1,1);

    Matrix4d Q_per_t              = yaml_node_["ddp_Q_per_t"].as<Matrix4d>();
    config.dyn_ddp_Q_per_t_00          = Q_per_t(0,0);
    config.dyn_ddp_Q_per_t_11          = Q_per_t(1,1);
    config.dyn_ddp_Q_per_t_22          = Q_per_t(2,2);
    config.dyn_ddp_Q_per_t_33          = Q_per_t(3,3);

    config.dyn_ddp_disk_penl_max  = (yaml_node_["ddp_disk_penl_minmax"].as<Vector2d>())(1);

    first_time = false;
  }
  config_ = config;
  ind_count_--;
}

void CallBackDslDdp::setupTopicsAndNames(void){
  ind_count_++;

  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*setting up topic names"<<endl;

  // Input topics
  strtop_odom_       = yaml_node_["strtop_odom"].as<string>();
  strtop_pose_start_ = yaml_node_["strtop_pose_start"].as<string>();
  strtop_pose_goal_  = yaml_node_["strtop_pose_goal"].as<string>();
  strtop_og_          = yaml_node_["strtop_og"].as<string>();
  strtop_lidar_ = yaml_node_["strtop_lidar"].as<string>();
  strtop_clicked_point_ = yaml_node_["strtop_clicked_point"].as<string>();

  // output topics
  strtop_diag_        = yaml_node_["strtop_diag"].as<string>();
  strtop_marker_rviz_ = yaml_node_["strtop_marker_rviz"].as<string>();
  strtop_og_dild_     = yaml_node_["strtop_og_dild"].as<string>();
  strtop_ctrl_       = yaml_node_["strtop_ctrl"].as<string>();

  strtop_rviz_pose_start_      = yaml_node_["strtop_rviz_pose_start"].as<string>();
  strtop_rviz_pose_goal_       = yaml_node_["strtop_rviz_pose_goal"].as<string>();
  strtop_rviz_posearr_ddp_     = yaml_node_["strtop_rviz_posearr_ddp"].as<string>();
  strtop_rviz_posearr_dsl_     = yaml_node_["strtop_rviz_posearr_dsl"].as<string>();
  strtop_rviz_posearr_dsl_intp_= yaml_node_["strtop_rviz_posearr_dsl_intp"].as<string>();

  // Frames
  strfrm_world_        = yaml_node_["strfrm_world"].as<string>();
  strfrm_robot_      = yaml_node_["strfrm_robot"].as<string>();
  strfrm_og_org_         = yaml_node_["strfrm_og_org"].as<string>();

  ind_count_--;
}

void CallBackDslDdp::initSubsPubsAndTimers(void){
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Initializing subs pubs and timers"<<endl;

  //Setup subscribers
  sub_odom_       = nh_.subscribe(strtop_odom_,1, &CallBackDslDdp::cbOdom,this, ros::TransportHints().tcpNoDelay());
  sub_og_   = nh_.subscribe(strtop_og_,  1, &CallBackDslDdp::cbOccGrid, this);
  sub_pose_start_ = nh_.subscribe(strtop_pose_start_, 1, &CallBackDslDdp::cbPoseStart, this);
  sub_pose_goal_  = nh_.subscribe(strtop_pose_goal_, 1, &CallBackDslDdp::cbPoseGoal, this);
  sub_lidar_  =     nh_.subscribe(strtop_lidar_, 1, &CallBackDslDdp::cbLidar, this);
  sub_clicked_point_  = nh_.subscribe(strtop_clicked_point_, 1, &CallBackDslDdp::cbClickedPoint, this);

  //Setup Publishers
  pub_diag_     = nh_.advertise<visualization_msgs::Marker>( strtop_diag_, 0 );
  pub_vis_      = nh_.advertise<visualization_msgs::Marker>( strtop_marker_rviz_, 0 );
  pub_og_final_ = nh_.advertise<nav_msgs::OccupancyGrid>(strtop_og_dild_,0,true);
  pub_ctrl_     = nh_.advertise<gcop_comm::GcarCtrl>(strtop_ctrl_,0);

  pub_rviz_pose_start_      = nh_.advertise<geometry_msgs::PoseStamped>(strtop_rviz_pose_start_,0,true);
  pub_rviz_pose_goal_       = nh_.advertise<geometry_msgs::PoseStamped>(strtop_rviz_pose_goal_,0,true);
  pub_rviz_posearr_ddp_     = nh_.advertise<geometry_msgs::PoseArray>(strtop_rviz_posearr_ddp_,0,true);
  pub_rviz_posearr_dsl_     = nh_.advertise<geometry_msgs::PoseArray>(strtop_rviz_posearr_dsl_,0,true);
  pub_rviz_posearr_dsl_intp_= nh_.advertise<geometry_msgs::PoseArray>(strtop_rviz_posearr_dsl_intp_,0,true);

  //Setup timers
  timer_vis_ = nh_.createTimer(ros::Duration(0.1), &CallBackDslDdp::cbTimerVis, this);
  timer_vis_.stop();

  timer_dsl_ = nh_.createTimer(ros::Duration(config_.dyn_dsl_loop_durn), &CallBackDslDdp::cbTimerDsl, this);
  timer_dsl_.stop();

  timer_ddp_ = nh_.createTimer(ros::Duration(config_.dyn_ddp_loop_durn), &CallBackDslDdp::cbTimerDdp, this);
  timer_ddp_.stop();
  ind_count_--;
}

void CallBackDslDdp::cbTimerVis(const ros::TimerEvent& event){
  pub_vis_.publish( marker_path_dsl_ );
}

void CallBackDslDdp::cbTimerDsl(const ros::TimerEvent& event){
  if(dslPlan() && config_.dyn_dsl_disp_rviz){
    rvizShowPath(pub_rviz_posearr_dsl_, path_gcar_w2b_dsl_,
                 marker_path_dsl_, marker_wp_dsl_,
                 marker_prop_path_dsl_, marker_prop_wp_dsl_);
    rvizShowPath(pub_rviz_posearr_dsl_intp_, path_gcar_w2b_dsl_intp_,
                 marker_path_dsl_intp_, marker_wp_dsl_intp_,
                 marker_prop_path_dsl_intp_, marker_prop_wp_dsl_intp_);
  }else{
    rvizRemovePath(marker_path_dsl_, marker_wp_dsl_);
    rvizRemovePath(marker_path_dsl_intp_, marker_wp_dsl_intp_);
  }
}


void CallBackDslDdp::cbTimerDdp(const ros::TimerEvent& event){
  if(event.last_real.isZero())
    ddp_force_cold_start_ = true;
  else
    ddp_force_cold_start_ = false;

  ddp_hot_start_delt_ = (event.current_real-event.last_real).toSec();
  //cout<<"ddp _hot_start:"<<ddp_hot_start_delt_<<endl;

  if(ddpPlan() && config_.dyn_ddp_disp_rviz){
    rvizShowPath(pub_rviz_posearr_ddp_, path_gcar_w2b_ddp_, marker_path_ddp_, marker_wp_ddp_, marker_prop_path_ddp_, marker_prop_wp_ddp_);
    rvizShowObstacles();
  }else{
    rvizRemovePath(marker_path_ddp_, marker_wp_ddp_);
    // rvizRemoveObstacles();
  }
}

void CallBackDslDdp::cbOdom(const nav_msgs::OdometryConstPtr& msg_odom){
  time_curr_ = msg_odom->header.stamp;
  poseMsg2Eig(pose_aff3d_w2b_curr_,msg_odom->pose.pose);
  vel_binb_curr_ = msg_odom->twist.twist.linear.x;
}

void CallBackDslDdp::cbPoseStart(const geometry_msgs::PoseWithCovarianceStampedConstPtr& msg_pose_start){
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Initial pose received from rviz"<<endl;
  poseMsg2Eig(pose_w2b_rviz_start_,msg_pose_start->pose.pose);
  rvizShowText(pub_rviz_pose_start_,pose_w2b_rviz_start_, marker_text_start_, marker_prop_text_start_);
  Affine3d pose_ll2b_rviz_start = tfm_world2og_ll_.inverse()*pose_w2b_rviz_start_;
  Vector3d rpy_rviz_start; SO3::Instance().g2q(rpy_rviz_start,pose_ll2b_rviz_start.rotation().matrix());
  Vector3d axy_ll_rviz_start(rpy_rviz_start(2),pose_ll2b_rviz_start.translation()(0),pose_ll2b_rviz_start.translation()(1));
  IOFormat iof(StreamPrecision,0,", ",",\n",indStr(3)+"","","","");
  if(config_.dyn_debug_verbose_on){
    cout<<indStr(2)+"In world frame:\n"
        <<pose_w2b_rviz_start_.affine().format(iof)<<endl;
    cout<<indStr(2)+"In ll frame axy:\n"
        <<axy_ll_rviz_start.transpose().format(iof)<<endl;
  }

  ind_count_--;
}

void CallBackDslDdp::cbPoseGoal(const geometry_msgs::PoseStampedConstPtr& msg_pose_goal){
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Initial pose received from rviz"<<endl;
  poseMsg2Eig(pose_w2b_rviz_goal_,msg_pose_goal->pose);
  rvizShowText(pub_rviz_pose_goal_,pose_w2b_rviz_goal_, marker_text_goal_, marker_prop_text_goal_);
  Affine3d pose_ll2b_rviz_goal = tfm_world2og_ll_.inverse()*pose_w2b_rviz_goal_;
  Vector3d rpy_rviz_goal; SO3::Instance().g2q(rpy_rviz_goal,pose_ll2b_rviz_goal.rotation().matrix());
  Vector3d axy_ll_rviz_goal(rpy_rviz_goal(2),pose_ll2b_rviz_goal.translation()(0),pose_ll2b_rviz_goal.translation()(1));
  IOFormat iof(StreamPrecision,0,", ",",\n",indStr(3)+"","","","");
  if(config_.dyn_debug_verbose_on){
    cout<<indStr(2)+"In world frame:\n"
        <<pose_w2b_rviz_goal_.affine().format(iof)<<endl;
    cout<<indStr(2)+"In ll frame axy:\n"
        <<axy_ll_rviz_goal.transpose().format(iof)<<endl;
  }

  ind_count_--;
}


void CallBackDslDdp::cbOccGrid(const nav_msgs::OccupancyGridConstPtr& msg_occ_grid){
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


void CallBackDslDdp::cbLidar(const sensor_msgs::LaserScanConstPtr& pmsg_lidar){
  msg_lidar_ = *pmsg_lidar;
}

void CallBackDslDdp::cbClickedPoint(const geometry_msgs::PointStampedConstPtr& pmsg_cp ){
  clicked_points_.push_back(*pmsg_cp);
}

void CallBackDslDdp::setTfmsWorld2OgLL(void){
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

void CallBackDslDdp::occGridProcessAndPub(void){
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

void CallBackDslDdp::occGridDilateAndFilterUnseen(const nav_msgs::OccupancyGrid& og_original, nav_msgs::OccupancyGrid& og_dild_fild){
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

void CallBackDslDdp::occGridResize(const nav_msgs::OccupancyGrid& og_dild_fild, nav_msgs::OccupancyGrid& og_final){
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


void CallBackDslDdp::occGridFromImg(const cv::Mat& img,const geometry_msgs::Pose pose_org, const double res_m_per_pix,  nav_msgs::OccupancyGrid& occ_grid){
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

void CallBackDslDdp::dslDelete(){
  delete p_dsl_map_;

  delete p_dsl_costcar_;
  delete p_dsl_gridcar_;
  delete p_dsl_conncar_;
  delete p_dsl_searchcar_;
}

void CallBackDslDdp::dslInit(){
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Initializing dsl now that a processed occ grid is ready"<<endl;

  //get required params from yaml_node_

  bool expand_at_start = yaml_node_["dsl_expand_at_start"].as<bool>();
  double cell_width = og_final_.info.resolution;
  double cell_height = og_final_.info.resolution;
  double cell_delth  = M_PI/16;
  int map_width = og_final_.info.width;
  int map_height = og_final_.info.height;

  int  dsl_prim_w_div = yaml_node_["dsl_prim_w_div"].as<int>();
  double dsl_maxtanphi = yaml_node_["dsl_maxtanphi"].as<double>();
  int  dsl_prim_vx_div = yaml_node_["dsl_prim_vx_div"].as<int>();
  double dsl_maxvx = yaml_node_["dsl_maxvx"].as<double>();
  double bp = yaml_node_["dsl_backward_penalty"].as<double>();
  double onlyfwd = yaml_node_["dsl_onlyfwd"].as<bool>();

  prim_cfg_.fwdonly = yaml_node_["dsl_onlyfwd"].as<bool>();
  prim_cfg_.tphioverlmax = yaml_node_["dsl_tanphioverlmax"].as<double>();
  prim_cfg_.lmin = yaml_node_["dsl_prim_lmin"].as<double>();
  prim_cfg_.lmax = yaml_node_["dsl_prim_lmax"].as<double>();
  prim_cfg_.nl = (uint) yaml_node_["dsl_prim_nl"].as<int>();
  prim_cfg_.amax = yaml_node_["dsl_prim_amax"].as<double>();
  prim_cfg_.na = (uint)yaml_node_["dsl_prim_na"].as<int>();

  bool dsl_save_final_map = yaml_node_["dsl_save_final_map"].as<bool>();
  dsl::CarGeom car_geom;
  Vector4d geom = yaml_node_["dsl_car_dim_and_org"].as<Vector4d>();
  car_geom.l = geom(0); car_geom.b = geom(1); car_geom.ox = geom(2); car_geom.oy = geom(3);
  double dsl_cost_scale=1;

  if(expand_at_start && config_.dyn_debug_verbose_on){
    cout<<indStr(1)+"og_final resolution:"<<og_final_.info.resolution<<endl;
    cout<<indStr(1)+"the size of final og:"<<map_width<<", "<<map_height<<endl;
    cout<<indStr(1)+"Expanding search graph at start will take some time"<<endl;
  }

  ros::Time t_start = ros::Time::now();
  dslDelete();
  p_dsl_map_ = new double[og_final_.info.width*og_final_.info.height];
  for (int i = 0; i < og_final_.info.width*og_final_.info.height; ++i)
    p_dsl_map_[i] = 1000*(double)img_og_final_.data[i];
  dsl::Map2d dsl_map2d(map_width, map_height, p_dsl_map_);

  if(dsl_save_final_map){
    if(config_.dyn_debug_on)
      cout<<indStr(1)+"Saving the final processed occ grid in gcop_ctrl/map/map_final.ppm"<<endl;
    string file = ros::package::getPath("gcop_ctrl") + string("/map/map_final.ppm");
    dsl::save_map((const char*) img_og_final_.data, map_width, map_height, file.c_str());
  }

  p_dsl_gridcar_   =  new dsl::CarGrid(car_geom,dsl_map2d, cell_width, cell_height, cell_delth, dsl_cost_scale, 0.5);
  if(yaml_node_["dsl_use_right_connectivity"].as<bool>()){
    p_dsl_conncar_   =  new dsl::CarConnectivity(*p_dsl_gridcar_,prim_cfg_);
    for(size_t i=0; i< p_dsl_conncar_->vss.size(); i++)
      cout<<"n prims["<<i<<"]:"<<p_dsl_conncar_->vss[i].size()<<endl;
  }else
    p_dsl_conncar_   =  new dsl::CarConnectivity(*p_dsl_gridcar_,bp, onlyfwd, dsl_prim_w_div,dsl_maxtanphi, dsl_prim_vx_div,dsl_maxvx);
  p_dsl_costcar_   =  new dsl::CarCost();
  p_dsl_searchcar_ =  new dsl::GridSearch<3, Matrix3d>(*p_dsl_gridcar_, *p_dsl_conncar_, *p_dsl_costcar_, expand_at_start);

  ros::Time t_end =  ros::Time::now();

  if(config_.dyn_debug_on){
    cout<<indStr(1)+"DSL grid initialized with map size:"<<og_final_.info.width<<" X "<<og_final_.info.height<<endl;
    cout<<indStr(1)+"delta t:"<<(t_end - t_start).toSec()<<" sec"<<endl;
  }

  //visualize the primitives
    int img_hsize = ceil(prim_cfg_.lmax/cell_width);
    int img_size = 2*img_hsize + 1;

    cv::Mat img_prim;
    Affine3d igorg_to_mgcorg = Scaling(Vector3d(1/cell_width, 1/cell_width, 1)) *Translation3d(Vector3d(img_hsize*cell_width, img_hsize*cell_width, 0));



//    for(int idx_a=0; idx_a < p_dsl_conncar_->vss.size() && !g_shutdown_requested ;idx_a++){
    if(yaml_node_["dsl_vis_prim"].as<bool>()){
      cv::namedWindow("img_prim", cv::WINDOW_NORMAL);
      for(int idx_a=15; idx_a < 16 && !g_shutdown_requested ;idx_a++){
        img_prim = cv::Mat(img_size,img_size,CV_8UC1, cv::Scalar(0));
        double th = p_dsl_gridcar_->xlb(0) + p_dsl_gridcar_->cs(0)*(idx_a+0.5);
        //cout<<"theta:"<<th<<" at idx_a:"<<idx_a<<endl;
        Affine3d igorg_to_car = igorg_to_mgcorg * AngleAxisd(th,Vector3d::UnitZ());
        for(int idx_p=0; idx_p< p_dsl_conncar_->vss[idx_a].size();idx_p++){
          //cout<<"prim["<<idx_p<<"]:"<<p_dsl_conncar_->vss[idx_a][idx_p].transpose()<<endl;
          Matrix3d dg; SE2::Instance().exp(dg, p_dsl_conncar_->vss[idx_a][idx_p]);
          Vector3d xyzend(dg(0,2),dg(1,2),0);
          Vector3i idxi = (igorg_to_car * xyzend).cast<int>();

          if( idxi(1) < img_size && idxi(2) < img_size )
            img_prim.at<uchar>(idxi(1),idxi(0)) = 255;

        }
        cv::imshow("img_prim", img_prim);
        cv::waitKey(0);
      }
    cv::destroyWindow("img_prim");
    }


  ind_count_--;
}

bool CallBackDslDdp::dslFeasible(void){
  float w = og_final_.info.width;
  float h = og_final_.info.height;
  return w && h && dsl_cond_feas_s_ && dsl_cond_feas_g_;
}

bool CallBackDslDdp::dslSetStartIfFeasible(void){
  float wpix = og_final_.info.resolution;//width of pixel in meter
  float hpix = og_final_.info.resolution;//height of pixel in meter
  int w = og_final_.info.width;
  int h = og_final_.info.height;

  Affine3d pose_ll2b_dsl_start = tfm_world2og_ll_.inverse()*pose_w2b_dsl_start_;
  Vector3d rpy_start; SO3::Instance().g2q(rpy_start,pose_ll2b_dsl_start.rotation().matrix());
  Vector3d axy_ll_dsl_start = Vector3d(rpy_start(2),pose_ll2b_dsl_start.translation()(0),pose_ll2b_dsl_start.translation()(1));

  dsl_cond_feas_s_ = (axy_ll_dsl_start(1)>=0) && (axy_ll_dsl_start(2)>=0) && (axy_ll_dsl_start(1)<w*wpix) && (axy_ll_dsl_start(2)<h*hpix);
  if(dsl_cond_feas_s_){
    axy_ll_dsl_start(0) = axy_ll_dsl_start(0) > p_dsl_gridcar_->xub(0)? axy_ll_dsl_start(0)-2*M_PI:axy_ll_dsl_start(0);
    axy_ll_dsl_start(0) = axy_ll_dsl_start(0) < p_dsl_gridcar_->xlb(0)? axy_ll_dsl_start(0)+2*M_PI:axy_ll_dsl_start(0);
    dsl_cond_feas_s_ = p_dsl_searchcar_->SetStart(axy_ll_dsl_start);

    if(!dsl_cond_feas_s_)
      rvizRemoveText(marker_text_start_);
  }
  return dsl_cond_feas_s_;
}

bool CallBackDslDdp::dslSetGoalIfFeasible(void){
  float wpix = og_final_.info.resolution;//width of pixel in meter
  float hpix = og_final_.info.resolution;//height of pixel in meter
  int w = og_final_.info.width;
  int h = og_final_.info.height;

  Affine3d pose_ll2b_dsl_goal = tfm_world2og_ll_.inverse()*pose_w2b_dsl_goal_;
  Vector3d rpy_goal; SO3::Instance().g2q(rpy_goal,pose_ll2b_dsl_goal.rotation().matrix());
  Vector3d axy_ll_dsl_goal = Vector3d(rpy_goal(2),pose_ll2b_dsl_goal.translation()(0),pose_ll2b_dsl_goal.translation()(1));

  dsl_cond_feas_g_ = (axy_ll_dsl_goal(1)>=0) && (axy_ll_dsl_goal(2)>=0) && (axy_ll_dsl_goal(1)<w*wpix) && (axy_ll_dsl_goal(2)<h*hpix);
  if(dsl_cond_feas_g_){
    axy_ll_dsl_goal(0) = axy_ll_dsl_goal(0) > p_dsl_gridcar_->xub(0)? axy_ll_dsl_goal(0)-2*M_PI:axy_ll_dsl_goal(0);
    axy_ll_dsl_goal(0) = axy_ll_dsl_goal(0) < p_dsl_gridcar_->xlb(0)? axy_ll_dsl_goal(0)+2*M_PI:axy_ll_dsl_goal(0);
    dsl_cond_feas_g_ = p_dsl_searchcar_->SetGoal(axy_ll_dsl_goal);
    if(!dsl_cond_feas_g_)
      rvizRemoveText(marker_text_goal_);
  }
  return dsl_cond_feas_g_;
}

bool CallBackDslDdp::dslPlan(){
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Planning dsl path"<<endl;

  //Set dsl start and goal
  if(config_.dyn_dsl_from_curr_posn)
    pose_w2b_dsl_start_ = pose_aff3d_w2b_curr_;
  else
    pose_w2b_dsl_start_ = pose_w2b_rviz_start_;
  pose_w2b_dsl_goal_ = pose_w2b_rviz_goal_;

  dslSetStartIfFeasible();
  dslSetGoalIfFeasible();

  vector<Vector3d> path_axy_ll2b_initial;
  if(dslFeasible()){

    //Plan
    ros::Time t_start = ros::Time::now();
    dsl::SE2Path path_dsl_ll2b_opt;
    if(!(p_dsl_searchcar_->Plan(path_dsl_ll2b_opt))){
      if(config_.dyn_debug_on)
        cout<<indStr(1)+"Planning failed! No path from start to goal."<<endl;
      dsl_done_=false;
      dslInit(); //because it doesn't work after it has failed once. Bug probably.
      return false;
    }
    ros::Time t_end =  ros::Time::now();

    //convert the se2 path in to the gcar state path
    path_gcar_w2b_dsl_.clear(); path_gcar_w2b_dsl_.resize(path_dsl_ll2b_opt.cells.size());
    for (size_t i=0 ; i<path_dsl_ll2b_opt.cells.size() ; i++){
      path_gcar_w2b_dsl_[i].g = (toAffine2d(tfm_world2og_ll_) * axyToAffine2d(path_dsl_ll2b_opt.cells[i].c)).matrix();
      path_gcar_w2b_dsl_[i].v = config_.dyn_dsl_avg_speed;
    }

    //convert the se2 path in to a simpler axy path
    path_axy_ll2b_opt_.clear();
    for (vector<dsl::SE2Cell>::iterator it = path_dsl_ll2b_opt.cells.begin(); it !=  path_dsl_ll2b_opt.cells.end(); ++it)
      path_axy_ll2b_opt_.push_back(it->c);
    path_axy_ll2b_initial = path_axy_ll2b_opt_;


    // debug info
    if(config_.dyn_debug_on){
      cout<<indStr(1)+"Planned a path and optimized it and obtained a path(with "<<path_axy_ll2b_opt_.size()<<"nodes)"<<endl;
      cout<<indStr(1)+"delta t:"<<(t_end - t_start).toSec()<<" sec"<<endl;
    }
    if(config_.dyn_debug_verbose_on){
      VectorXd pt_x(path_axy_ll2b_initial.size());
      VectorXd pt_y(path_axy_ll2b_initial.size());
      for (int i = 0; i < path_axy_ll2b_initial.size(); i++){
        pt_x(i) = path_axy_ll2b_initial[i](1);
        pt_y(i) = path_axy_ll2b_initial[i](2);
      }
      cout<<indStr(1)+"The number of initial dsl path points: "<<pt_x.size()<<endl;
      cout<<indStr(1)+"The dsl initial path points are: "<<endl;
      cout<<indStr(2)+"x:"<<pt_x.transpose()<<endl;
      cout<<indStr(2)+"y:"<<pt_y.transpose()<<endl;
    }

    dslInterpolate();
    dsl_done_ = true;
  }else{
    if(config_.dyn_debug_on)
      cout<<indStr(1)+"Planning with DSL not possible because it's infeasible"<<endl;
    dsl_done_=false;
  }

  ind_count_--;

  if(dslFeasible())
    return true;
  else
    return false;
}

void CallBackDslDdp::dslInterpolate(void){
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Interpolating dsl path with a more regularly spaced(in time) path "<<endl;

  float m_per_cell = og_final_.info.resolution;
  int n = path_axy_ll2b_opt_.size();

  VectorXd a_ll_opt(n),x_ll_opt(n),y_ll_opt(n); a_ll_opt.setZero(); x_ll_opt.setZero(); y_ll_opt.setZero();
  VectorXd dela(n), delx(n),dely(n); dela.setZero(); delx.setZero(); dely.setZero();
  VectorXd delt(n),t_opt(n);  delt.setZero(); t_opt.setZero();

  for (int i = 0; i < n; i++){
    a_ll_opt(i) = path_axy_ll2b_opt_[i](0);
    x_ll_opt(i) = path_axy_ll2b_opt_[i](1);
    y_ll_opt(i) = path_axy_ll2b_opt_[i](2);
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
  while( abs(t_opt(n-1) - t_dsl_preint_stl.back()) > tol){
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

  if(config_.dyn_debug_verbose_on){
    cout<<indStr(1)+"The number of pre-interpolation points: "<<pt_x_dsl_preint.size()<<endl;

    cout<<indStr(1)+"The pre-interpolated points in ll frame are: "<<endl;
    cout<<indStr(2)+"x:"<<pt_x_dsl_preint.transpose()<<endl;
    cout<<indStr(2)+"y:"<<pt_y_dsl_preint.transpose()<<endl;
    cout<<indStr(2)+"t:"<<   t_dsl_preint.transpose()<<endl;
  }

  //Main Interpolation

  //  Create spline interpolator from preint points
  p_spline_x_ll2b_.reset(new SplineFunction(t_dsl_preint,pt_x_dsl_preint,config_.dyn_dsl_interp_deg));
  p_spline_y_ll2b_.reset(new SplineFunction(t_dsl_preint,pt_y_dsl_preint,config_.dyn_dsl_interp_deg));
  SplineFunction& intp_x(*p_spline_x_ll2b_);
  SplineFunction& intp_y(*p_spline_y_ll2b_);

  //  Interpolate
  //  Change the input interp_delt to the value that is permissible
  if(config_.dyn_dsl_interp_delt > t_opt(n-1) ){
    config_.dyn_dsl_interp_delt = t_opt(n-1);
    update_dyn_server = true;
  }

  int n_segs; n_segs = round((double)(t_opt(n-1))/config_.dyn_dsl_interp_delt);
  int n_nodes = n_segs+1;
  double dt= (double)(t_opt(n-1))/(double)n_segs;

  x_ll2b_intp_.resize(n_nodes); y_ll2b_intp_.resize(n_nodes);a_ll2b_intp_.resize(n_nodes); t_dsl_intp_.resize(n_nodes);
  x_ll2b_intp_(0) = x_ll_opt(0);     y_ll2b_intp_(0) = y_ll_opt(0);     t_dsl_intp_(0) = t_opt(0);

  for(int i=1;i<n_nodes;i++){
    t_dsl_intp_(i) = i*dt;
    x_ll2b_intp_(i) = intp_x[i*dt];
    y_ll2b_intp_(i) = intp_y[i*dt];

    double dx = x_ll2b_intp_(i) -x_ll2b_intp_(i-1);
    double dy = y_ll2b_intp_(i) -y_ll2b_intp_(i-1);
    a_ll2b_intp_(i-1) = atan2(dy,dx);
  }
  a_ll2b_intp_(n_nodes-1) = a_ll2b_intp_(n_nodes-2);

  //convert the interpolated axy path in to the gcar state path
  path_gcar_w2b_dsl_intp_.resize(n_nodes);
  for(int i=0;i<n_nodes;i++){
    path_gcar_w2b_dsl_intp_[i].g = toAffine2d(tfm_world2og_ll_*axyToAffine3d(a_ll2b_intp_[i],x_ll2b_intp_[i],y_ll2b_intp_[i])).matrix();
    path_gcar_w2b_dsl_intp_[i].v = config_.dyn_dsl_avg_speed;
  }

  if(config_.dyn_debug_verbose_on){
    cout<<"  The number of final interpolation points: "<<x_ll2b_intp_.size()<<endl;

    cout<<indStr(1)+"The final interpolated points are: "<<endl;
    cout<<indStr(2)+"x:"<<x_ll2b_intp_.transpose()<<endl;
    cout<<indStr(2)+"y:"<<y_ll2b_intp_.transpose()<<endl;
    cout<<indStr(2)+"a:"<<   a_ll2b_intp_.transpose()<<endl;
    cout<<indStr(2)+"t:"<<   t_dsl_intp_.transpose()<<endl;
  }

  //Update the values in the reconfigure node.
  if(update_dyn_server)
    dyn_server_.updateConfig(config_);

  ind_count_--;
}

bool CallBackDslDdp::ddpFeasible(void){
  return dsl_done_;
}

bool CallBackDslDdp::ddpInit(void){
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Setting DDP params"<<endl;

  //Fetch and set all the parameter from yaml file
  ddp_Q_per_t_  = yaml_node_["ddp_Q_per_t"].as<Matrix4d>();
  ddp_R_per_t_  = yaml_node_["ddp_R_per_t"].as<Matrix2d>();

  ddp_disk_penl_minmax_ = yaml_node_["ddp_disk_penl_minmax"].as<Vector2d>();
  ddp_disk_coln_rad_ = yaml_node_["ddp_disk_coln_rad"].as<double>();
  ddp_mu_ =yaml_node_["ddp_mu"].as<double>();
  ddp_debug_on_ =yaml_node_["ddp_debug_on"].as<bool>(); // turn off debug for speed
  ddp_nseg_min_= (yaml_node_["ddp_nseg_minmax"].as<Vector2d>())(0);
  ddp_nseg_max_= (yaml_node_["ddp_nseg_minmax"].as<Vector2d>())(1);
  ddp_tseg_ideal_ =yaml_node_["ddp_tseg_ideal"].as<double>();
  ddp_nit_max_=yaml_node_["ddp_nit_max"].as<int>();
  ddp_tol_abs_=yaml_node_["ddp_tol_abs"].as<double>();
  ddp_tol_rel_=yaml_node_["ddp_tol_rel"].as<double>();
  ddp_tol_goal_m_=yaml_node_["ddp_tol_goal_m"].as<double>();
  ddp_init_type_ = yaml_node_["ddp_init_type"].as<int>();
  ddp_force_cold_start_ = true;

  //get obstacle detection parameters
  obs_cfg_.search_radius_max       = yaml_node_["obs_search_radius_max"].as<double>();
  obs_cfg_.search_radius_min       = yaml_node_["obs_search_radius_min"].as<double>();
  obs_cfg_.search_angle_fwd    = yaml_node_["obs_search_angle_fwd"].as<double>();
  obs_cfg_.cluster_count_max   = yaml_node_["obs_cluster_count_max"].as<int>();
  obs_cfg_.cluster_radius_max  = yaml_node_["obs_cluster_radius_max"].as<double>();
  obs_cfg_.map_cell_size       = yaml_node_["obs_map_cell_size"].as<double>();
  obs_cfg_.p_laserscan_msg.reset(&msg_lidar_);
  obs_clicked_rad_         = yaml_node_["obs_clicked_rad"].as<double>();

  ind_count_--;
}


bool CallBackDslDdp::ddpPlan(void){
  ind_count_++;
  ros::Time t_start =  ros::Time::now();

  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Entering DDP planning"<<endl;

  if(!ddpFeasible()){
    if(config_.dyn_debug_on)
      cout<<indStr(1)+"DDP not feasible as dsl path not ready"<<endl;
    return false;
  }
  float m_per_cell = og_final_.info.resolution;

  //DDP start
  //if dyn_ddp_from_curr_posn=true then start is current position else
  if(config_.dyn_ddp_from_curr_posn){
    time_ddp_start_ = time_curr_;
    pose_aff3d_w2b_ddp_start_ = pose_aff3d_w2b_curr_;
    vel_binb_ddp_start_  =  vel_binb_curr_;
  }else{
    time_ddp_start_ = ros::Time::now();
    pose_aff3d_w2b_ddp_start_ = pose_w2b_rviz_start_;
    vel_binb_ddp_start_  = 0;
  }
  if(config_.dyn_debug_on){
    IOFormat iof(StreamPrecision,0,", ",",\n",indStr(2)+"","","","");
    cout<<indStr(1)+"DDP start pose(affine matrix) is"<<endl;
    cout<< pose_aff3d_w2b_ddp_start_.affine().format(iof)<<endl;
    cout<<indStr(1)+"DDP start forward velocity is:"<<vel_binb_ddp_start_<<endl;
  }

  //Don't do DDP planning if robot is near path goal
  //TODO: put better stopping criteria
  double dist_start_goal = (pose_aff3d_w2b_ddp_start_.translation() - pose_w2b_dsl_goal_.translation()).head<2>().norm();
  if(dist_start_goal < ddp_tol_goal_m_){
    if(config_.dyn_debug_on)
      cout<<indStr(1)+"DDP planning not done because robot is already close to goal. Dist to goal:"
      <<dist_start_goal<<endl;
    if(config_.dyn_send_gcar_ctrl && yaml_node_["ddp_brake_at_goal"].as<bool>()){
      gcop_comm::GcarCtrl msg_ctrl;
      msg_ctrl.ts_eph.push_back(time_ddp_start_);
      msg_ctrl.us_vel.push_back(-1.0);
      msg_ctrl.us_phi.push_back(0);
      pub_ctrl_.publish(msg_ctrl);
    }
    ind_count_--;
    return false;
  }

  //Select ddp goal position by finding a point on the dsl way point
  //  If the t_away point lies on the obstacle then move it forward
  //  that is t_away sec away from current position

  //  find nearest row(idx_nearest) on the [x_ll_intp_ ,y_ll_intp_] to posn_ddp_start_
  int n_nodes_dsl = x_ll2b_intp_.size();
  Vector3d pos_ll_ddp_start = (tfm_world2og_ll_.inverse()*pose_aff3d_w2b_ddp_start_).translation();
  VectorXd dist_sq =(x_ll2b_intp_ - VectorXd::Ones(n_nodes_dsl)*pos_ll_ddp_start(0)).array().square()
                                                           +(y_ll2b_intp_ - VectorXd::Ones(n_nodes_dsl)*pos_ll_ddp_start(1)).array().square();
  VectorXd::Index idx_min; dist_sq.minCoeff(&idx_min);
  int idx_nearest = (int)idx_min;

  //  set pose_ddp_goal_ to xy_dsl_intp_(idx_t_away) where idx_t_away is the index of dsl_intp path
  //    which is t_away sec ahead of dsl_intp path at idx_nearest
  vector<double> t_stl(t_dsl_intp_.size()); Map<VectorXd>(t_stl.data(),t_dsl_intp_.size()) = t_dsl_intp_;
  double t_away = config_.dyn_ddp_t_away;
  vector<double>::iterator it_t_away = upper_bound(t_stl.begin(),t_stl.end(),t_stl[idx_nearest]+t_away);
  int idx_t_away = it_t_away-t_stl.begin()==t_stl.size()? t_stl.size()-1: it_t_away-t_stl.begin();

  //  DDP goal
  pose_aff3d_w2b_ddp_goal_ = tfm_world2og_ll_ * axyToAffine3d(a_ll2b_intp_(idx_t_away),
                                                               x_ll2b_intp_(idx_t_away),
                                                               y_ll2b_intp_(idx_t_away));
  //length in distance and time of DSL path segment(which ddp will track)
  double tf = t_dsl_intp_(idx_t_away) - t_dsl_intp_(idx_nearest);
  double len_ddp_path = tf*config_.dyn_dsl_avg_speed;

  //Determine the number of segments for ddp based on tf, nseg_max, tseg_ideal
  //  and resize ts xs and us based on that
  int ddp_nseg = ceil(tf/ddp_tseg_ideal_);
  ddp_nseg = ddp_nseg>ddp_nseg_max_?ddp_nseg_max_:ddp_nseg;
  ddp_nseg = ddp_nseg<ddp_nseg_min_?ddp_nseg_min_:ddp_nseg;
  int n_nodes_ddp = ddp_nseg + 1;
  double h=tf/ddp_nseg;
  ddp_ts_.resize(n_nodes_ddp);
  ddp_xs_.resize(n_nodes_ddp);
  ddp_us_.resize(ddp_nseg);

  // Start and goal ddp state(GcarState. M3 is se2 elem and V1 is forward vel)
  GcarState x0(toAffine2d(pose_aff3d_w2b_ddp_start_).matrix(), vel_binb_ddp_start_);
  GcarState xf(toAffine2d(pose_aff3d_w2b_ddp_goal_).matrix(), config_.dyn_dsl_avg_speed);

  //The Desired trajectory is the dsl trajectory segment from dsl_nearest to dsl_t_away
  double t_dsl_nearest = t_dsl_intp_(idx_nearest);
  vector<GcarState> gcar_reftraj(n_nodes_ddp);
//  SplineFunction& intp_x(*p_spline_x_ll2b_);
//  SplineFunction& intp_y(*p_spline_y_ll2b_);
  SplineFunction intp_x(t_dsl_intp_, x_ll2b_intp_,3); //TODO: get the degree from yaml file
  SplineFunction intp_y(t_dsl_intp_, y_ll2b_intp_,3); //TODO: get the degree from yaml file
  vector<double> xd_ll2b(n_nodes_ddp), yd_ll2b(n_nodes_ddp), ad_ll2b(n_nodes_ddp);
  xd_ll2b[0] = intp_x[t_dsl_nearest]; yd_ll2b[0] = intp_y[t_dsl_nearest];
  for(int i=1;i<n_nodes_ddp;i++){
    double t = t_dsl_nearest + i*h;
    xd_ll2b[i] = intp_x[t];
    yd_ll2b[i] = intp_y[t];

    double dx = xd_ll2b[i] -xd_ll2b[i-1];
    double dy = yd_ll2b[i] -yd_ll2b[i-1];
    ad_ll2b[i-1] = atan2(dy,dx);
  }
  ad_ll2b[n_nodes_ddp-1] = ad_ll2b[n_nodes_ddp-2];//repeat last node

  for(int i=0;i<n_nodes_ddp;i++){
    gcar_reftraj[i] = GcarState(toAffine2d(tfm_world2og_ll_*axyToAffine3d(ad_ll2b[i],xd_ll2b[i],yd_ll2b[i])).matrix(),
                                config_.dyn_dsl_avg_speed);
  }

  Vector3d rpy_start; SO3::Instance().g2q(rpy_start,pose_aff3d_w2b_ddp_start_.linear());
  Vector3d rpy_goal; SO3::Instance().g2q(rpy_goal,pose_aff3d_w2b_ddp_goal_.linear());
  if(config_.dyn_debug_on){
    cout<<indStr(1)+"The ddp request is as follows"<<endl;
    cout<<indStr(2)+"Start x:"<<pose_aff3d_w2b_ddp_start_.translation()(0)<<"\ty:"<<pose_aff3d_w2b_ddp_start_.translation()(1)<<"\ta:"<<rpy_start(2)<<endl;
    cout<<indStr(2)+"Goal x:"<<pose_aff3d_w2b_ddp_goal_.translation()(0)<<"\ty:"<<pose_aff3d_w2b_ddp_goal_.translation()(1)<<"\ta:"<<rpy_goal(2)<<endl;
    cout<<indStr(2)+"tf:"<< tf<<" sec";
    cout<<indStr(2)+"path length:"<<len_ddp_path<<endl;
    cout<<indStr(2)+"nseg:"<<ddp_nseg<<endl;
  }

  //detect obstacles in the frame of the robots current position(in body frame) and convert it to world frame
  disks_.clear();
  vector<Vector3d> centers_encirc_oinb;
  vector<double> rads_encirc;
  //obsDetect(centers_encirc_oinb, rads_encirc, obs_cfg_);
  size_t n_obs_detected = rads_encirc.size();
  vector<Vector3d>centers_encirc_oinw(centers_encirc_oinb.size());
  transform(centers_encirc_oinb.begin(),centers_encirc_oinb.end(), centers_encirc_oinw.begin(),
            [&](Vector3d& c_oinb){return pose_aff3d_w2b_curr_* c_oinb;});


  // Detect the nearest obstacle for regularly spaced position

  //Add clicked points to list of obstacles
  centers_encirc_oinw.resize(n_obs_detected + clicked_points_.size());
  rads_encirc.resize(n_obs_detected +clicked_points_.size());
  for(size_t i=n_obs_detected; i < n_obs_detected + clicked_points_.size();i++){
    centers_encirc_oinw[i] << clicked_points_[i].point.x, clicked_points_[i].point.y, 0;
    rads_encirc[i] = obs_clicked_rad_;
  }

  //Update the disks_ vector with all the obstacles
  for(int i=0; i< centers_encirc_oinw.size();i++)
    disks_.push_back(Disk(centers_encirc_oinw[i].head<2>(), rads_encirc[i]));
  int n_obs = disks_.size();

  if(config_.dyn_debug_on)
    cout<<indStr(1)+"The number of obstacles detected:"<<n_obs<<endl;


  //Set tf and xf for ddp_cost_lq_
  ddp_cost_lq_.tf = tf;
  ddp_cost_lq_.xf = &xf;
  ddp_cost_lq_.xds = &gcar_reftraj;

  //Initialize trajectory(ts, xs and us)
  for (int k = 0; k <ddp_nseg+1; ++k)
    ddp_ts_[k] = k*h;

  ddp_xs_[0]=x0;

  //Initialize ddp initial controls
  //cold start by default
  for (int k = 0; k <ddp_nseg; ++k)
    ddp_us_[k].setZero();// or put a function that sets the initial value of u

  //hot start
  if(config_.dyn_ddp_plan_loop && config_.dyn_ddp_hot_start && !ddp_force_cold_start_){
    if(config_.dyn_debug_on)
      cout<<indStr(2)+"Hot start used"<<endl;
    VectorXd u0_prev(ddp_us_prev_.size());
    VectorXd u1_prev(ddp_us_prev_.size());
    VectorXd t_prev(ddp_us_prev_.size());
    for(int i=0; i<ddp_us_prev_.size();i++){
      u0_prev(i) = ddp_us_prev_[i](0);
      u1_prev(i) = ddp_us_prev_[i](1);
      t_prev(i)  = ddp_ts_prev_[i];
    }

    SplineFunction lint_u0(t_prev,u0_prev,1);
    SplineFunction lint_u1(t_prev,u1_prev,1);

    //copy some of the prev controls to the next based on time(no extrapolation)
    //this is the hotstart part
    ddp_hot_start_delt_ = 0; //testing remove for final thing
    for (int k = 0; k <ddp_nseg && ddp_hot_start_delt_+ t_prev(k) < ddp_ts_prev_.back(); ++k){
      ddp_us_[k](0) = lint_u0[ddp_hot_start_delt_+ t_prev(k)];
      ddp_us_[k](1) = lint_u1[ddp_hot_start_delt_+ t_prev(k)];
    }
  }

  //Update ddp params
  ddp_cost_lq_.Qf.setZero(); //since we are tracking a trajectory
  ddp_cost_lq_.Q = ddp_Q_per_t_*tf;
  ddp_cost_lq_.R = ddp_R_per_t_*tf;

  ddp_cost_lq_.UpdateGains();

  //Setup multicost for DDP
  vector<GcarDiskConstraint_ptr> constraints(n_obs);
  vector<DiskConstraintCost_ptr> ddp_cost_disks(n_obs);
  MultiCost<GcarState, 4, 2> ddp_mcost(sys_gcar_,tf); ddp_mcost.costs.resize(n_obs+1);
  for (int i = 0; i < disks_.size(); ++i) {
    constraints[i].reset(new GcarDiskConstraint(disks_[i], 1.0));
    constraints[i]->func = GcarStateToVector2d;
    DiskConstraintCost_ptr ptr_diskcost(new DiskConstraintCost(sys_gcar_, tf, *constraints[i]));
    ddp_cost_disks[i] = ptr_diskcost;
    ddp_cost_disks[i]->b = ddp_disk_penl_minmax_(0);
    ddp_mcost.costs[i] = ddp_cost_disks[i].get();
  }
  ddp_mcost.costs.back() = &ddp_cost_lq_;
  double disk_penalty_step = exp(log(ddp_disk_penl_minmax_(1)/ddp_disk_penl_minmax_(0))/(ddp_nit_max_-1));

  //setup DDP solver
  GcarDdp ddp_solver(sys_gcar_, ddp_mcost, ddp_ts_, ddp_xs_, ddp_us_);
  ddp_solver.mu =ddp_mu_;
  ddp_solver.debug = ddp_debug_on_;
  ddp_solver.Update();

  //Run an iteration loop until convergence
  int n_it(0);
  bool ddp_conv=false;
  double v_prev; v_prev=ddp_solver.V;

  while(!ddp_conv && !g_shutdown_requested){
    for(size_t i=0;i<disks_.size();i++)
      ddp_cost_disks[i]->b = ddp_disk_penl_minmax_(0)* pow(disk_penalty_step, n_it);
    ddp_solver.Iterate();


    if(n_it  > ddp_nit_max_)
      ddp_conv=true;

    //    if(                                     n_it  > ddp_nit_max_
    //        ||             abs(ddp_solver.V - v_prev) < ddp_tol_abs_
    //        || abs(ddp_solver.V - v_prev)/abs(v_prev) < ddp_tol_rel_)
    //      ddp_conv=true;
    //    v_prev=ddp_solver.V;
    n_it++;
  }

  //save prev controls for hot start if enabled
  ddp_us_prev_ = ddp_us_;
  ddp_ts_prev_ = ddp_ts_;

  //create the GcarCtrl message and publish it
  if(config_.dyn_send_gcar_ctrl){
    gcop_comm::GcarCtrl msg_ctrl;
    msg_ctrl.ts_eph.resize(ddp_us_.size());
    msg_ctrl.us_vel.resize(ddp_us_.size());
    msg_ctrl.us_phi.resize(ddp_us_.size());
    for (int i = 0; i < ddp_us_.size(); ++i){
      msg_ctrl.ts_eph[i] = time_ddp_start_+ ros::Duration(ddp_ts_[i]);
      msg_ctrl.us_vel[i] = ddp_xs_[i].v;
      msg_ctrl.us_phi[i] = atan(ddp_us_[i](1));
    }
    pub_ctrl_.publish(msg_ctrl);
  }

  ros::Time t_end =  ros::Time::now();

  if(config_.dyn_debug_verbose_on){
    cout<<indStr(1)+"DDP planning done:"<<endl;
    cout<<indStr(1)+"delta t:"<<(t_end - t_start).toSec()<<" sec"<<endl;
  }
  ind_count_--;
  return true;
}

void CallBackDslDdp::obsDetect(vector<Vector3d>& centers_encirc, vector<double>& rads_encirc, const ObsDetectionCfg& cfg){
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Detecting obstacles"<<endl;
  if(!cfg.p_laserscan_msg->ranges.size()){
    if(config_.dyn_debug_on)
      cout<<indStr(1)+"Obstacle detection not done as it seems there is no lidar message"<<endl;
    ind_count_--;
    return;
  }

  ros::Time t_start =  ros::Time::now();
  //range values out of these ranges are not correct
  double dela = cfg.p_laserscan_msg->angle_increment;
  double mina = cfg.p_laserscan_msg->angle_min;
  double maxa = cfg.p_laserscan_msg->angle_max;
  int n_skip = abs(mina + cfg.search_angle_fwd)/dela;

  // Find lidar endpoints which lie in the radius
  vector<float>::iterator it_begin = cfg.p_laserscan_msg->ranges.begin()+n_skip;
  vector<float>::iterator it_end   = cfg.p_laserscan_msg->ranges.end()-n_skip;

  std::vector<size_t> idxs_inrange;
  vector<float>::iterator it= find_if(it_begin,it_end , [=](float r){return ((r < cfg.search_radius_max)&& (r > cfg.search_radius_min) );});
  while (it != it_end) {
    idxs_inrange.push_back(std::distance(cfg.p_laserscan_msg->ranges.begin(), it));
    it = find_if(++it, it_end, [=](float r){return ((r < cfg.search_radius_max)&& (r > cfg.search_radius_min) );});
  }

  //Get the obstacle points
  vector<Vector2d> pts_cart(idxs_inrange.size());
  for(int i=0;i<pts_cart.size();i++){
    double r = cfg.p_laserscan_msg->ranges[idxs_inrange[i]];
    double a = cfg.p_laserscan_msg->angle_min + cfg.p_laserscan_msg->angle_increment*idxs_inrange[i];
    pts_cart[i] = Vector2d(r*cos(a), r*sin(a));
  }

  //find the KNN closes to center
  vector<vector<size_t>> clusters;
  double r_obs = cfg.map_cell_size/sqrt(2);

  //find cluster centers
  findKClustersWithSizeConstraint(clusters,pts_cart,r_obs, cfg.cluster_count_max, cfg.cluster_radius_max);

  //get cluster radius and center
  vector<Vector2d> centers2d_encirc;
  findCircumCircle(centers2d_encirc, rads_encirc, clusters, pts_cart);
  centers_encirc.resize(centers2d_encirc.size());
  transform(centers2d_encirc.begin(), centers2d_encirc.end(), centers_encirc.begin(),
            [](const Vector2d& c){return Vector3d(c(0), c(1), 0);});

  ros::Time t_end =  ros::Time::now();
  if(config_.dyn_debug_verbose_on){
    cout<<indStr(1)+"Obstacle detection done:"<<endl;
    cout<<indStr(1)+"delta t:"<<(t_end - t_start).toSec()<<" sec"<<endl;
  }

  ind_count_--;
}


void CallBackDslDdp::obsFindCloseToTraj(vector<Vector3d>& centers_encirc_inlidar, vector<double>& rads_encirc,const ObsDetectionCfg& cfg){
  ind_count_++;
   if(config_.dyn_debug_on)
     cout<<indStr(0)+"*Detecting obstacles"<<endl;
   if(!cfg.p_laserscan_msg->ranges.size()){
     if(config_.dyn_debug_on)
       cout<<indStr(1)+"Obstacle detection not done as it seems there is no lidar message"<<endl;
     ind_count_--;
     return;
   }

   ros::Time t_start =  ros::Time::now();

   double dela = cfg.p_laserscan_msg->angle_increment;
   double mina = cfg.p_laserscan_msg->angle_min;
   double maxa = cfg.p_laserscan_msg->angle_max;

   vector<float>::iterator it_begin = cfg.p_laserscan_msg->ranges.begin();
   vector<float>::iterator it_end   = cfg.p_laserscan_msg->ranges.end();

   for(size_t idx_node=0; idx_node < cfg.p_lidar2nodes->size(); idx_node++){
     Affine2d& lidar2node = (*cfg.p_lidar2nodes)[idx_node];
     std::vector<size_t> idxs_inrange(cfg.p_laserscan_msg->ranges.size()); //TODO replace it with filtered ranges
     std::iota(idxs_inrange.begin(), idxs_inrange.end(),0);

     //Get the obstacle points
     vector<Vector2d> pts_cart_innode(idxs_inrange.size());
     for(int i=0;i<pts_cart_innode.size();i++){
       double r = cfg.p_laserscan_msg->ranges[idxs_inrange[i]];
       double a = cfg.p_laserscan_msg->angle_min + cfg.p_laserscan_msg->angle_increment*idxs_inrange[i];
       pts_cart_innode[i] = lidar2node.inverse() * Vector2d(r*cos(a), r*sin(a));
     }

     //find the KNN closest to center
     vector<vector<size_t>> clusters;
     double r_obs = cfg.map_cell_size/sqrt(2);

     //find the nearest cluster center
     findKClustersWithSizeConstraint(clusters,pts_cart_innode,r_obs, 1, cfg.cluster_radius_max);

     //get cluster radius and center
     vector<Vector2d> centers2d_encirc_innode;
     vector<double> rads_encirc_fornodei;
     findCircumCircle(centers2d_encirc_innode, rads_encirc_fornodei, clusters, pts_cart_innode);
     centers_encirc_inlidar.resize(centers2d_encirc_innode.size());

     Vector2d c = centers2d_encirc_innode[0];
     rads_encirc.push_back(rads_encirc_fornodei[0]);
     centers_encirc_inlidar.push_back(toAffine3d(lidar2node) * Vector3d(c(0), c(1), 0));

   }


   ros::Time t_end =  ros::Time::now();
   if(config_.dyn_debug_verbose_on){
     cout<<indStr(1)+"Obstacle detection done:"<<endl;
     cout<<indStr(1)+"delta t:"<<(t_end - t_start).toSec()<<" sec"<<endl;
   }

   ind_count_--;
}



void CallBackDslDdp::rvizColorMsgEdit(std_msgs::ColorRGBA& rgba_msg, VectorXd& rgba_vec){
  rgba_msg.r = rgba_vec(0);
  rgba_msg.g = rgba_vec(1);
  rgba_msg.b = rgba_vec(2);
  rgba_msg.a = rgba_vec(3);
}

void CallBackDslDdp::rvizMarkersEdit(visualization_msgs::Marker& marker, VectorXd& prop){
  ind_count_++;
  switch(prop.size())
  {
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
      marker.scale.x = prop(4);
      break;
    case 6:
      marker.color.r = prop(0);// red
      marker.color.g = prop(1);// blue
      marker.color.b = prop(2);// green
      marker.color.a = prop(3);
      marker.scale.x = prop(4);
      marker.scale.y = prop(5);
      break;
    case 7:
      marker.color.r = prop(0);// red
      marker.color.g = prop(1);// blue
      marker.color.b = prop(2);// green
      marker.color.a = prop(3);//alpha
      marker.scale.x = prop(4);
      marker.scale.y = prop(5);
      marker.scale.z = prop(6);
      break;
    case 10: //repeat of 5
      marker.color.r = prop(0);// red
      marker.color.g = prop(1);// blue
      marker.color.b = prop(2);// green
      marker.color.a = prop(3);//alpha
      marker.scale.x = prop(4);
      break;
    case 12: //repeat of 6
      marker.color.r = prop(0);// red
      marker.color.g = prop(1);// blue
      marker.color.b = prop(2);// green
      marker.color.a = prop(3);
      marker.scale.x = prop(4);
      marker.scale.y = prop(5);
      break;
    default:
      cout<<indStr(0)+"*Error setting marker properties"<<endl;
      break;
  }
  ind_count_--;
}


void CallBackDslDdp::rvizMarkersInit(void){
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Initializing all rviz markers."<<endl;

  VectorXd prop_path_n_wp;
  marker_id_=0;

  marker_prop_obs_           = yaml_node_["marker_prop_obs"].as<VectorXd>();
  marker_prop_text_start_    = yaml_node_["marker_prop_text_start"].as<VectorXd>();
  marker_prop_text_goal_     = yaml_node_["marker_prop_text_goal"].as<VectorXd>();
  marker_prop_path_dsl_      = yaml_node_["marker_prop_path_dsl"].as<VectorXd>();
  marker_prop_wp_dsl_        = yaml_node_["marker_prop_wp_dsl"].as<VectorXd>();
  marker_prop_path_dsl_intp_ = yaml_node_["marker_prop_path_dsl_intp"].as<VectorXd>();
  marker_prop_wp_dsl_intp_   = yaml_node_["marker_prop_wp_dsl_intp"].as<VectorXd>();
  marker_prop_path_ddp_      = yaml_node_["marker_prop_path_ddp"].as<VectorXd>();
  marker_prop_wp_ddp_        = yaml_node_["marker_prop_wp_ddp"].as<VectorXd>();

  //Marker for dsl path
  marker_path_dsl_.header.frame_id = strfrm_world_;
  marker_path_dsl_.header.stamp = ros::Time();
  marker_path_dsl_.ns = "dsl_ddp_planner";
  marker_path_dsl_.id = marker_id_;
  marker_path_dsl_.type = visualization_msgs::Marker::LINE_STRIP;
  marker_path_dsl_.action = visualization_msgs::Marker::ADD;
  marker_path_dsl_.lifetime = ros::Duration(0);
  rvizMarkersEdit(marker_path_dsl_,marker_prop_path_dsl_);
  marker_id_++;

  //Marker for dsl path way points
  marker_wp_dsl_.header.frame_id = strfrm_world_;
  marker_wp_dsl_.header.stamp = ros::Time();
  marker_wp_dsl_.ns = "dsl_ddp_planner";
  marker_wp_dsl_.id = marker_id_;
  marker_wp_dsl_.type = visualization_msgs::Marker::POINTS;
  marker_wp_dsl_.action = visualization_msgs::Marker::ADD;
  marker_wp_dsl_.lifetime = ros::Duration(0);
  rvizMarkersEdit(marker_wp_dsl_,marker_prop_wp_dsl_);
  marker_id_++;

  //Marker for dsl path interpolated

  marker_path_dsl_intp_.header.frame_id = strfrm_world_;
  marker_path_dsl_intp_.header.stamp = ros::Time();
  marker_path_dsl_intp_.ns = "dsl_ddp_planner";
  marker_path_dsl_intp_.id = marker_id_;
  marker_path_dsl_intp_.type = visualization_msgs::Marker::LINE_STRIP;
  marker_path_dsl_intp_.action = visualization_msgs::Marker::ADD;
  marker_path_dsl_intp_.lifetime = ros::Duration(0);
  rvizMarkersEdit(marker_path_dsl_intp_,marker_prop_path_dsl_intp_);
  marker_id_++;

  //Marker for dsl path interpolated way points
  marker_wp_dsl_intp_.header.frame_id = strfrm_world_;
  marker_wp_dsl_intp_.header.stamp = ros::Time();
  marker_wp_dsl_intp_.ns = "dsl_ddp_planner";
  marker_wp_dsl_intp_.id = marker_id_;
  marker_wp_dsl_intp_.type = visualization_msgs::Marker::POINTS;
  marker_wp_dsl_intp_.action = visualization_msgs::Marker::ADD;
  marker_wp_dsl_intp_.lifetime = ros::Duration(0);
  rvizMarkersEdit(marker_wp_dsl_intp_,marker_prop_wp_dsl_intp_);
  marker_id_++;

  //Marker for ddp path
  marker_path_ddp_.header.frame_id = strfrm_world_;
  marker_path_ddp_.header.stamp = ros::Time();
  marker_path_ddp_.ns = "dsl_ddp_planner";
  marker_path_ddp_.id = marker_id_;
  marker_path_ddp_.type = visualization_msgs::Marker::LINE_STRIP;
  marker_path_ddp_.action = visualization_msgs::Marker::ADD;
  marker_path_ddp_.lifetime = ros::Duration(0);
  rvizMarkersEdit(marker_path_ddp_,marker_prop_path_ddp_);
  marker_id_++;

  //Marker for ddp  way points
  marker_wp_ddp_.header.frame_id = strfrm_world_;
  marker_wp_ddp_.header.stamp = ros::Time();
  marker_wp_ddp_.ns = "dsl_ddp_planner";
  marker_wp_ddp_.id = marker_id_;
  marker_wp_ddp_.type = visualization_msgs::Marker::POINTS;
  marker_wp_ddp_.action = visualization_msgs::Marker::ADD;
  marker_wp_ddp_.lifetime = ros::Duration(0);
  rvizMarkersEdit(marker_wp_ddp_,marker_prop_wp_ddp_);
  marker_id_++;

  //Marker for "start" text
  marker_text_start_.header.frame_id = strfrm_world_;
  marker_text_start_.header.stamp = ros::Time();
  marker_text_start_.ns = "dsl_ddp_planner";
  marker_text_start_.id = marker_id_;
  marker_text_start_.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker_text_start_.action = visualization_msgs::Marker::ADD;
  marker_text_start_.text="S";
  rvizMarkersEdit(marker_text_start_,marker_prop_text_start_);
  marker_text_start_.lifetime = ros::Duration(0);
  marker_id_++;

  //Marker for "goal" text
  marker_text_goal_.header.frame_id = strfrm_world_;
  marker_text_goal_.header.stamp = ros::Time();
  marker_text_goal_.ns = "dsl_ddp_planner";
  marker_text_goal_.id = marker_id_;
  marker_text_goal_.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker_text_goal_.action = visualization_msgs::Marker::ADD;
  marker_text_goal_.text="G";
  rvizMarkersEdit(marker_text_goal_,marker_prop_text_goal_);
  marker_text_goal_.lifetime = ros::Duration(0);
  marker_id_++;

  //Marker for obstacle
  marker_obs_.header.frame_id = strfrm_world_;
  marker_obs_.header.stamp = ros::Time();
  marker_obs_.ns = "dsl_ddp_planner";
  marker_obs_.id = marker_id_;
  marker_obs_.type = visualization_msgs::Marker::CYLINDER;
  marker_obs_.action = visualization_msgs::Marker::ADD;
  marker_obs_.scale.z = 0.5;
  marker_obs_.lifetime = ros::Duration(config_.dyn_ddp_loop_durn);
  rvizMarkersEdit(marker_obs_,marker_prop_obs_);
  marker_id_ += obs_cfg_.cluster_count_max;

  ind_count_--;
}

void CallBackDslDdp::rvizShowObstacles(void){
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Sending obstacle markers to rviz"<<endl;

  //remove old obstacles first

  //rvizRemoveObstacles();
  //remove start test related visualization marker
  marker_obs_.action    = visualization_msgs::Marker::ADD;

  for(int i=0; i<disks_.size();i++){
    marker_obs_.id = marker_id_+i;
    marker_obs_.scale.x = 2*disks_[i].r;
    marker_obs_.scale.y = 2*disks_[i].r;
    marker_obs_.pose.position.x = disks_[i].o(0);
    marker_obs_.pose.position.y = disks_[i].o(1);
    marker_obs_.pose.position.z = marker_obs_.scale.z/2;
    pub_vis_.publish( marker_obs_ );
  }
  ind_count_--;
}

//void CallBackDslDdp::rvizRemoveObstacles(void){
//  ind_count_++;
//  if(config_.dyn_debug_on)
//    cout<<indStr(0)+"*Removing obstacle markers from rviz"<<endl;
//
//  //remove start test related visualization marker
//  marker_obs_.action    = visualization_msgs::Marker::DELETE;
//
//  for(int i=0; i<marker_obs_ids_.size();i++){
//    marker_obs_.id = marker_obs_ids_.back();marker_obs_ids_.pop_back();
//    pub_vis_.publish( marker_obs_ );
//  }
//  ind_count_--;
//}

void CallBackDslDdp::rvizShowPath(ros::Publisher& pub_posearr, const vector<GcarState>& path_gcar,
                                  visualization_msgs::Marker& marker_path,
                                  visualization_msgs::Marker& marker_wp,
                                  VectorXd& marker_prop_path, VectorXd& marker_prop_wp){
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Sending a path to rviz"<<endl;

  geometry_msgs::PoseArray posearr;posearr.poses.resize(path_gcar.size());
   posearr.header.frame_id = strfrm_world_;
   for (int i = 0; i < path_gcar.size(); i++)
     eig2PoseMsg(posearr.poses[i], toAffine3d(toAffine2d(path_gcar[i].g)));
   pub_posearr.publish(posearr);

   if(marker_path.color.a<0.0001)
     return;

  //Properties of the negative path
  VectorXd marker_prop_path_nve, marker_prop_wp_nve;
  if(marker_prop_path.size()>7)
    marker_prop_path_nve = marker_prop_path.tail(marker_prop_path.size()/2);
  else
    marker_prop_path_nve = marker_prop_path;
  if(marker_prop_wp.size()>7)
    marker_prop_wp_nve = marker_prop_wp.tail(marker_prop_wp.size()/2);
  else
    marker_prop_wp_nve = marker_prop_wp;

  marker_path.action      = visualization_msgs::Marker::ADD;
  marker_wp.action        = visualization_msgs::Marker::ADD;
  marker_path.points.resize(path_gcar.size());
  marker_path.colors.resize(path_gcar.size());
  marker_wp.points.resize(path_gcar.size());
  marker_wp.colors.resize(path_gcar.size());

  for (int i = 0; i < path_gcar.size(); i++){
    marker_path.points[i].x = path_gcar[i].g(0,2);
    marker_path.points[i].y = path_gcar[i].g(1,2);
    marker_path.points[i].z = 0.2;

    marker_wp.points[i].x = path_gcar[i].g(0,2);
    marker_wp.points[i].y = path_gcar[i].g(1,2);
    marker_wp.points[i].z = 0.2;

    //Set path colors
    if(marker_prop_path.size()>7){
      if(path_gcar[i].v>0)
        rvizColorMsgEdit(marker_path.colors[i],marker_prop_path);
      else
        rvizColorMsgEdit(marker_path.colors[i],marker_prop_path_nve);
    }else
      rvizColorMsgEdit(marker_path.colors[i],marker_prop_path);

    //Set wp colors
    if(marker_prop_wp.size()>7){
      if(path_gcar[i].v>0)
        rvizColorMsgEdit(marker_wp.colors[i],marker_prop_wp);
      else
        rvizColorMsgEdit(marker_wp.colors[i],marker_prop_wp_nve);
    }else
      rvizColorMsgEdit(marker_wp.colors[i],marker_prop_wp);
  }

  pub_vis_.publish( marker_path );
  pub_vis_.publish( marker_wp );
  ind_count_--;
}

void CallBackDslDdp::rvizRemovePath(visualization_msgs::Marker& marker_path,
                                    visualization_msgs::Marker& marker_wp){
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Removing path from rviz"<<endl;

  //remove ddp related visualization marker
  marker_path.action      = visualization_msgs::Marker::DELETE;
  marker_wp.action        = visualization_msgs::Marker::DELETE;
  pub_vis_.publish( marker_path);
  pub_vis_.publish( marker_wp);
  ind_count_--;
}

void CallBackDslDdp::rvizShowText(ros::Publisher& pub_ps, Affine3d& pose_w2b, visualization_msgs::Marker& marker_text, VectorXd& prop_text){
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Sending text marker to rviz"<<endl;

  geometry_msgs::PoseStamped ps;
  ps.header.frame_id=strfrm_world_;
  eig2PoseMsg(ps.pose, pose_w2b);
  pub_ps.publish(ps);

  marker_text.action = visualization_msgs::Marker::ADD;
  marker_text.pose.position.x =pose_w2b.translation()(0);
  marker_text.pose.position.y =pose_w2b.translation()(1);
  marker_text.pose.position.z =0.2;

  pub_vis_.publish( marker_text);
  ind_count_--;
}

void CallBackDslDdp::rvizRemoveText(visualization_msgs::Marker& marker_text){
  ind_count_++;
  if(config_.dyn_debug_on)
    cout<<indStr(0)+"*Removing text marker from rviz"<<endl;

  //remove goal test related visualization marker
  marker_text.action    = visualization_msgs::Marker::DELETE;
  pub_vis_.publish( marker_text );
  ind_count_--;
}



//------------------------------------------------------------------------
//--------------------------------MAIN -----------------------------------
//------------------------------------------------------------------------

int main(int argc, char** argv){
  ros::init(argc,argv,"dsl_ddp_planner",ros::init_options::NoSigintHandler);
  signal(SIGINT,mySigIntHandler);

  ros::NodeHandle nh;

  CallBackDslDdp cbc;

  while(!g_shutdown_requested){
    ros::spinOnce();

    cbc.loop_rate_main_.sleep();
  }
  return 0;

}


//  //add 3 closest obstacles from the previously found obstacles if they lie in the interest area and are not repeats of the currently found obstacles
//  Vector3d c_body= pose_w2b_ddp_start_.translation();
//  std::sort(centers_encirc_oinw_prev_.begin(),
//            centers_encirc_oinw_prev_.end(),
//            [&](const Vector3d& center_obs1, const Vector3d& center_obs2){ return (center_obs1 - c_body).norm() < (center_obs2 - c_body).norm() ; });
//  vector<double> dist_ofromb_prev(centers_encirc_oinw_prev_.size());
//  transform(centers_encirc_oinw_prev_.begin(),centers_encirc_oinw_prev_.end(), dist_ofromb_prev.begin(),
//            [&](const Vector3d& c_oinw){return (c_body - c_oinw).norm();});
//
//  vector<Vector3d> centers_oinw_prev_selected;
//  size_t nobs_prev = centers_encirc_oinw_prev_.size();
//  if(nobs_prev){
//    for(size_t i=0;i<3;i++){
//      if(dist_ofromb_prev[i]< obs_search_radius_max_)
//        centers_oinw_prev_selected.push_back(centers_encirc_oinw_prev_[i]);
//    }
//  }
//
//  for(size_t i=0;i<centers_oinw_prev_selected.size();i++){
//   Vector3d& center = centers_oinw_prev_selected[i];
//   vector<Vector3d>& centers = centers_encirc_oinw;
//   vector<Vector3d>::iterator it =  find_if(centers.begin(), centers.end(), [&](const Vector3d& c){return (c-center).norm()<0.01;});
//   bool obs_repeated = it!=centers.end();
//   if(!obs_repeated)
//     centers_encirc_oinw.push_back(centers_oinw_prev_selected[i]);
//   }
//  centers_encirc_oinw_prev_ = centers_encirc_oinw;




