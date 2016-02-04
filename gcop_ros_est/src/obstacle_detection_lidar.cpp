/*
 * obstacle_detection_lidar.cpp
 *
 *  Created on: Jan 14, 2016
 *      Author: subhransu
 */
//ROS
#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

// ROS standard messages
#include <std_msgs/String.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/LaserScan.h>

//ROS dynamic reconfigure
#include <dynamic_reconfigure/server.h>
//#include <obstacle_detection_lidar/ObsLidarConfig.h>

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
#include <vector>
#include <iterator>

//yaml
#include <yaml-cpp/yaml.h>

//local includes
#include <gcop_ros_utils/eigen_ros_conv.h>
#include <gcop_ros_utils/yaml_eig_conv.h>

// Eigen Includes
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "miniball.hpp"
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
void findKClustersWithSizeConstraint(vector<vector<size_t>>& clusters, const vector<Vector2d>& pts_polar,double robs,int K, double rcmax)
{
  if(!pts_polar.size())
    return;

  //  Make a vector of ranges and convert ptpol to cartesian
  vector<int> clid(pts_polar.size(),-1);//-1 means cluster numbers have not been assigned
  vector<double> ranges(pts_polar.size());
  vector<size_t> idxs_noclid(pts_polar.size());//indexes of all the obstacles that don't have a cluster id
  vector<Vector2d> pts_cart(pts_polar.size());
  for(size_t i=0;i<pts_polar.size();i++){
    pts_cart[i] = Vector2d(pts_polar[i](0)*cos(pts_polar[i](1)), pts_polar[i](0)*sin(pts_polar[i](1)));
    ranges[i] = pts_polar[i](0);
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
    Vector2d pt_polar_closest = pts_polar[idxs_noclid[idx_innoclid_closest]];

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
    for(size_t i=0;i<pts_polar.size();i++){
      if(clid[i]==-1)
        idxs_noclid.push_back(i);
    }
    k++;
  }
}

typedef std::vector<std::vector<double> >::const_iterator PointIterator;
typedef std::vector<double>::const_iterator CoordIterator;
typedef Miniball::Miniball <Miniball::CoordAccessor<PointIterator, CoordIterator> > MB;

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
//------------------------------------------------------------------------
//-----------------------------CALLBACK CLASS ----------------------------
//------------------------------------------------------------------------

class CallBackObsLidar
{
public:
  typedef Transform<double,2,Affine> Transform2d;
  typedef sensor_msgs::LaserScan::_ranges_type::const_iterator RangesConstIt;
public:
  CallBackObsLidar();
  ~CallBackObsLidar();

public:
  ros::Rate loop_rate_main_;

private:
  ros::NodeHandle nh_, nh_p_;
  YAML::Node yaml_node_;
  bool debug_on_;
  int ind_count_;
  string ind_str_;
  //gcop_ctrl::ObsLidarConfig config_;
  //dynamic_reconfigure::Server<obstacle_detection_lidar::ObsLidarConfig> dyn_server_;

  ros::Subscriber sub_lidar_;
  ros::Publisher pub_;
  ros::Timer timer_;

  string strtop_lidar_;
  string strtop_output_;

  tf::TransformBroadcaster tf_br_;
  tf::TransformListener tf_lr_;

  //obstacle properties
  double obs_search_radius_;
  int obs_cluster_count_max_;
  double obs_cluster_radius_max_;
  double obs_map_cell_size_;

private:
  //void cbReconfig(obstacle_detection_lidar::ObsLidarConfig &config, uint32_t level);
  string indStr(int count_extra);
  void setupFromYaml(void);
  void setupTopicsAndNames(void);
  void initSubsPubsAndTimers(void);

  void cbLidar(const sensor_msgs::LaserScanConstPtr& pmsg_lidar);
};

CallBackObsLidar::CallBackObsLidar():
                     nh_p_("~"),
                     loop_rate_main_(1000),
                     ind_count_(-1),
                     ind_str_("  "){
  cout<<"**************************************************************************"<<endl;
  cout<<"**************************************************************************"<<endl;
  cout<<"*Entering constructor of cbc"<<endl;

  //Setup YAML reading and parsing
  string strfile_params;nh_p_.getParam("strfile_params",strfile_params);
  cout<<"loading yaml param file into yaml_node"<<endl;
  yaml_node_ = YAML::LoadFile(strfile_params);

  //setup dynamic reconfigure
  //dynamic_reconfigure::Server<obstacle_detection_lidar::ObsLidarConfig>::CallbackType dyn_cb_f;
  //dyn_cb_f = boost::bind(&CallBackObsLidar::cbReconfig, this, _1, _2);
  //dyn_server_.setCallback(dyn_cb_f);

  // Setup general settings from yaml
  setupFromYaml();
  cout<<"Setup topic names from yaml file done"<<endl;

  // Setup topic names
  setupTopicsAndNames();
  cout<<"Setup topic names from yaml file done"<<endl;

  //Setup Subscriber, publishers and Timers
  initSubsPubsAndTimers();
  cout<<"Initialized publishers, subscriber and timers"<<endl;

}

CallBackObsLidar::~CallBackObsLidar(){

}

/*
void
CallBackObsLidar::cbReconfig(obstacle_detection_lidar::ObsLidarConfig &config, uint32_t level){
  static bool first_time=true;

  if(!first_time)
  {
    //loop rate setting
    if(config_.dyn_loop_rate_main != config.dyn_loop_rate_main)
      loop_rate_main_= ros::Rate(config.dyn_loop_rate_main);
  }
  else
  {
    cout<<"First time in reconfig. Setting config from yaml"<<endl;

    //general settings
    config.dyn_debug_on           = yaml_node_["debug_on"].as<bool>();
    config.dyn_loop_rate_main     = yaml_node_["loop_rate_main"].as<double>();
    loop_rate_main_= ros::Rate(config.dyn_loop_rate_main);

    first_time = false;
  }
  config_ = config;
}
*/

string
CallBackObsLidar::indStr(int count_extra){
  string ind;
  for(int i=0; i<ind_count_+count_extra; i++)
    ind = ind+ ind_str_;
  return ind;
}

void
CallBackObsLidar::setupFromYaml(void){
  debug_on_ = yaml_node_["debug_on"].as<bool>();

  obs_search_radius_       = yaml_node_["obs_search_radius"].as<double>();
  obs_cluster_count_max_   = yaml_node_["obs_cluster_count_max"].as<int>();
  obs_cluster_radius_max_  = yaml_node_["obs_cluster_radius_max"].as<double>();
  obs_map_cell_size_       = yaml_node_["obs_map_cell_size"].as<double>();
}

void
CallBackObsLidar::setupTopicsAndNames(void){
  // Input Topics
  strtop_lidar_ = yaml_node_["strtop_lidar"].as<string>();

  // Output Topics
  strtop_output_ = yaml_node_["strtop_output"].as<string>();
}

void
CallBackObsLidar::initSubsPubsAndTimers(void){
  //Setup subscribers
  sub_lidar_  = nh_.subscribe(strtop_lidar_, 1, &CallBackObsLidar::cbLidar, this);

  //Setup Publishers

  //Setup timers


  //Setup service servers
  //Setup service clients

}

void
CallBackObsLidar::cbLidar(const sensor_msgs::LaserScanConstPtr& pmsg_lidar){
  ind_count_++;
  if(debug_on_)
      cout<<indStr(0)+"Got a new lidar message"<<endl;

  //range values outof these ranges are not correct
  float rng_min = 1.0;
  float rng_max = 60;

  int n = pmsg_lidar->ranges.size();
  std::vector<size_t> idxs_inrange;

  // Find ranges which lie in the radius
  RangesConstIt it_begin = pmsg_lidar->ranges.begin();
  RangesConstIt it_end   = pmsg_lidar->ranges.end();
  RangesConstIt it= find_if(it_begin,it_end , [=](float r){return ((r < obs_search_radius_)&& (r > rng_min) && (r < rng_max));});
  while (it != pmsg_lidar->ranges.end()) {
     idxs_inrange.push_back(std::distance(pmsg_lidar->ranges.begin(), it));
     it = find_if(++it, it_end, [=](float r){return ((r < obs_search_radius_)&& (r > rng_min) && (r < rng_max));});
  }

  //create an image with obstacles in range
  int radius_pix = ceil(0.5 + obs_search_radius_/obs_map_cell_size_);
  cv::Size2i size(2*radius_pix-1,2*radius_pix-1);
  cout<<"size of image:"<<size<<endl;
  cv::Point2i pt2i_center(radius_pix-1,radius_pix-1);
  cv::Mat img_obs(size,CV_8UC1, cv::Scalar(0));
  vector<Vector2d> pts_polar(idxs_inrange.size());
  vector<Vector2d> pts_cart(idxs_inrange.size());
  for(int i=0;i<pts_polar.size();i++){
    double range = pmsg_lidar->ranges[idxs_inrange[i]];
    double angle = pmsg_lidar->angle_min + pmsg_lidar->angle_increment*idxs_inrange[i];
    pts_polar[i] = Vector2d(range,angle);
    pts_cart[i] = Vector2d(pts_polar[i](0)*cos(pts_polar[i](1)), pts_polar[i](0)*sin(pts_polar[i](1)));
    cv::Point2i pt(round(range*cos(angle)/obs_map_cell_size_),round(range*sin(angle)/obs_map_cell_size_));
    img_obs.at<uchar>(pt+pt2i_center) = 100;
  }

  //find the KNN closes to center
  vector<vector<size_t>> clusters;
  double r_obs = obs_map_cell_size_/sqrt(2);

  findKClustersWithSizeConstraint(clusters,pts_polar,r_obs, obs_cluster_count_max_, obs_cluster_radius_max_);
  cout<<"total number of clusters detected:"<<clusters.size()<<endl;

  //get cluster radius and center
  vector<Vector2d> pts_cart_center_circle;
  vector<double>   radiuses_circle;
  findCircumCircle(pts_cart_center_circle, radiuses_circle, clusters, pts_cart);

  for(int i=0; i< radiuses_circle.size();i++){
    cout<<"n_points:"<<clusters[i].size()<<endl;
    cout<<"rad:"<<radiuses_circle[i]<<endl;
    cout<<"center:"<<pts_cart_center_circle[i].transpose()<<endl<<endl;
  }

  ind_count_--;
}

//------------------------------------------------------------------------
//--------------------------------MAIN -----------------------------------
//------------------------------------------------------------------------

int main(int argc, char** argv)
{
  ros::init(argc,argv,"obstacle_detection_lidar",ros::init_options::NoSigintHandler);
  signal(SIGINT,mySigIntHandler);

  ros::NodeHandle nh;
  CallBackObsLidar cbc;

  while(!g_shutdown_requested)
  {
    ros::spinOnce();
    cbc.loop_rate_main_.sleep();
  }
  return 0;
}






