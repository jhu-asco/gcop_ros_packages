#ifndef _HROTOR_OVS_H_
#define _HROTOR_OVS_H_

#include <vector>

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/Vector3.h>

#include <gcop/body3d.h>

//ROS dynamic reconfigure
#include <dynamic_reconfigure/server.h>
#include <gcop_ctrl/HrotorOVSConfig.h>

#include <gcop_comm/CtrlTraj.h>

#include <tf/transform_listener.h>

#include "gcop_comm/gcop_trajectory_visualizer.h"

class HrotorOVS
{
public:
  HrotorOVS(ros::NodeHandle nh, ros::NodeHandle nh_private);

private:
  void handleDepth(const sensor_msgs::ImageConstPtr& msg);
  void handleImage(const sensor_msgs::ImageConstPtr& msg);
  void handleImage2(const sensor_msgs::ImageConstPtr& msg);
  void handleCameraInfo(const sensor_msgs::CameraInfoConstPtr& msg);
  void handleVelocity(const geometry_msgs::Vector3ConstPtr& msg);
  void cbReconfig(gcop_ctrl::HrotorOVSConfig &config, uint32_t level);

  void ovsHrotor(std::vector<Eigen::Vector3d> pts3d, std::vector<Eigen::Vector2d> pts2d, 
    Eigen::Matrix3d K, std::vector<gcop::Body3dState>& xs,  std::vector<Eigen::Vector4d>& us, int N, 
    double tf);
  void ovsB3d(std::vector<Eigen::Vector3d> pts3d, std::vector<Eigen::Vector2d> pts2d, 
    Eigen::Matrix3d K, std::vector<gcop::Body3dState>& xs,  
    std::vector<Eigen::Matrix<double, 6, 1>>& us, int N, 
    double tf);
  void filterKeypointMatches( std::vector < std::vector< cv::DMatch > >& matches, 
    std::vector< cv::DMatch >& filtered_matches, double match_ratio);
  void filterKeypointsEpipolarConstraint(const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2, std::vector<cv::Point2f>& pts1_out, 
    std::vector<cv::Point2f>& pts2_out);
  void getFilteredFeatureMatches(cv::Mat im1, cv::Mat im2, 
    std::vector<cv::Point2f>& ps1_out, 
    std::vector<cv::Point2f>& ps2_out);
  void getKeypointsAndDescriptors(cv::Mat& im, std::vector<cv::KeyPoint>& kps, 
    cv::Mat& desc_gpu);
  void generateTrajectory(cv::Mat im, cv::Mat depths, cv::Mat im_goal);
  void saveGoalImage();
  void ovsCallback(const ros::TimerEvent&);

  GcopTrajectoryVisualizer gtv;

  bool has_intrinsics;
  ros::Time img_time_stamp;

  Eigen::Vector3d current_velocity;
  cv::Mat current_image, current_image2, current_depth, im_goal;
  cv::Mat K;
  cv::Mat distcoeff;
  Eigen::Matrix3d K_eig;
  Eigen::Matrix4d cam_transform;
  ros::NodeHandle nh, nh_private;

  ros::Subscriber camera_info_sub;
  ros::Subscriber image_sub;
  ros::Subscriber image_sub2;
  ros::Subscriber depth_sub;
  ros::Subscriber velocity_sub;

  ros::Publisher traj_pub;
  ros::Publisher traj_marker_pub;

  ros::Timer ovs_timer;

  dynamic_reconfigure::Server<gcop_ctrl::HrotorOVSConfig> dyn_server;

  gcop_comm::CtrlTraj traj_msg;
  
  double final_time;
  int b3d_iterations;
  int hrotor_iterations;
  double imageQ;
  bool use_velocities;
  bool use_depth_mm;
  bool iterate_cont;
  bool send_trajectory;
  std::string world_frame, body_frame;

  tf::StampedTransform start_tf;
  tf::TransformListener tflistener;
};

#endif
