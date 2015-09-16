#ifndef _VS_EVALUATE_H_
#define _VS_EVALUATE_H_

#include <vector>

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>

class VSEvaluate
{
public:
  VSEvaluate(ros::NodeHandle nh, ros::NodeHandle nh_private);

private:
  void handleImage(const sensor_msgs::ImageConstPtr& msg);
  void handleDepth(const sensor_msgs::ImageConstPtr& msg);
  void handleCameraInfo(const sensor_msgs::CameraInfoConstPtr& msg);

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

  bool has_intrinsics, has_depth;
  ros::Time img_time_stamp;

  cv::Mat current_image, current_depth, im_goal;
  cv::Mat K;
  cv::Mat distcoeff;
  Eigen::Matrix3d K_eig;
  ros::NodeHandle nh, nh_private;

  ros::Subscriber camera_info_sub;
  ros::Subscriber image_sub;
  ros::Subscriber depth_sub;
};

#endif
