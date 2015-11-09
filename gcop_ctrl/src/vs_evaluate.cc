#include "gcop_ctrl/vs_evaluate.h"
#include "gcop_ctrl/imagecost.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <cv_bridge/cv_bridge.h>

#include <fstream>
#include <iomanip>
#include <iostream>
using namespace std;
using namespace Eigen;

VSEvaluate::VSEvaluate(ros::NodeHandle nh, ros::NodeHandle nh_private) :
  nh(nh),
  nh_private(nh_private),
  has_intrinsics(false),
  has_depth(false)
{
  std::string im_goal_filename;
  if (!nh_private.getParam ("im_goal_filename", im_goal_filename))
    im_goal_filename = "im_goal.jpg";

  Mat im_goal_color = imread(im_goal_filename);
  cvtColor( im_goal_color, im_goal, CV_BGR2GRAY );
  imshow("Goal Image", im_goal);
  waitKey(10);

  depth_sub = nh.subscribe<sensor_msgs::Image>("depth", 1000,
    &VSEvaluate::handleDepth,
    this, ros::TransportHints().tcpNoDelay());
  image_sub = nh.subscribe<sensor_msgs::Image>("image", 1000,
    &VSEvaluate::handleImage,
    this, ros::TransportHints().tcpNoDelay());
  camera_info_sub = nh.subscribe<sensor_msgs::CameraInfo>("camera_info", 1000,
    &VSEvaluate::handleCameraInfo,
    this, ros::TransportHints().tcpNoDelay());

  ROS_INFO("Initialized");
}

void VSEvaluate::handleDepth(const sensor_msgs::ImageConstPtr& msg)
{
  if(has_intrinsics)
  {
    cv_bridge::CvImageConstPtr cvImg = cv_bridge::toCvCopy(msg);
    current_depth = cvImg->image;
    has_depth = true;
  }
}

void VSEvaluate::handleImage(const sensor_msgs::ImageConstPtr& msg)
{
  if(has_intrinsics)
  {
    cv_bridge::CvImageConstPtr cvImg = cv_bridge::toCvCopy(msg);
    undistort(cvImg->image, current_image, K, distcoeff);

    
    // Match features between images
    std::vector<cv::Point2f> ps, ps_goal;
    getFilteredFeatureMatches(current_image, im_goal, ps, ps_goal);

    // visualize matches & compute error
    double error = 0;
    Mat match_img;
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::KeyPoint> kps, kps_goal;
    matches.resize(ps.size());
    for(int i = 0; i < ps.size(); i++)
    {
      error += cv::norm(ps[i]-ps_goal[i])/ps.size();
      kps.push_back(KeyPoint(ps[i].x, ps[i].y, 1));
      kps_goal.push_back(KeyPoint(ps_goal[i].x, ps_goal[i].y, 1));
      matches[i].push_back(cv::DMatch(i,i, 1));
    }
    std::cout << (msg->header.stamp - img_time_stamp).toSec() << " " << error << std::endl;
    drawMatches(current_image, kps, im_goal, kps_goal, matches, match_img); 
    imshow("OVS Matches", match_img);
    waitKey(10);

    /*
    // Backproject points in current image to 3D
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d, pts2d_start;

    double fx = K_eig(0,0);
    double fy = K_eig(1,1);
    double cx = K_eig(0,2);
    double cy = K_eig(1,2);
    Mat distcoeffcvPnp = (Mat_<double>(4,1) << 0, 0, 0, 0);
    Mat Rvec, t;

    for(int i = 0; i < ps.size(); i++)
    {
      int yidx = round(ps[i].y);
      int xidx = round(ps[i].x);
      if(yidx < 0 || yidx >= current_depth.rows || xidx < 0 || xidx >= current_depth.cols)
        continue;
      double depth = current_depth.at<float>(yidx, xidx);
      if(depth <= 0 || std::isnan(depth))
        continue; 
      cv::Point3f pt3d(depth*(ps[i].x-cx)/fx, depth*(ps[i].y-cy)/fy, depth);
      pts3d.push_back(pt3d);
      pts2d.push_back(ps_goal[i]);
      pts2d_start.push_back(ps[i]);
    }

    std::vector<int> inliers;
    solvePnPRansac(pts3d, pts2d, K, distcoeffcvPnp, Rvec, t, false, 300, 8, 100, inliers);

    // visualize matches & compute error
    double error = 0;
    Mat match_img;
    std::vector<std::vector<cv::DMatch>> matches;
    std::vector<cv::KeyPoint> kps, kps_goal;
    matches.resize(inliers.size());
    for(int i = 0; i < inliers.size(); i++)
    {
      error += cv::norm(pts2d[inliers[i]]-pts2d_start[inliers[i]])/pts2d.size();
      kps.push_back(KeyPoint(pts2d_start[i].x, pts2d_start[i].y, 1));
      kps_goal.push_back(KeyPoint(pts2d[i].x, pts2d[i].y, 1));
      matches[i].push_back(cv::DMatch(i,i, 1));
    }
    std::cout << (msg->header.stamp - img_time_stamp).toSec() << " " << error << std::endl;
    drawMatches(current_image, kps, im_goal, kps_goal, matches, match_img); 
    imshow("OVS Matches", match_img);
    waitKey(10);
    */
   
  }
}

void VSEvaluate::handleCameraInfo(const sensor_msgs::CameraInfoConstPtr& msg)
{
  ROS_INFO("camera_info received");

  K_eig << msg->K[0], msg->K[1], msg->K[2],
                 msg->K[3], msg->K[4], msg->K[5],
                 msg->K[6], msg->K[7], msg->K[8];
  K = (Mat_<double>(3,3) << K_eig(0,0), K_eig(0,1), K_eig(0,2),
                            K_eig(1,0), K_eig(1,1), K_eig(1,2),
                            K_eig(2,0), K_eig(2,1), K_eig(2,2));
  distcoeff = (Mat_<double>(5,1) << msg->D[0], msg->D[1], msg->D[2], msg->D[3], msg->D[4]);
  img_time_stamp = msg->header.stamp;
  has_intrinsics = true;
  camera_info_sub.shutdown();
}

void VSEvaluate::getKeypointsAndDescriptors(Mat& im, std::vector<KeyPoint>& kps, cv::Mat& desc_gpu)
{
  SURF surf_gpu;
  surf_gpu(im, Mat(), kps, desc_gpu);
}

void VSEvaluate::filterKeypointMatches( std::vector < std::vector< DMatch > >& matches, 
  std::vector< DMatch >& filtered_matches, double match_ratio)
{
  for(unsigned int i = 0; i < matches.size(); i++)
  {
    if(matches[i][0].distance < match_ratio*matches[i][1].distance)
    {
      filtered_matches.push_back(matches[i][0]);
    }
  }
}

void VSEvaluate::filterKeypointsEpipolarConstraint(const std::vector<cv::Point2f>& pts1,
  const std::vector<cv::Point2f>& pts2, std::vector<cv::Point2f>& pts1_out, 
  std::vector<cv::Point2f>& pts2_out)
{
  assert(pts1.size() == pts2.size());

  pts1_out.clear();
  pts2_out.clear();
  std::vector<unsigned char> status;
  cv::Mat fMat = findFundamentalMat(pts1, pts2, CV_FM_RANSAC, 3, .99, status);
  for(int i = 0; i < status.size(); i++)
  {
    if(status[i])
    {
      pts1_out.push_back(pts1[i]);
      pts2_out.push_back(pts2[i]);
    }
  }
}

void VSEvaluate::getFilteredFeatureMatches(Mat im1, Mat im2, std::vector<Point2f>& ps1_out, 
  std::vector<Point2f>& ps2_out)
{
  ps1_out.clear();
  ps2_out.clear();

  std::vector<KeyPoint> kps1, kps2;
  Mat  desc_gpu1, desc_gpu2;
  std::vector < std::vector< DMatch > > matches;
  std::vector< DMatch > good_matches;

  getKeypointsAndDescriptors(im1, kps1, desc_gpu1);
  getKeypointsAndDescriptors(im2, kps2, desc_gpu2);
  //std::cout << "Initial Kps: " << kps1.size() << " " << kps2.size() << std::endl;

  BFMatcher matcher;
  matcher.knnMatch(desc_gpu1, desc_gpu2, matches, 2);
  filterKeypointMatches(matches, good_matches, 0.8);

  std::vector<Point2f> ps1, ps2;
  for(unsigned int i = 0; i < good_matches.size(); i++)
  {
    Point2f p1 = kps1[good_matches[i].queryIdx].pt;
    Point2f p2 = kps2[good_matches[i].trainIdx].pt;
    ps1.push_back(p1);
    ps2.push_back(p2);
  }
  //std::cout << "Ratio Test Matches: " << ps1.size() << " " << ps2.size() << std::endl;

  std::vector<Point2f> ps1_filt, ps2_filt;
  filterKeypointsEpipolarConstraint(ps1, ps2, ps1_filt, ps2_filt);
  //std::cout << "Epi Constraint Matches: " << ps1_filt.size() << " " << ps2_filt.size() << std::endl;

  ps1_out = ps1_filt;
  ps2_out = ps2_filt;
}

int main(int argc, char** argv)
{
  ros::init (argc, argv, "vs_evaluate");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  VSEvaluate vse(nh, nh_private);
  ros::spin ();
  return 0;
}
