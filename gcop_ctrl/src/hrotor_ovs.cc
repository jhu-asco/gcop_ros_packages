#include "gcop_ctrl/hrotor_ovs.h"
#include "gcop_ctrl/imagecost.h"
#include "gcop_ctrl/utils.h"


#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <cv_bridge/cv_bridge.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <ctime>

#include <gcop/body3d.h>
#include <gcop/hrotor.h>
//#include <gcop/utils.h>
#include <gcop/controltparam.h>
#include <gcop/flatoutputtparam.h>
#include <gcop/gndocp.h>
#include <gcop/body3dcost.h>

#include <visualization_msgs/Marker.h>

#include <tf/transform_listener.h>
#include <tf_conversions/tf_eigen.h>

using namespace std;
using namespace Eigen;

typedef gcop::GnDocp<Body3dState, 12, 6> Body3dGn;
typedef gcop::GnDocp<Body3dState, 12, 4> HrotorGn;

HrotorOVS::HrotorOVS(ros::NodeHandle nh, ros::NodeHandle nh_private) :
  nh(nh),
  nh_private(nh_private),
  has_intrinsics(false),
  final_time(4),
  hrotor_iterations(200),
  imageQ(.01),
  use_velocities(true),
  iterate_cont(false),
  send_trajectory(false),
  gtv(nh, "world")
{
  std::string im_goal_filename;
  if (!nh_private.getParam ("im_goal_filename", im_goal_filename))
    im_goal_filename = "/home/matt/catkin_ws/src/gcop_ros_packages/gcop_ctrl/data/frame0006.jpg";
  if (!nh_private.getParam ("world_frame", world_frame))
    world_frame = "world";
  if (!nh_private.getParam ("body_frame", body_frame))
    body_frame = "body";
  if(!nh_private.getParam ("use_depth_mm", use_depth_mm))
    use_depth_mm = false;

  Mat im_goal_color = imread(im_goal_filename);
  cvtColor( im_goal_color, im_goal, CV_BGR2GRAY );
  imshow("Goal Image", im_goal);
  waitKey(10);

  // TODO: RIGHT NOW YOU HAVE TO EDIT cam_vel_transform in find_stable_final_pose FOR THIS TO WORK
  cam_transform << 0,  0,  1, 0.125,
                   -1,  0,  0, 0,
                   0,  -1,  0, 0.09,
                   0,  0,  0, 1;

  
  // Setup cam transform
  /*
  tf::StampedTransform cam_tf;
  try
  {
    bool result = tflistener.waitForTransform("pixhawk", "camera",
                                       ros::Time(0), ros::Duration(1.0));

    tflistener.lookupTransform("pixhawk", "camera",
                             ros::Time(0), cam_tf);
    tf::Vector3 o =  cam_tf.getOrigin();
    Eigen::Quaterniond qe;
    tf::quaternionTFToEigen(cam_tf.getRotation(), qe);
    cam_transform.topLeftCorner<3,3>() = qe.toRotationMatrix();
    cam_transform.block<3,1>(0,3) = Vector3d(o.x(), o.y(), o.z());
  }
  catch(tf::TransformException ex){
    ROS_ERROR("%s",ex.what());
    ros::Duration(1.0).sleep();
  }
  */
  start_tf.setOrigin(tf::Vector3(0,0,0));
  start_tf.setRotation(tf::Quaternion(0,0,0,1));  

  dynamic_reconfigure::Server<gcop_ctrl::HrotorOVSConfig>::CallbackType dyn_cb_f;
  dyn_cb_f = boost::bind(&HrotorOVS::cbReconfig, this, _1, _2);
  dyn_server.setCallback(dyn_cb_f);
  depth_sub = nh.subscribe<sensor_msgs::Image>("depth", 1,
    &HrotorOVS::handleDepth,
    this, ros::TransportHints().tcpNoDelay());
  image_sub = nh.subscribe<sensor_msgs::Image>("image", 1,
    &HrotorOVS::handleImage,
    this, ros::TransportHints().tcpNoDelay());
  camera_info_sub = nh.subscribe<sensor_msgs::CameraInfo>("camera_info", 1000,
    &HrotorOVS::handleCameraInfo,
    this, ros::TransportHints().tcpNoDelay());

  traj_pub = nh.advertise<gcop_comm::CtrlTraj>("/hrotor_ovs/traj", 1);
  //traj_marker_pub = nh.advertise<visualization_msgs::Marker>("/hrotor_ovs/traj_marker", 1);

  ovs_timer = nh.createTimer(ros::Rate(2), &HrotorOVS::ovsCallback, this); 

  ROS_INFO("Initialized");
}

void HrotorOVS::cbReconfig(gcop_ctrl::HrotorOVSConfig &config, uint32_t level)
{
  imageQ = config.imageQ;
  final_time = config.final_time;
  hrotor_iterations = config.hrotor_iterations;
  use_velocities = config.use_velocities;
  if(iterate_cont && !config.iterate_cont)
  {
    config.send_trajectory = false;
    config.iterate = false;
  }
  if(config.iterate && !config.send_trajectory)
  {
    generateTrajectory(current_image, current_depth, im_goal); 
    config.iterate = false;
  }
  else if(config.send_trajectory && !config.iterate)
  {
    if(!iterate_cont)
    {
      traj_pub.publish(traj_msg);
      config.send_trajectory = false;
      ROS_INFO("Sent trajectory");
    }
  }
  else if (config.save_goal_image)
  {
    saveGoalImage();
    config.save_goal_image = false;
    ROS_INFO("Saved goal image");
  }
  if(config.iterate_cont)
  {
    ovs_timer.start();
  }
  else
  {
    ovs_timer.stop();
  }
  iterate_cont = config.iterate_cont;
  send_trajectory = config.send_trajectory; 
}

void HrotorOVS::ovsCallback(const ros::TimerEvent&)
{
  generateTrajectory(current_image, current_depth, im_goal); 
  if(send_trajectory)
  {
    traj_pub.publish(traj_msg);
  }
}


void HrotorOVS::saveGoalImage()
{
  im_goal = current_image;
  time_t t = time(0);   // get time now
  struct tm * now = localtime( & t );
  std::string im_goal_filename(
    std::string("~/.ros/ovs_goal_")+std::to_string(now->tm_year+1900)
    +"_"+std::to_string(now->tm_mon + 1)+"_"
    + std::to_string(now->tm_mday)+"_"+std::to_string(now->tm_hour)+"_"
    + std::to_string(now->tm_min)+"_"+std::to_string(now->tm_sec));
  std::cout << im_goal_filename << " position=" << start_tf.getOrigin() << std::endl;
  imwrite(im_goal_filename, im_goal);
  imshow("Goal Image", im_goal);
  waitKey(10);
}

void HrotorOVS::handleDepth(const sensor_msgs::ImageConstPtr& msg)
{
  if(has_intrinsics)
  {
    cv_bridge::CvImageConstPtr cvImg = cv_bridge::toCvCopy(msg);
    current_depth = cvImg->image;
    namedWindow("Input Depth");
    Mat depth8;
    if(use_depth_mm)
    {
      current_depth.convertTo(depth8, CV_8UC1);
    }
    else
    {
      cv::convertScaleAbs(current_depth, depth8, 255/20.);
    }
    imshow("Input Depth", depth8);
    /*
    if(use_depth_mm)
    {
      std::cout << "middle depth=" << current_depth.at<int16_t>(120, 160)/100. << std::endl;
    }
    else
    {
      std::cout << "middle depth=" << current_depth.at<float>(120, 160) << std::endl;
    }
    */
    waitKey(1);
  }
}

void HrotorOVS::handleImage(const sensor_msgs::ImageConstPtr& msg)
{
  if(has_intrinsics)
  {
    cv_bridge::CvImageConstPtr cvImg = cv_bridge::toCvCopy(msg);
    current_image = cvImg->image;
    //img_time_stamp = msg->header.stamp;

    // Get pose where picture was taken
    start_tf.setOrigin(tf::Vector3(0,0,0));
    start_tf.setRotation(tf::Quaternion(0,0,0,1));
    try
    {
      bool result = tflistener.waitForTransform(world_frame, body_frame,
                                         //img_time_stamp, ros::Duration(1.0));
                                         ros::Time(0), ros::Duration(1.0));
  
      tflistener.lookupTransform(world_frame, body_frame,
                               //img_time_stamp, start_tf);
                               ros::Time(0), start_tf);
    }
    catch(tf::TransformException ex){
      ROS_ERROR("%s",ex.what());
      ros::Duration(1.0).sleep();
    }
    namedWindow("Input Image");
    imshow("Input Image", current_image);
    waitKey(1);
  }
  
}

void HrotorOVS::handleCameraInfo(const sensor_msgs::CameraInfoConstPtr& msg)
{
  ROS_INFO("camera_info received");

  K_eig << msg->K[0], msg->K[1], msg->K[2],
                 msg->K[3], msg->K[4], msg->K[5],
                 msg->K[6], msg->K[7], msg->K[8];
  K = (Mat_<double>(3,3) << K_eig(0,0), K_eig(0,1), K_eig(0,2),
                            K_eig(1,0), K_eig(1,1), K_eig(1,2),
                            K_eig(2,0), K_eig(2,1), K_eig(2,2));
  distcoeff = (Mat_<double>(5,1) << msg->D[0], msg->D[1], msg->D[2], msg->D[3], msg->D[4]);
  std::cout << "K=" << K << std::endl;
  has_intrinsics = true;
  camera_info_sub.shutdown();
}

void HrotorOVS::ovsHrotor(std::vector<Eigen::Vector3d> pts3d, std::vector<Eigen::Vector2d> pts2d, 
  Eigen::Matrix3d K, vector<gcop::Body3dState>& xs,  vector<Vector4d>& us, int N, double tf)
{
  double h = tf/N; // time-step

  // system
  gcop::Hrotor sys;
  sys.m = 1.764;

  gcop::Body3dState xfs;
  xfs.R.setIdentity();
  xfs.p.setZero();
  xfs.w.setZero();
  xfs.v.setZero();

  // cost 
  ImageCost<4> cost(pts3d, pts2d, K, sys, tf, xfs, cam_transform);  
  cost.imageQ = imageQ;//.012;
  cost.imageQf = .0;
  cost.Q.setZero();
  cost.Qf.setZero();
  cost.Q(6,6) = 4; cost.Q(7,7) = 4; cost.Q(8,8) = 10;
  cost.Q(9,9) = 1; cost.Q(10,10) = 1; cost.Q(11,11) = 1;
  //double vcost = 100;
  //cost.Qf(0,0) = 0; cost.Qf(1,1) = 0; cost.Qf(2,2) = 0;
  //cost.Qf(3,3) = 0; cost.Qf(4,4) = 0; cost.Qf(5,5) = 0;
  //cost.Qf(6,6) = 0; cost.Qf(7,7) = 0; cost.Qf(8,8) = 0;
  //cost.Qf(9,9) = vcost; cost.Qf(10,10) = vcost; cost.Qf(11,11) = vcost;

  cost.R(0,0) = 3e4; cost.R(1,1) = 3e4; cost.R(2,2) = 3e4; 
  cost.R(3,3) = 0.8; 
  cost.UpdateGains();

  // times
  vector<double> ts(N+1);
  for (int k = 0; k <= N; ++k)
    ts[k] = k*h;

  int ny = 4;
  int nder = 4;
  int Nk = 14;

  // initial controls (e.g. hover at one place)
  us.resize(N);
  for (int i = 0; i < N; ++i) {
    us[i].head(3).setZero();
    us[i][3] = 9.81*sys.m;
  }

  gcop::FlatOutputTparam<gcop::Body3dState, 12, 4, Dynamic> ctp(sys, ny, Nk, nder, nder-1, true);
  VectorXd s((Nk-1-nder)*ny);
  ctp.To(s, ts, xs, us);
  ctp.From(ts, xs, us, s);

  double J = 0;
  for (int k = 0; k < N; ++k) {
    double h = ts[k+1] - ts[k];
    J += cost.L(ts[k], xs[k], us[k], h);
  }
  J += cost.L(ts[N], xs[N], us[N-1], 0);
  cout << "Initial error: " <<  J << endl;

  HrotorGn gn(sys, cost, ctp, ts, xs, us, NULL, false);
  gn.numdiff_stepsize = 1e-5;

  double cur_cost = 999999;
  double last_cost = cur_cost;
  for (int i = 0; i < hrotor_iterations; ++i) 
  {    
    std::clock_t start = std::clock();
    gn.Iterate();
    cur_cost = min(cur_cost, gn.J);
    if(last_cost-cur_cost < 1e-3)
      break;
    last_cost = cur_cost;
    cout << "Cost " << i << " = " << gn.J << "\t Time = " << 1000.*(std::clock()-start)/CLOCKS_PER_SEC
      << endl;
    //getchar();
  }
  cout << "done!" << endl;
}

void HrotorOVS::getKeypointsAndDescriptors(Mat& im, std::vector<KeyPoint>& kps, cv::Mat& desc_gpu)
{
  SURF surf_gpu;
  surf_gpu(im, Mat(), kps, desc_gpu);
}

void HrotorOVS::filterKeypointMatches( std::vector < std::vector< DMatch > >& matches, 
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

void HrotorOVS::filterKeypointsEpipolarConstraint(const std::vector<cv::Point2f>& pts1,
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

void HrotorOVS::getFilteredFeatureMatches(Mat im1, Mat im2, std::vector<Point2f>& ps1_out, 
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
  std::cout << "Initial Kps: " << kps1.size() << " " << kps2.size() << std::endl;

  BFMatcher matcher;
  matcher.knnMatch(desc_gpu1, desc_gpu2, matches, 2);
  filterKeypointMatches(matches, good_matches, 0.85);

  std::vector<Point2f> ps1, ps2;
  for(unsigned int i = 0; i < good_matches.size(); i++)
  {
    Point2f p1 = kps1[good_matches[i].queryIdx].pt;
    Point2f p2 = kps2[good_matches[i].trainIdx].pt;
    ps1.push_back(p1);
    ps2.push_back(p2);
  }
  std::cout << "Ratio Test Matches: " << ps1.size() << " " << ps2.size() << std::endl;

  //std::vector<Point2f> ps1_filt, ps2_filt;
  //filterKeypointsEpipolarConstraint(ps1, ps2, ps1_filt, ps2_filt);
  //std::cout << "Epi Constraint Matches: " << ps1_filt.size() << " " << ps2_filt.size() << std::endl;

  ps1_out = ps1;
  ps2_out = ps2;
}

void HrotorOVS::generateTrajectory(Mat im, Mat depths, Mat im_goal)
{
  vector<gcop::Body3dState> xs1;
  vector<Vector6d> us1;
  vector<Vector4d> hr_us1;
  int N = 64;
  double tf = final_time;
  Eigen::Matrix4d attitude_transform = Eigen::Matrix4d::Identity();
  attitude_transform.setIdentity();  
  xs1.resize(N+1);
  us1.resize(N);
  for(int i = 0; i < xs1.size(); i++)
  {
    xs1[i].Clear();
  }

  // Match features between images
  std::vector<cv::Point2f> ps, ps_goal;
  getFilteredFeatureMatches(im, im_goal, ps, ps_goal);
  // visualize matches 
  Mat match_img;
  std::vector<std::vector<cv::DMatch>> matches;
  std::vector<cv::KeyPoint> kps, kps_goal;
  matches.resize(ps.size());
  for(int i = 0; i < ps.size(); i++)
  {
    kps.push_back(KeyPoint(ps[i].x, ps[i].y, 1));
    kps_goal.push_back(KeyPoint(ps_goal[i].x, ps_goal[i].y, 1));
    matches[i].push_back(cv::DMatch(i,i, 1));
  }
  drawMatches(im, kps, im_goal, kps_goal, matches, match_img); 
  namedWindow("OVS Matches");
  imshow("OVS Matches", match_img);
  waitKey(1);

  // Put points in flat frame
  tf::Matrix3x3 rotmat = start_tf.getBasis();
  tf::Vector3 result(0,0,0);
  rotmat.getEulerYPR(result[2], result[1], result[0]);

  tf::Quaternion vs_in_opti(0, 0, result[2]);
  Eigen::Quaterniond equat, vs_in_opti_eig;
  tf::quaternionTFToEigen(vs_in_opti.inverse()*start_tf.getRotation(), equat);
  tf::quaternionTFToEigen(vs_in_opti, vs_in_opti_eig);
  
  // attitude_transform is the rotation of the quad in the vs frame
  attitude_transform.topLeftCorner<3,3>() = equat.toRotationMatrix();

  // Backproject points in current image to 3D
  std::vector<Eigen::Vector3d> pts3d, inlier_pts3d;
  std::vector<Eigen::Vector2d> pts2d, inlier_pts2d, pts2d_start;

  double fx = K_eig(0,0);
  double fy = K_eig(1,1);
  double cx = K_eig(0,2);
  double cy = K_eig(1,2);

  for(int i = 0; i < ps.size(); i++)
  {
    int yidx = round(ps[i].y);
    int xidx = round(ps[i].x);
    if(yidx < 0 || yidx >= depths.rows || xidx < 0 || xidx >= depths.cols)
      continue;
    double depth;
    if(use_depth_mm)
    {
      depth = depths.at<int16_t>(yidx, xidx)/100.;
    }
    else
    {
      depth = depths.at<float>(yidx, xidx);
    }
    std::cout << depth << std::endl;
    if(depth <= 0 || std::isnan(depth))
      continue; 
    Eigen::Vector4d pt3d(depth*(ps[i].x-cx)/fx, depth*(ps[i].y-cy)/fy, depth, 1);
    pts3d.push_back((attitude_transform*cam_transform*pt3d).head<3>());
    pts2d.push_back(Eigen::Vector2d(ps_goal[i].x, ps_goal[i].y));
    pts2d_start.push_back(Eigen::Vector2d(ps[i].x, ps[i].y));
  }

  // Do Optimization
  std::cout << "Finding final pose..." << std::endl;
  std::vector<int> inliers;
  std::clock_t start = std::clock();
  find_stable_final_pose_ransac(pts3d, pts2d, K_eig, cam_transform, Eigen::MatrixXd::Identity(3,3), 
    xs1.back(), 200, inliers);
  std::cout << "RANSAC inliers " << inliers.size() << "/" << pts3d.size() 
    << ", time=" << 1000.*(std::clock()-start)/CLOCKS_PER_SEC << std::endl;
  std::cout << "Relative final position: " << xs1.back().p.transpose() << std::endl;

  Mat inlier_match_img;
  std::vector<std::vector<cv::DMatch>> inlier_matches;
  std::vector<cv::KeyPoint> inlier_kps, inlier_kps_goal;
  inlier_matches.resize(inliers.size());
  for(int i = 0; i < inliers.size(); i++)
  {
    inlier_pts2d.push_back(pts2d[inliers[i]]);
    inlier_pts3d.push_back(pts3d[inliers[i]]);
    inlier_kps_goal.push_back(KeyPoint(pts2d[inliers[i]](0), pts2d[inliers[i]](1), 1));
    inlier_kps.push_back(KeyPoint(pts2d_start[inliers[i]](0), pts2d_start[inliers[i]](1), 1));
    inlier_matches[i].push_back(cv::DMatch(i,i, 1));
  }
  drawMatches(im, inlier_kps, im_goal, inlier_kps_goal, inlier_matches, inlier_match_img); 
  imshow("OVS Inlier Matches", inlier_match_img);
  waitKey(1);

  xs1[0].R = attitude_transform.topLeftCorner<3,3>(); 
  // TODO: Set velocity in VS frame here
  //xs1[0].v =  
  Vector3d logRi;
  Matrix3d Ri;
  //gcop::SO3::Instance().log(logRf, xs1.back().R);
  for(int i = 1; i < xs1.size()-1; i++)
  {
    xs1[i].p = (double(i)/xs1.size())*xs1.back().p;
    //gcop::SO3::Instance().exp(xs1[i].R, logRf*(double(i)/xs1.size()));
    //gcop::SO3::Instance().exp(xs1[i].R, logRf*(double(i)/xs1.size()));
    gcop::SO3::Instance().log(logRi, xs1[0].R.transpose()*xs1.back().R);
    gcop::SO3::Instance().exp(Ri, logRi*(double(i)/xs1.size()));
    xs1[i].R = xs1[0].R*Ri;
  }
  ovsHrotor(inlier_pts3d, inlier_pts2d, K_eig, xs1, hr_us1, N, tf);

  // Create trajectory message
  Eigen::Matrix4d vs_in_opti_tf;
  vs_in_opti_tf.setIdentity();
  vs_in_opti_tf.topLeftCorner<3,3>() = vs_in_opti_eig.toRotationMatrix();
  vs_in_opti_tf.block<3,1>(0,3) = Eigen::Vector3d(start_tf.getOrigin().x(),  
                                    start_tf.getOrigin().y(), start_tf.getOrigin().z());
  traj_msg.N = xs1.size()-1;
  traj_msg.statemsg.resize(xs1.size());
  double max_sp = 0;
  for(int i = 0; i < xs1.size(); i++)
  {
    traj_msg.time.push_back(i*tf/N);

    gcop_comm::State state;
    Eigen::Vector3d tjpt =
      (vs_in_opti_tf*Eigen::Vector4d(xs1[i].p(0), xs1[i].p(1), xs1[i].p(2), 1)).head<3>();
    state.basepose.translation.x = tjpt(0);
    state.basepose.translation.y = tjpt(1);
    state.basepose.translation.z = tjpt(2);
    //std::cout << "xs["<<i<<"].p=" << tjpt.transpose() << std::endl;

    Eigen::Quaterniond qx(xs1[i].R);
    qx = vs_in_opti_eig*qx;
    state.basepose.rotation.w = qx.w();
    state.basepose.rotation.x = qx.x();
    state.basepose.rotation.y = qx.y();
    state.basepose.rotation.z = qx.z();

    Eigen::Vector3d v = vs_in_opti_eig.toRotationMatrix()*xs1[i].v;
    max_sp = max(v.norm(), max_sp);
    if(use_velocities)
    {
      state.basetwist.linear.x = v(0);
      state.basetwist.linear.y = v(1);
      state.basetwist.linear.z = v(2);
    } 
    else
    {
      state.basetwist.linear.x = 0;
      state.basetwist.linear.y = 0;
      state.basetwist.linear.z = 0;
    }
    std::cout << "xs["<<i<<"].v=" << v.transpose() << std::endl;
    traj_msg.statemsg[i] = state;
  }
  std::cout << "max v=" << max_sp << std::endl;

  // Publish trajectory visualization
  /*
  visualization_msgs::Marker traj_marker_msg;
  traj_marker_msg.id = 1;
  traj_marker_msg.ns = "hrotor_ovs";
  traj_marker_msg.points.resize(xs1.size());
  traj_marker_msg.header.frame_id = "optitrak";
  traj_marker_msg.header.stamp = ros::Time::now();
  traj_marker_msg.action = visualization_msgs::Marker::ADD;
  traj_marker_msg.pose.orientation.w = 1;
  traj_marker_msg.type = visualization_msgs::Marker::LINE_STRIP;
  traj_marker_msg.scale.x = 0.05;
  traj_marker_msg.color.b = 1;
  traj_marker_msg.color.a = 1;

  for(int i = 0; i < tjpts.size(); i++)
  {
    traj_marker_msg.points.at(i).x = tjpts[i](0);
    traj_marker_msg.points.at(i).y = tjpts[i](1);
    traj_marker_msg.points.at(i).z = tjpts[i](2);
  }
  
  traj_marker_pub.publish(traj_marker_msg);
  */
  gtv.publishTrajectory(traj_msg);
  ROS_INFO("Sent trajectory marker");
}

int main(int argc, char** argv)
{
  ros::init (argc, argv, "hrotor_ovs");
  ros::NodeHandle nh;
  ros::NodeHandle nh_private("~");
  HrotorOVS hovs(nh, nh_private);
  ros::spin ();
  return 0;
}
