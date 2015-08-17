#include "gcop_ctrl/hrotor_ovs.h"
#include "gcop_ctrl/imagecost.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/nonfree/gpu.hpp>
#include <cv_bridge/cv_bridge.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <gcop/body3d.h>
#include <gcop/hrotor.h>
//#include <gcop/utils.h>
#include <gcop/controltparam.h>
#include <gcop/flatoutputtparam.h>
#include <gcop/gndocp.h>
#include <gcop/body3dcost.h>

#include <gcop_comm/CtrlTraj.h>

#include <visualization_msgs/Marker.h>

#include <tf/transform_listener.h>

using namespace std;
using namespace Eigen;

typedef gcop::GnDocp<Body3dState, 12, 6> Body3dGn;
typedef gcop::GnDocp<Body3dState, 12, 4> HrotorGn;

HrotorOVS::HrotorOVS(ros::NodeHandle nh, ros::NodeHandle nh_private) :
  nh(nh),
  nh_private(nh_private),
  has_intrinsics(false)
{
  std::string im_goal_filename;
  if (!nh_private.getParam ("im_goal_filename", im_goal_filename))
    im_goal_filename = "im_goal.png";

  Mat im_goal_color = imread(im_goal_filename);
  cvtColor( im_goal_color, im_goal, CV_BGR2GRAY );

  cam_transform << 1,  0,  0, 0,
                   0,  0,  1, 0,
                   0, -1,  0, 0,
                   0,  0,  0, 1;

  //TODO: add param for ce or gn

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
  traj_marker_pub = nh.advertise<visualization_msgs::Marker>("/hrotor_ovs/traj_marker", 1);
}

void HrotorOVS::cbReconfig(gcop_ctrl::HrotorOVSConfig &config, uint32_t level)
{
  if(config.iterate)
  {
    generateAndSendTrajectory(current_image, current_depth, im_goal); 
    config.iterate = false;
  }
}

void HrotorOVS::handleDepth(const sensor_msgs::ImageConstPtr& msg)
{
  if(has_intrinsics)
  {
    cv_bridge::CvImageConstPtr cvImg = cv_bridge::toCvShare(msg);
    current_depth = cvImg->image;
  }
}

void HrotorOVS::handleImage(const sensor_msgs::ImageConstPtr& msg)
{
  if(has_intrinsics)
  {
    cv_bridge::CvImageConstPtr cvImg = cv_bridge::toCvShare(msg);
    current_image = cvImg->image;
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
  has_intrinsics = true;
  camera_info_sub.shutdown();
}

void HrotorOVS::ovsHrotor(std::vector<Eigen::Vector3d> pts3d, std::vector<Eigen::Vector2d> pts2d, Eigen::Matrix3d K, vector<gcop::Body3dState>& xs,  vector<Vector4d>& us, int N, double tf)
{
  double h = tf/N; // time-step

  // system
  gcop::Hrotor sys;

  gcop::Body3dState xfs;
  xfs.R.setIdentity();
  xfs.p.setZero();
  xfs.w.setZero();
  xfs.v.setZero();

  // cost 
  ImageCost<4> cost(pts3d, pts2d, K, sys, tf, xfs, cam_transform);  
  cost.imageQ = .001;
  cost.Q.setZero();
  cost.Qf.setZero();
  cost.Q(6,6) = 4; cost.Q(7,7) = 4; cost.Q(8,8) = 10;
  cost.Q(9,9) = .1; cost.Q(10,10) = .1; cost.Q(11,11) = .1;
  double vcost = 100;
  cost.Qf(0,0) = 0; cost.Qf(1,1) = 0; cost.Qf(2,2) = 0;
  cost.Qf(3,3) = 0; cost.Qf(4,4) = 0; cost.Qf(5,5) = 0;
  cost.Qf(6,6) = 0; cost.Qf(7,7) = 0; cost.Qf(8,8) = 0;
  cost.Qf(9,9) = vcost; cost.Qf(10,10) = vcost; cost.Qf(11,11) = vcost;

  cost.R(0,0) = .000; cost.R(1,1) = .000; cost.R(2,2) = .000; 
  cost.R(3,3) = 0.5; 
  cost.UpdateGains();

  // times
  vector<double> ts(N+1);
  for (int k = 0; k <= N; ++k)
    ts[k] = k*h;

  int ny = 4;
  int nder = 3;
  int Nk = 11;
  vector<double> tks(Nk+1);
  for (int k = 0; k <=Nk; ++k)
    tks[k] = k*(tf/Nk);

  for(int i = 0; i < 3; i++)
  {
    xs[i].R.setIdentity();
    xs[i].w.setZero();
    xs[i].v.setZero();
    xs[i].p.setZero();
  }

  // initial controls (e.g. hover at one place)
  us.resize(N);
  for (int i = 0; i < N; ++i) {
    us[i].head(3).setZero();
    us[i][3] = 9.81*sys.m;
  }

  gcop::FlatOutputTparam<gcop::Body3dState, 12, 4, Dynamic> ctp(sys, ny, Nk, nder, true);
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
  gn.numdiff_stepsize = 4e-3;

  for (int i = 0; i < 20; ++i) 
  {    
    std::clock_t start = std::clock();
    gn.Iterate();
    cout << "Cost " << i << " = " << gn.J << "\t Time = " << 1000.*(std::clock()-start)/CLOCKS_PER_SEC
      << endl;
    //getchar();
  }
  cout << "done!" << endl;
}

void HrotorOVS::ovsB3d(std::vector<Eigen::Vector3d> pts3d, std::vector<Eigen::Vector2d> pts2d, 
  Eigen::Matrix3d K, vector<gcop::Body3dState>& xs,  vector<Vector6d>& us, int N, double tf)
{
  double h = tf/N; // time-step

  // system
  gcop::Body3d<> sys;

  gcop::Body3dState xfs;
  xfs.R.setIdentity();
  xfs.p.setZero(); 
  xfs.w.setZero();
  xfs.v.setZero();

  // cost 
  ImageCost<6> cost(pts3d, pts2d, K, sys, tf, xfs, cam_transform);  
  cost.Q.setZero();
  cost.Qf.setZero();
  cost.Q(9,9) = 0; cost.Q(10,10) = 0; cost.Q(11,11) = 0;

  double vcost = 0;
  cost.Qf(0,0) = 3e4; cost.Qf(1,1) = 3e4; cost.Qf(2,2) = 0;
  cost.Qf(3,3) = 0; cost.Qf(4,4) = 0; cost.Qf(5,5) = 0;
  cost.Qf(6,6) = 0; cost.Qf(7,7) = 0; cost.Qf(8,8) = 0;
  cost.Qf(9,9) = vcost; cost.Qf(10,10) = vcost; cost.Qf(11,11) = vcost;
  //cost.Qf(6,6) = 0; cost.Qf(7,7) = 0; cost.Qf(8,8) = 0;
  //cost.Qf(9,9) = 0; cost.Qf(10,10) = 0; cost.Qf(11,11) = 0;

  cost.R(0,0) = .000; cost.R(1,1) = .000; cost.R(2,2) = .000; cost.R(3,3) = .00000;
  //cost.R(0,0) = .0005; cost.R(1,1) = .005; cost.R(2,2) = .000; cost.R(3,3) = .0000;
  cost.UpdateGains();

  // times
  vector<double> ts(N+1);
  for (int k = 0; k <= N; ++k)
    ts[k] = k*h;

  int ny = 6;
  int nder = 2;
  int Nk = 4;
  vector<double> tks(Nk+1);
  for (int k = 0; k <=Nk; ++k)
    tks[k] = k*(tf/Nk);

  // states
  xs.resize(N+1);
  for(int i = 0; i < N+1; i++)
  {
    xs[i].R.setIdentity();
    xs[i].w.setZero();
    xs[i].v.setZero();
    xs[i].p.setZero();
  }

  // initial controls (e.g. hover at one place)
  us.resize(N);
  for (int i = 0; i < N; ++i) {
    us[i].head(3).setZero();
    us[i][3] = 9.81*sys.m;
  }

  gcop::FlatOutputTparam<gcop::Body3dState, 12, 6, Dynamic> ctp(sys, ny, Nk, nder);
  VectorXd s((Nk-1)*ny);
  ctp.To(s, ts, xs, us);
  ctp.From(ts, xs, us, s);

  double J = 0;
  for (int k = 0; k < N; ++k) {
    double h = ts[k+1] - ts[k];
    J += cost.L(ts[k], xs[k], us[k], h);
  }
  J += cost.L(ts[N], xs[N], us[N-1], 0);
  cout << "Initial error: " <<  J << endl;

  Body3dGn gn(sys, cost, ctp, ts, xs, us);

  for (int i = 0; i < 300; ++i) 
  {    
    std::clock_t start = std::clock();
    gn.Iterate();
    cout << "Cost " << i << " = " << gn.J << "\t Time = " << 1000.*(std::clock()-start)/CLOCKS_PER_SEC
      << endl;
    //getchar();
  }
  cout << "done!" << endl;
}

void HrotorOVS::getKeypointsAndDescriptors(Mat& im, std::vector<KeyPoint>& kps, gpu::GpuMat& desc_gpu)
{
  gpu::GpuMat kps_gpu, im_gpu(im);

  gpu::SURF_GPU surf_gpu;
  surf_gpu(im_gpu, gpu::GpuMat(), kps_gpu, desc_gpu);
  surf_gpu.downloadKeypoints(kps_gpu, kps);
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
  gpu::GpuMat  desc_gpu1, desc_gpu2;
  std::vector < std::vector< DMatch > > matches;
  std::vector< DMatch > good_matches;

  getKeypointsAndDescriptors(im1, kps1, desc_gpu1);
  getKeypointsAndDescriptors(im2, kps2, desc_gpu2);

  gpu::BFMatcher_GPU matcher;
  matcher.knnMatch(desc_gpu1, desc_gpu2, matches, 2);
  filterKeypointMatches(matches, good_matches, 0.5);

  std::vector<Point2f> ps1, ps2;
  for(unsigned int i = 0; i < good_matches.size(); i++)
  {
    Point2f p1 = kps1[good_matches[i].queryIdx].pt;
    Point2f p2 = kps2[good_matches[i].trainIdx].pt;
    ps1.push_back(p1);
    ps2.push_back(p2);
  }

  std::vector<Point2f> ps1_filt, ps2_filt;
  filterKeypointsEpipolarConstraint(ps1, ps2, ps1_filt, ps2_filt);

  ps1_out = ps1_filt;
  ps2_out = ps2_filt;
}

void HrotorOVS::generateAndSendTrajectory(Mat im, Mat depths, Mat im_goal)
{
  vector<gcop::Body3dState> xs1, xs2;
  vector<Vector6d> us1, us2;
  vector<Vector4d> hr_us1, hr_us2;
  int N = 64;
  double tf = 8.0;

  // Match features between images
  std::vector<cv::Point2f> ps, ps_goal;
  getFilteredFeatureMatches(im, im_goal, ps, ps_goal);

  // Backproject points in current image to 3D
  std::vector<Eigen::Vector3d> pts3d;
  std::vector<Eigen::Vector2d> pts2d;

  double fx = K_eig(0,0);
  double fy = K_eig(1,1);
  double cx = K_eig(0,2);
  double cy = K_eig(1,2);

  for(int i = 0; i < ps.size(); i++)
  {
    double depth = depths.at<float>(ps[i].y, ps[i].x);
    if(depth == 0)
      continue; 
    Eigen::Vector4d pt3d(depth*(ps[i].x-cx)/fx, depth*(ps[i].y-cy)/fy, depth, 1);
    pts3d.push_back((cam_transform*pt3d).head<3>());
    pts2d.push_back(Eigen::Vector2d(ps_goal[i].x, ps_goal[i].y));
  }

  // Do Optimization
  ovsB3d(pts3d, pts2d, K_eig, xs1, us1, N, tf);
  ovsHrotor(pts3d, pts2d, K_eig, xs1, hr_us1, N, tf);

  // Transform traj with motion capture initial pos
  static tf::TransformListener tflistener;
  tf::StampedTransform start_tf;
  tflistener.lookupTransform("pixhawk", "optitrak", ros::Time::now(), start_tf);

  // Publish trajectory message
  gcop_comm::CtrlTraj traj_msg;
  traj_msg.N = N;
  for(int i = 0; i < xs1.size(); i++)
  {
    traj_msg.time.push_back(i*tf/N);
    gcop_comm::State state;
    state.basepose.translation.x = xs1[i].p(0) + start_tf.getOrigin().x();
    state.basepose.translation.y = xs1[i].p(1) + start_tf.getOrigin().y();
    state.basepose.translation.z = xs1[i].p(2) + start_tf.getOrigin().z();
    Eigen::Quaterniond qx(xs1[i].R);
    state.basepose.rotation.w = qx.w();
    state.basepose.rotation.x = qx.x();
    state.basepose.rotation.y = qx.y();
    state.basepose.rotation.z = qx.z();
    state.basetwist.linear.x = xs1[i].v(0);
    state.basetwist.linear.y = xs1[i].v(1);
    state.basetwist.linear.z = xs1[i].v(2);
    traj_msg.statemsg.push_back(state);
  }
  traj_pub.publish(traj_msg);

  // Publish trajectory visualization
  visualization_msgs::Marker traj_marker_msg;
  traj_marker_msg.id = 1;
  traj_marker_msg.ns = "hrotor_ovs";
  traj_marker_msg.points.resize(N);
  traj_marker_msg.header.frame_id = "optitrak";
  traj_marker_msg.header.stamp = ros::Time::now();
  traj_marker_msg.action = visualization_msgs::Marker::ADD;
  traj_marker_msg.pose.orientation.w = 1;
  traj_marker_msg.type = visualization_msgs::Marker::LINE_STRIP;
  traj_marker_msg.scale.x = 0.05;
  traj_marker_msg.color.b = 1;
  traj_marker_msg.color.a = 1;

  for(int i = 0; i < xs1.size(); i++)
  {
    traj_marker_msg.points.at(i).x = xs1[i].p(0) + start_tf.getOrigin().x();
    traj_marker_msg.points.at(i).y = xs1[i].p(1) + start_tf.getOrigin().y();
    traj_marker_msg.points.at(i).z = xs1[i].p(2) + start_tf.getOrigin().z();
  }
  
  traj_marker_pub.publish(traj_marker_msg);
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
