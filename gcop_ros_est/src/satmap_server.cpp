//rampage_logdata_viz.cpp
//Author: Subhransu Mishra
//Note: This ros node is used for displaying the gps, visual odometry
//      wheel odometry, satellite images and other log data 
//TODO: 1) read gps data and convert that into meters relative to somepoint
//      2) Import a satellite image, scale it and display it so that it 
//         matches the scale in meters
//      3) Read the encoder data and pass it through a model to display it 
//         on the map 
//      4) Plot the deadreconing visual odometry result on the map
//      5) Find the scale of the visual odometry somehow from the stereo correspondenses
//      6) Parse the lidar data to form a point cloud at regular intervals
//      7) Study the effect of the radio commands side by side with the encoder data
//      8) take in the imu messages and see if it can be used with the visual odometry to 
//         make it better
//      9) 
//

#include <ros/ros.h>
#include <ros/package.h>
#include <resource_retriever/retriever.h>

//ROS & OpenCV
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>

//ROS msgs
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/String.h>

//Other includes
#include <iostream>
#include <signal.h>

//gcop ros utils
#include <gcop_ros_utils/eigen_ros_conv.h>
#include <gcop_ros_utils/eig_splinterp.h>
#include <gcop_ros_utils/yaml_eig_conv.h>
#include <gcop_ros_utils/resource_url_parser.h>

//operations on path
#include <boost/filesystem/path.hpp>

//-------------------------------------------------------------------------
//-----------------------NAME SPACES ---------------------------------
//-------------------------------------------------------------------------
using namespace std;

//-------------------------------------------------------------------------
//-----------------------GLOBAL VARIABLES ---------------------------------
//-------------------------------------------------------------------------
sig_atomic_t g_shutdown_requested=0;
visualization_msgs::Marker g_marker;
//------------------------------------------------------------------------
//-----------------------FUNCTION DEFINITIONS ----------------------------
//------------------------------------------------------------------------

void mySigIntHandler(int signal)
{
  g_shutdown_requested=1;
  cout<<"\nshutdown requested\n"<<endl;
}

void initMarker(YAML::Node& node)
{
  g_marker.header.frame_id = node["strfrm_mesh"].as<string>();
  g_marker.header.stamp = ros::Time();
  g_marker.ns = "rampage";
  g_marker.id = 0;
  g_marker.type = visualization_msgs::Marker::MESH_RESOURCE;
  g_marker.action = visualization_msgs::Marker::ADD;
  g_marker.pose.position.x = 0;
  g_marker.pose.position.y = 0;
  g_marker.pose.orientation.x = 0.0;
  g_marker.pose.orientation.y = 0.0;
  g_marker.pose.orientation.z = 0;
  g_marker.pose.orientation.w = 1;
  g_marker.scale.x = 1;
  g_marker.scale.y = 1;
  g_marker.scale.z = 1;
  g_marker.color.a = 0.0; // Don't forget to set the alpha!
  g_marker.color.r = 0.0;
  g_marker.color.g = 0.0;
  g_marker.color.b = 0.0;
  g_marker.mesh_use_embedded_materials = true;
  //g_marker.lifetime = ros::Duration(1);
  g_marker.mesh_resource = node["strurl_mesh"].as<string>();
}

//------------------------------------------------------------------------
//-----------------------------MAIN LOOP ---------------------------------
//------------------------------------------------------------------------

int main(int argc, char** argv)
{
ros::init(argc,argv,"rampage_map_server",ros::init_options::NoSigintHandler);
signal(SIGINT,mySigIntHandler);

ros::NodeHandle nh;
ros::NodeHandle nh_p("~");

YAML::Node yaml_node_params, yaml_node_map;
string strfile_params;nh_p.getParam("strfile_params",strfile_params);
cout<<"loading yaml param file into yaml_node"<<endl;
yaml_node_params = YAML::LoadFile(strfile_params);

//Find map.yaml file and load it to a yaml node
string strurl_map_yaml = yaml_node_params["strurl_map"].as<string>();
resource_retriever::Retriever ret;
resource_retriever::MemoryResource resource;
resource = ret.get(strurl_map_yaml);
yaml_node_map = YAML::Load(reinterpret_cast<const char*>(resource.data.get()));

//Find map jpg file
string strfile_map_yaml = getPathFromPkgUrl(strurl_map_yaml);
cout<<"satmap_yaml_file:"<<strfile_map_yaml<<endl;
boost::filesystem::path path_map_yaml(strfile_map_yaml);
string strfile_map_img = path_map_yaml.parent_path().string() +string("/") +yaml_node_map["image"].as<string>();
cout<<"satmap_img_file:"<<strfile_map_img<<endl;

//Read map jpg file
cv::Mat mat_jhu1 = cv::imread(strfile_map_img,CV_LOAD_IMAGE_GRAYSCALE);
if(mat_jhu1.data==NULL)
  ROS_ERROR("Couldn't read the map image");

mat_jhu1 *= 0.39;
mat_jhu1 = 100 - mat_jhu1;
cv::Mat mat_jhu2;
cv::flip(mat_jhu1,mat_jhu2,0);

nav_msgs::OccupancyGrid og_jhu;
og_jhu.header.frame_id=yaml_node_params["strfrm_map"].as<string>();
og_jhu.header.stamp = ros::Time::now();
og_jhu.info.height = mat_jhu2.rows;
og_jhu.info.width = mat_jhu2.cols;
og_jhu.info.resolution =0.36;//meter/pixel
double res_tru_x = 0.365;
double res_tru_y = 0.354;
cv::Point LL_pix(9.4,714.6);
cv::Point UL_pix(19.5,10.1);
cv::Point UR_pix(649.5,9.5);
cv::Point LR_pix(654,713);

geometry_msgs::Pose pose_jhu;
//pose_jhu.position.x = -og_jhu.info.resolution*og_jhu.info.width/2;
//pose_jhu.position.y = -og_jhu.info.resolution*og_jhu.info.height/2;
pose_jhu.position.x = -res_tru_x*LL_pix.x;
pose_jhu.position.y = -res_tru_y*(og_jhu.info.height-LL_pix.y);
og_jhu.info.origin.position= pose_jhu.position;
og_jhu.info.origin.orientation= pose_jhu.orientation;
int len = mat_jhu2.rows*mat_jhu2.cols;
og_jhu.data.reserve(mat_jhu2.rows*mat_jhu2.cols);
og_jhu.data.assign(mat_jhu2.data, mat_jhu2.data+len) ;

string strtop_map_sat = yaml_node_params["strtop_map_sat"].as<string>();
ros::Publisher pub_oc = nh.advertise<nav_msgs::OccupancyGrid>(strtop_map_sat,1,true);
pub_oc.publish(og_jhu);

string strtop_mesh = yaml_node_params["strtop_mesh"].as<string>();
ros::Publisher pub_vis = nh.advertise<visualization_msgs::Marker>( "mesh_latrobe", 1, true );
initMarker(yaml_node_params);
pub_vis.publish( g_marker);

ros::Rate loop_rate(0.5);

while(ros::ok() && !g_shutdown_requested)
{
  ros::spinOnce();
  loop_rate.sleep();
}
return 0;

}
