#ifndef GCOP_TRAJECTORY_VISUALIZER_H
#define GCOP_TRAJECTORY_VISUALIZER_H
#include "ros/ros.h"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <gcop_comm/CtrlTraj.h>
#include <tf/transform_datatypes.h>

class GcopTrajectoryVisualizer{
  private:
  visualization_msgs::Marker line_strip_;
  visualization_msgs::MarkerArray arrow_strip_;
  visualization_msgs::MarkerArray axis_strip_;
  visualization_msgs::Marker default_arrow_marker_;

  ros::NodeHandle &nh;
  ros::Publisher visualization_marker_pub_;
  ros::Publisher visualization_markerarray_pub_;

  double axis_length_;

  private:
  inline void publishLineStrip(gcop_comm::CtrlTraj &gcop_trajectory)
  {
    //Resize line strip
    line_strip_.points.resize(gcop_trajectory.N+1);
    line_strip_.header.stamp  = ros::Time::now();
    //Fill line strip points:
    for(int i = 0; i < (gcop_trajectory.N+1); i++)
    {
      line_strip_.points[i].x = gcop_trajectory.statemsg[i].basepose.translation.x;
      line_strip_.points[i].y = gcop_trajectory.statemsg[i].basepose.translation.y;
      line_strip_.points[i].z = gcop_trajectory.statemsg[i].basepose.translation.z;
    }
    visualization_marker_pub_.publish(line_strip_);
  }

  inline void publishAxis(gcop_comm::CtrlTraj &gcop_trajectory)
  {
    for(int j = 0; j < 3; j++)
    {
      //Resize line list strip:
      axis_strip_.markers[j].points.resize(2*(gcop_trajectory.N+1));
      axis_strip_.markers[j].header.stamp  = ros::Time::now();
    }
    //Fill line strip points:
    tf::Transform basepose;

    for(int i = 0; i < gcop_trajectory.N +1; i++)
    {
      //Get Base Pose
      tf::transformMsgToTF(gcop_trajectory.statemsg[i].basepose, basepose);
      const tf::Matrix3x3 &basis = basepose.getBasis();
      for(int j = 0; j < 3; j++)
      {
        //Set Base Point
        geometry_msgs::Point &point = axis_strip_.markers[j].points[2*i];//Temporary variable
        geometry_msgs::Point &point1 = axis_strip_.markers[j].points[2*i+1];//Temporary variable
        point.x = gcop_trajectory.statemsg[i].basepose.translation.x;
        point.y = gcop_trajectory.statemsg[i].basepose.translation.y;
        point.z = gcop_trajectory.statemsg[i].basepose.translation.z;

        tf::Vector3 column = basis.getColumn(j);
        point1.x = point.x + axis_length_*column.x();
        point1.y = point.y + axis_length_*column.y();
        point1.z = point.z + axis_length_*column.z();
      }
    }
    visualization_markerarray_pub_.publish(axis_strip_);
  }

  inline void publishVelocities(gcop_comm::CtrlTraj &gcop_trajectory)
  {
    arrow_strip_.markers.resize(gcop_trajectory.N+1, default_arrow_marker_);

    for(int i = 0; i < gcop_trajectory.N+1; i++)
    {
      //Set Base Point
      geometry_msgs::Point &point = arrow_strip_.markers[i].points[0];
      geometry_msgs::Point &point1 = arrow_strip_.markers[i].points[1];
      point.x = gcop_trajectory.statemsg[i].basepose.translation.x;
      point.y = gcop_trajectory.statemsg[i].basepose.translation.y;
      point.z = gcop_trajectory.statemsg[i].basepose.translation.z;

      point1.x = point.x + gcop_trajectory.statemsg[i].basetwist.linear.x;
      point1.y = point.y + gcop_trajectory.statemsg[i].basetwist.linear.y;
      point1.z = point.z + gcop_trajectory.statemsg[i].basetwist.linear.z;

      arrow_strip_.markers[i].id = i+1;
    }
    visualization_markerarray_pub_.publish(arrow_strip_);
  }

  public:
  GcopTrajectoryVisualizer(ros::NodeHandle &nh_):nh(nh_), axis_length_(0.1)
  {
    visualization_marker_pub_ = nh.advertise<visualization_msgs::Marker>("/desired_traj",5);
    visualization_markerarray_pub_ = nh.advertise<visualization_msgs::MarkerArray>("/desired_traj_array",5);

    //Set necessary default initializations:
    line_strip_.header.frame_id = "/optitrak";
    line_strip_.action = visualization_msgs::Marker::ADD;
    line_strip_.pose.orientation.w = 1.0;
    line_strip_.ns = "trajectory";
    line_strip_.id = 1;
    line_strip_.type = visualization_msgs::Marker::LINE_STRIP;
    line_strip_.scale.x = 0.02;
    line_strip_.color.b = 1.0;
    line_strip_.color.a = 1.0;

    //Axis Strip:
    axis_strip_.markers.resize(3);
    for(int i = 0; i < 3; i++)
    {
      axis_strip_.markers[i].header.frame_id = "/optitrak";
      axis_strip_.markers[i].action = visualization_msgs::Marker::ADD;
      axis_strip_.markers[i].pose.orientation.w = 1.0;
      axis_strip_.markers[i].ns = "axis";
      axis_strip_.markers[i].id = i+1;
      axis_strip_.markers[i].type = visualization_msgs::Marker::LINE_LIST;
      axis_strip_.markers[i].scale.x = 0.02;
      axis_strip_.markers[i].color.r = (i==0?1:0);
      axis_strip_.markers[i].color.g = (i==1?1:0);
      axis_strip_.markers[i].color.b = (i==2?1:0);
      axis_strip_.markers[i].color.a = 1.0;
    }

    //Arrow Strip:
    default_arrow_marker_.header.frame_id = "/optitrak";
    default_arrow_marker_.action = visualization_msgs::Marker::ADD;
    default_arrow_marker_.pose.orientation.w = 1.0;
    default_arrow_marker_.ns = "velocities";
    default_arrow_marker_.id = 1;
    default_arrow_marker_.type = visualization_msgs::Marker::ARROW;
    default_arrow_marker_.scale.x = 0.01;
    default_arrow_marker_.scale.y = 0.015;
    default_arrow_marker_.color.r = 1.0;
    default_arrow_marker_.color.b = 1.0;
    default_arrow_marker_.color.a = 1.0;
    default_arrow_marker_.points.resize(2);

  }

  ~GcopTrajectoryVisualizer()
  {
    visualization_marker_pub_.shutdown();
    visualization_markerarray_pub_.shutdown();
  }

  void publishTrajectory(gcop_comm::CtrlTraj &gcop_trajectory)
  {
    ROS_ASSERT(gcop_trajectory.N == gcop_trajectory.statemsg.size()-1);
    publishLineStrip(gcop_trajectory);
    publishAxis(gcop_trajectory);
    publishVelocities(gcop_trajectory);
  }
};

#endif // GCOP_TRAJECTORY_VISUALIZER_H
