/*
 * obs_detection_lidar.h
 *
 *  Created on: Mar 10, 2016
 *      Author: subhransu
 */

#ifndef OBS_DETECTION_LIDAR_H
#define OBS_DETECTION_LIDAR_H

#include <memory>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>

using namespace std;
using namespace Eigen;

namespace obs_detection_lidar {

struct ObsDetectionCfg{
  double search_radius_max;
  double search_radius_min;      //! The laser points
  double search_angle_fwd;       //! restrict the search angle of the lidar data
  int cluster_count_max;         //! max number of obstacles to be returned
  double cluster_radius_max;     //! the max cluster radius
  double map_cell_size;         //! cell size of the map
  shared_ptr<sensor_msgs::LaserScan> p_laserscan_msg;//! Pointer to the laserscan message
  shared_ptr<vector<Affine3d>> p_lidar2nodes;        //! Pointer to vector of transformation for lidar frame to nodei frame
};

class ObsDetectionLidar {
public:
  ObsDetectionLidar();
  virtual ~ObsDetectionLidar();
};

} /* namespace dsl_ddp_planner */

#endif /* OBS_DETECTION_LIDAR_H */
