#ifndef QROTORIDMODELCONTROL_H
#define QROTORIDMODELCONTROL_H
#include <gcop/qrotoridmodelcost.h>
#include <gcop/qrotoridgndocp.h>
#include <gcop/uniformsplinetparam.h>
#include <gcop/utils.h>
#include <gcop_comm/CtrlTraj.h>//msg for publishing ctrl trajectory
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Pose.h>
#include <Eigen/Eigenvalues>
#include <ros/ros.h>
#include <gcop_comm/gcop_trajectory_visualizer.h>
#include <gcop/params.h>

using namespace gcop;
using namespace Eigen;
using namespace std;

class QRotorIDModelControl
{
  protected:
    //typedef GnDocp<QRotorIDState, 15, 4, 13, 19> QRotorGn;
    typedef UniformSplineTparam<QRotorIDState, 15, 4,13 > SplineTparam;
    ros::NodeHandle &nh_;
    GcopTrajectoryVisualizer visualizer_;
    Params params_loader_;
    gcop_comm::CtrlTraj trajectory; ///< Trajectory message for publishing the optimized trajectory
    ros::Publisher traj_publisher_;///< ros Publisher for ctrltraj:
    vector<VectorXd> obstacles;///< Obstacles

public:
    QRotorIDModelControl(ros::NodeHandle &nh);
    void setGoal(const geometry_msgs::Pose &xf_);
    void setInitialState(const geometry_msgs::Vector3 &localpos, const geometry_msgs::Vector3 &vel,
                         const geometry_msgs::Vector3 &acc, const geometry_msgs::Vector3 &rpy, 
                         const geometry_msgs::Vector3 &omega, const geometry_msgs::Quaternion &rpytcommand);
    void iterate(int N = 30);
    void getControl(Vector4d &ures);
    void getCtrlTrajectory(gcop_comm::CtrlTraj &trajectory);
 protected:
    QRotorIdGnDocp *gn;
    SplineTparam *ctp;
    QRotorIDModel sys;
    QRotorIDModelCost *cost;
    vector<double> ts;
    VectorXd tks;
    Matrix<double,13,1> p_mean;
 public:
    vector<QRotorIDState> xs;
    vector<Vector4d> us;
    QRotorIDState xf;
    double tf;
};

#endif // QROTORIDMODELCONTROL_H
