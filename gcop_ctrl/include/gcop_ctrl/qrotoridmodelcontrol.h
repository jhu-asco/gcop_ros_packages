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
#include <fstream>
#include <boost/thread/mutex.hpp>

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
    SO3 &so3;
    bool logged_trajectory_;///< Only Log the GCOP Trajectory once for NOW
    double max_ko, gain_ko;
    double step_size_;///< h
    double J_;///< Cost from optimization
    double tf_;
    QRotorIDState xf;
    vector<QRotorIDState> xs;
    boost::mutex params_mutex;
    struct OptimizationState {
      vector<VectorXd> obstacles;
      QRotorIDState x0;
      VectorXd s;
    } state_right_, state_left_;
protected:
    inline void so3ToGeometryMsgsQuaternion(geometry_msgs::Quaternion &out, const Matrix3d &in);
    inline void eigenVectorToGeometryMsgsVector(geometry_msgs::Vector3 &out, const Vector3d &in);
    inline void addObstacles(const vector<VectorXd> &obstacles);

public:
    QRotorIDModelControl(ros::NodeHandle &nh, string frame_id="optitrak");
    //void setGoal(const geometry_msgs::Pose &xf_);
    void setInitialState(const geometry_msgs::Vector3 &vel, const geometry_msgs::Vector3 &rpy);
    void setObstacleCenter(const double &x, const double &y, const double &z);
    void iterate(bool fast_iterate = false);
    void getControl(Vector4d &ures);
    void getCtrlTrajectory(gcop_comm::CtrlTraj &trajectory, Matrix3d &yawM, Vector3d &pos_);
    void publishTrajectory(geometry_msgs::Vector3 &pos, geometry_msgs::Vector3 &rpy);
    void setParametersAndStdev(Vector7d &gains, Matrix7d &stdev_gains, Vector6d *mean_offset = 0, Matrix6d *stdev_offsets = 0);
    void logTrajectory(std::string filename);
    void resetControls();
    double getDesiredObjectDistance(double delay_send_time);
    const QRotorIDState &getInitialState();
    void saveSolutions();
  protected:
    QRotorIdGnDocp *gn;
    SplineTparam *ctp;
    QRotorIDModel sys;
    QRotorIDModelCost *cost;
    vector<double> ts;
    VectorXd tks;
    Matrix<double,13,1> p_mean;
    Matrix<double,13,1> p_mean_copy_;
    Matrix6d stdev_offsets_;
    Matrix7d stdev_gains_;
    int skip_publish_segments;///< Skip these segments for publishing gcop trajectory
    vector<Matrix3d> eigen_vectors_stdev;///<Stdev eigen vectors
    vector<Vector3d> eigen_values_stdev;///< Stdev of eigen values
 public:
    vector<Vector4d> us;
 public:
    //Getter Functions:
    const Matrix<double,13,1> getParameters()
    {
      Matrix<double,13,1> p_mean_local;
      params_mutex.lock();
      p_mean_local = p_mean_copy_;
      params_mutex.unlock();
      return p_mean_local;
    }

    const double &stepSize()
    {
        return step_size_;
    }

    const double &J()
    {
        return J_;
    }

    const double &tf()
    {
        return tf_;
    }
    const QRotorIDState &x0()
    {
        return xs[0];
    }
};

#endif // QROTORIDMODELCONTROL_H
