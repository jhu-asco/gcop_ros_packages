#ifndef QROTORIDMODELCONTROL_H
#define QROTORIDMODELCONTROL_H
#include <gcop/qrotoridmodelcost.h>
#include <gcop/ddp.h>
#include <gcop/utils.h>
#include <gcop_comm/CtrlTraj.h>//msg for publishing ctrl trajectory
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Pose.h>

using namespace gcop;
using namespace Eigen;
using namespace std;

class QRotorIDModelControl
{
  protected:
  typedef Ddp<QRotorIDState, 15, 4, 10> QRotorDdp;
public:
    QRotorIDModelControl(int N = 100, double tf_ = 2.0, bool debug = false);
    void setGoal(const geometry_msgs::Pose &xf_);
    void setInitialState(const geometry_msgs::Vector3 &localpos, const geometry_msgs::Vector3 &vel,
                         const geometry_msgs::Vector3 &acc, const geometry_msgs::Vector3 &rpy, 
                         const geometry_msgs::Vector3 &omega, const geometry_msgs::Quaternion &rpytcommand);
    void iterate(int N = 30);
    void getControl(Vector4d &ures);
    void getCtrlTrajectory(gcop_comm::CtrlTraj &trajectory);
 protected:
    QRotorDdp *ddp;
    QRotorIDModel sys;
    QRotorIDModelCost cost;
    vector<double> ts;
    double tf;
 public:
    vector<QRotorIDState> xs;
    vector<Vector4d> us;
    QRotorIDState xf;
};

#endif // QROTORIDMODELCONTROL_H
