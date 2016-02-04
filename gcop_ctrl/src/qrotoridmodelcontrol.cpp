#include <gcop_ctrl/qrotoridmodelcontrol.h>

using namespace gcop;
using namespace Eigen;
using namespace std;

QRotorIDModelControl::QRotorIDModelControl(int N, double tf_, bool debug):tf(tf_)
                                                                          , cost(sys,tf_,xf)
{
    //Fill xs
    QRotorIDState x0;
    x0.Clear();
    xs.resize(N+1,x0);
    //Fill us
    Vector4d u0;
    u0<<(9.81/sys.kt),0,0,0;
    us.resize(N,u0);
    //Fill ts
    double h = tf_/N;
    ts.resize(N+1);
    for (int k = 0; k <= N; ++k)
      ts[k] = k*h;
    //Clear xf:
    xf.Clear();
    //Fill Cost with default values:
    cost.Q.setZero();
    cost.R.diagonal()<<0.001,0.01,0.01,0.01;
    cost.Qf.diagonal()<<5,5,5, 40,40,40, .5,.5,.5, 5,5,5, 1e-3,1e-3,1e-3;
    //Create Optimizer
    ddp = new QRotorDdp(sys,cost,ts,xs,us);
    ddp->mu = 0.01;
    ddp->debug = debug;
}

static inline void eigenVectorTogeometrymsgsVector(geometry_msgs::Vector3 &out, const Vector3d &in)
{
    out.x = in[0];
    out.y = in[1];
    out.z = in[2];
}

static inline void so3TogeometrymsgsQuaternion(geometry_msgs::Quaternion &out, const Matrix3d &in)
{
  SO3 &so3 = SO3::Instance();
  Vector4d wxyz;
  so3.g2quat(wxyz,in);
  out.w = wxyz[0];
  out.x = wxyz[1];
  out.y = wxyz[2];
  out.z = wxyz[3];
}

void QRotorIDModelControl::setGoal(const geometry_msgs::Pose &xf_)
{
    Vector4d wxyz(xf_.orientation.w, xf_.orientation.x, xf_.orientation.y, xf_.orientation.z);
    SO3 &so3 = SO3::Instance();
    so3.quat2g(xf.R, wxyz);
    xf.p<<xf_.position.x, xf_.position.y, xf_.position.z;
    xf.v.setZero();
    xf.w.setZero();
    xf.u.setZero();
    //xf = xf_;
}

void QRotorIDModelControl::setInitialState(const geometry_msgs::Vector3 &localpos, const geometry_msgs::Vector3 &vel,
    const geometry_msgs::Vector3 &acc, const geometry_msgs::Vector3 &rpy, 
    const geometry_msgs::Vector3 &omega, const geometry_msgs::Quaternion &rpytcommand)
{
  Vector3d rpy_(rpy.x, rpy.y, rpy.z);
  SO3 &so3 = SO3::Instance();
  QRotorIDState &x0 = xs[0];
  x0.p<<localpos.x, localpos.y, localpos.z;
  x0.v<<vel.x, vel.y, vel.z;
  so3.q2g(x0.R, rpy_);
  x0.w<<omega.x, omega.y, omega.z;
  x0.u<<rpytcommand.x, rpytcommand.y, rpytcommand.z;
  Vector3d acc_(acc.x, acc.y, acc.z+9.81);//Acc in Global Frame + gravity
  sys.a0 = (acc_ - sys.kt*rpytcommand.w*x0.R.col(2));
}

void QRotorIDModelControl::iterate(int N)
{
    for(int k = 0; k <N; k++)
    {
        ddp->Iterate();
    }
}

void QRotorIDModelControl::getControl(Vector4d &ures)
{
    ures(0) = us[0](0);
    ures(1) = xs[1].u(0);
    ures(2) = xs[1].u(1);
    ures(3) = xs[1].u(2);
}

void QRotorIDModelControl::getCtrlTrajectory(gcop_comm::CtrlTraj &trajectory)
{
  int N = us.size();
  trajectory.N = N;

  trajectory.statemsg.resize(N+1);
  trajectory.ctrl.resize(N);

  for (int count = 0;count<N+1;count++)
  {
    eigenVectorTogeometrymsgsVector(trajectory.statemsg[count].basepose.translation, xs[count].p);
    so3TogeometrymsgsQuaternion(trajectory.statemsg[count].basepose.rotation, xs[count].R);
    eigenVectorTogeometrymsgsVector(trajectory.statemsg[count].basetwist.linear,xs[count].v);
    eigenVectorTogeometrymsgsVector(trajectory.statemsg[count].basetwist.angular,xs[count].w);
  }
  for (int count = 0;count<N;count++)
  {
    trajectory.ctrl[count].ctrlvec.resize(4);
    trajectory.ctrl[count].ctrlvec[0] = us[count][0];
    for(int count1 = 0;count1 < 3;count1++)
    {
      trajectory.ctrl[count].ctrlvec[count1+1] = xs[count+1].u(count1);
    }
  }
  trajectory.time = ts;
  //final goal:
  eigenVectorTogeometrymsgsVector(trajectory.finalgoal.basepose.translation, xf.p);
  so3TogeometrymsgsQuaternion(trajectory.finalgoal.basepose.rotation, xf.R);
  eigenVectorTogeometrymsgsVector(trajectory.finalgoal.basetwist.linear,xf.v);
  eigenVectorTogeometrymsgsVector(trajectory.finalgoal.basetwist.angular,xf.w);
  //DEBUG:
  SO3 &so3 = SO3::Instance();
  Vector3d rpy;
  so3.g2q(rpy, xs[N].R);
  cout<<" "<<xs[N].p.transpose()<<" "<<xs[N].v.transpose()<<" "<<rpy.transpose()<<" "<<xs[N].w.transpose()<<" "<<xs[N].u.transpose()<<endl;
}

