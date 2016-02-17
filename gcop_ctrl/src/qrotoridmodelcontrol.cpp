#include <gcop_ctrl/qrotoridmodelcontrol.h>

using namespace gcop;
using namespace Eigen;
using namespace std;

QRotorIDModelControl::QRotorIDModelControl(ros::NodeHandle &nh): nh_(nh)
                                                               , visualizer_(nh)
{
    //Temp Variables
    int N = 100;
    int spline_order = 2;
    int Nk = 5;
    bool debug = false;
    VectorXd meanp(13), Q(15), R(4), Qf(15), stdev_initial_state(15), obs_info(8), px0(15);
    Matrix6d stdev_offsets;
    Matrix7d stdev_gains;
    string param_file_string;

    nh.getParam("/qrotoridmodeltest/control_params",param_file_string);
    ROS_INFO("Param File: %s",param_file_string.c_str());

    params_loader_.Load(param_file_string.c_str());

    //Load Params from File
    params_loader_.GetInt("N",N);
    params_loader_.GetDouble("tf",tf);
    params_loader_.GetInt("Nk",Nk);
    params_loader_.GetInt("spline_order",spline_order);
    params_loader_.GetBool("debug",debug);
    params_loader_.GetVectorXd("mean_p",meanp);
    params_loader_.GetVectorXd("Q",Q);
    params_loader_.GetVectorXd("Qf",Qf);
    params_loader_.GetVectorXd("R",R);
    params_loader_.GetVectorXd("x0",px0);
    params_loader_.GetVectorXd("stdev_initial_state",stdev_initial_state);
    params_loader_.GetMatrix6d("stdev_offsets",stdev_offsets);
    params_loader_.GetMatrix7d("stdev_gains",stdev_gains);

    //Fill xs
    QRotorIDState x0;
    x0.p = px0.segment<3>(3);
    x0.v = px0.segment<3>(9);
    x0.w = px0.segment<3>(6);
    {
      SO3 &so3 = SO3::Instance();
      const Vector3d &rpy = px0.head<3>();
      so3.q2g(x0.R, px0);
    }
    x0.u<<0,0,0;
    xs.resize(N+1,x0);

    //Fill us
    Vector4d u0;
    u0<<(9.81/meanp[0]),0,0,0;
    us.resize(N,u0);

    //Fill ts
    double h = tf/N;
    ts.resize(N+1);
    for (int k = 0; k <= N; ++k)
      ts[k] = k*h;
    tks.resize(Nk+1);
    for(int k = 0; k <=Nk; ++k)
        tks[k] = k*(tf/Nk);

    //Clear xf:
    xf.Clear();

    //Fill Cost:
    cost = new QRotorIDModelCost(sys,tf,xf);
    cost->Q.setZero();
    cost->R.diagonal() = R;
    cost->Qf.diagonal() = Qf;
    cost->UpdateGains();
    ctp = new SplineTparam(sys,tks,spline_order);

    //Mean Params
    p_mean = meanp;

    //Create Optimizer
    gn = new QRotorIdGnDocp(sys,*cost,*ctp, ts,xs,us, &p_mean);
    gn->debug = debug;
    gn->stdev_initial_state.diagonal() = stdev_initial_state;
    gn->stdev_params.topLeftCorner<7,7>() = stdev_gains;
    gn->stdev_params.bottomRightCorner<6,6>() = stdev_offsets;
    //gn->stdev_initial_state.diagonal()<< 0.017,0.017,0.017, 0.05,0.05,0.05, 0.017,0.017,0.017, 0.05,0.05,0.05, 0.017,0.017,0.017;
    //gn->stdev_params.diagonal()<<0.001, 0.2,0.2,0.2, 0.2,0.2,0.2, 0.05,0.05,0.05, 0.05,0.05,0.05;

    //Obstacles:
    obstacles.resize(2);
    params_loader_.GetVectorXd("obs1",obs_info);
    obstacles[0] = obs_info;
    params_loader_.GetVectorXd("obs2",obs_info);
    obstacles[1] = obs_info;

    {
      vector<Obstacle> obs(2);
      obs[0].set(obstacles[0], (int)obstacles[0][7]);
      obs[1].set(obstacles[1], (int)obstacles[1][7]);
      gn->AddObstacles(obs);
    }
    cout<<"Problem Settings: "<<endl;
    cout<<"Cost Settings: "<<endl;
    cout<<"Q: "<<Q.transpose()<<endl;
    cout<<"Qf: "<<Qf.transpose()<<endl;
    cout<<"R: "<<R.transpose()<<endl;
    cout<<"Parameter Settings: "<<endl;
    cout<<"Initial State(p,v,w,u,R): "<<x0.p.transpose()<<" "<<x0.v.transpose()<<" "<<x0.w.transpose()<<" "<<x0.u.transpose()<<endl<<x0.R<<endl;
    cout<<"Mean Params: "<<meanp.transpose()<<endl;
    cout<<"stdev_init_state: "<<endl<<stdev_initial_state.transpose()<<endl;
    cout<<"stdev_params: "<<endl<<(gn->stdev_params)<<endl;

    //Publisher for gcop trajectory:
    traj_publisher_ = nh.advertise<gcop_comm::CtrlTraj>("ctrltraj",1);
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
  //Vector3d acc_(acc.x, acc.y, acc.z+9.81);//Acc in Global Frame + gravity
  //sys.a0 = (acc_ - sys.kt*rpytcommand.w*x0.R.col(2));
}

void QRotorIDModelControl::iterate(int N)
{
    for(int i = 0; i < obstacles.size(); i++)
      visualizer_.publishObstacle(obstacles[i].data(),i, obstacles[i][7]);
    gn->ko = 0.01;
    while(gn->ko < 100)
    {
      ros::Time curr_time = ros::Time::now();
      gn->Iterate();
      gn->ko = 2*gn->ko;//Increase ko;
      cout<<"gn.J: "<<(gn->J)<<endl;
      gn->Reset();
      ROS_INFO("Time taken for 1 iter: %f",(ros::Time::now() - curr_time).toSec());
      this->getCtrlTrajectory(trajectory);
      visualizer_.publishTrajectory(trajectory);
      if(traj_publisher_.getNumSubscribers() > 0)
        traj_publisher_.publish(trajectory);
      //ROS_INFO("Press Key to continue...");//DEBUG
      //cout<<"Stdev Final: "<<(gn->xs_std.rightCols<1>()).transpose()<<endl;
      //getchar();
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
 // printf("US size: %d",N);
  trajectory.N = N;

  trajectory.statemsg.resize(N+1);
  trajectory.pos_std.resize(N+1);
  trajectory.ctrl.resize(N);

  for (int count = 0;count<N+1;count++)
  {
    eigenVectorTogeometrymsgsVector(trajectory.statemsg[count].basepose.translation, xs[count].p);
    so3TogeometrymsgsQuaternion(trajectory.statemsg[count].basepose.rotation, xs[count].R);
    eigenVectorTogeometrymsgsVector(trajectory.statemsg[count].basetwist.linear,xs[count].v);
    eigenVectorTogeometrymsgsVector(trajectory.statemsg[count].basetwist.angular,xs[count].w);
    Vector3d x_std = gn->xs_std.col(count);
    eigenVectorTogeometrymsgsVector(trajectory.pos_std[count],4*x_std);//Diameter instead of radius
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
