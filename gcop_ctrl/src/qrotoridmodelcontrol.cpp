#include <gcop_ctrl/qrotoridmodelcontrol.h>

using namespace gcop;
using namespace Eigen;
using namespace std;

QRotorIDModelControl::QRotorIDModelControl(ros::NodeHandle &nh, string frame_id): nh_(nh)
                                                                                  , visualizer_(nh,frame_id)
                                                                                  , so3(SO3::Instance())
                                                                                  , skip_publish_segments(2), logged_trajectory_(false)
                                                                                  , max_ko(100), gain_ko(2), J_(1000)
{
    //Temp Variables
    int N = 100;
    int spline_order = 2;
    int Nk = 5;
    bool debug = false;
    VectorXd meanp(13), Q(15), R(4), Qf(15), stdev_initial_state(15), obs_info(8), px0(15), pxf(15);
    string param_file_string;

    nh.getParam("/qrotoridmodeltest/control_params",param_file_string);
    ROS_INFO("Param File: %s",param_file_string.c_str());
    nh.getParam("/display/skip_pub_segments",skip_publish_segments);

    params_loader_.Load(param_file_string.c_str());

    //Load Params from File
    params_loader_.GetInt("N",N);
    params_loader_.GetDouble("tf",tf_);
    params_loader_.GetInt("Nk",Nk);
    params_loader_.GetInt("spline_order",spline_order);
    params_loader_.GetBool("debug",debug);
    params_loader_.GetVectorXd("mean_p",meanp);
    params_loader_.GetVectorXd("Q",Q);
    params_loader_.GetVectorXd("Qf",Qf);
    params_loader_.GetVectorXd("R",R);
    params_loader_.GetVectorXd("x0",px0);
    params_loader_.GetVectorXd("xf",pxf);
    params_loader_.GetVectorXd("stdev_initial_state",stdev_initial_state);
    params_loader_.GetMatrix6d("stdev_offsets",stdev_offsets_);
    params_loader_.GetMatrix7d("stdev_gains",stdev_gains_);

    //Fill xs
    QRotorIDState x0;
    x0.p = px0.segment<3>(3);
    x0.v = px0.segment<3>(9);
    x0.w = px0.segment<3>(6);
    {
      const Vector3d &rpy = px0.head<3>();
      so3.q2g(x0.R, rpy);
    }
    x0.u<<0,0,0;
    state_left_.x0 = state_right_.x0 = x0;
    xs.resize(N+1,x0);
    eigen_values_stdev.resize(N+1);
    eigen_vectors_stdev.resize(N+1);

    //XF:
    xf.p = pxf.segment<3>(3);
    xf.v = pxf.segment<3>(9);
    xf.w = pxf.segment<3>(6);
    {
      const Vector3d &rpy = pxf.head<3>();
      so3.q2g(xf.R, rpy);
    }
    xf.u<<0,0,0;


    //Fill us
    Vector4d u0;
    u0<<(9.81/meanp[0]),0,0,0;
    us.resize(N,u0);

    //Fill ts
    double h = tf_/N;
    step_size_ = h;
    ts.resize(N+1);
    for (int k = 0; k <= N; ++k)
      ts[k] = k*h;
    tks.resize(Nk+1);
    for(int k = 0; k <=Nk; ++k)
        tks[k] = k*(tf_/Nk);

    //Fill Cost:
    cost = new QRotorIDModelCost(sys,tf_,xf);
    cost->Q.setZero();
    cost->R.diagonal() = R;
    cost->Qf.diagonal() = Qf;
    cost->UpdateGains();
    ctp = new SplineTparam(sys,tks,spline_order);

    //Mean Params
    p_mean_copy_ = p_mean = meanp;

    //Create Optimizer
    gn = new QRotorIdGnDocp(sys,*cost,*ctp, ts,xs,us, &p_mean);
    gn->debug = debug;
    gn->stdev_initial_state.diagonal() = stdev_initial_state;
    gn->stdev_params.topLeftCorner<7,7>() = stdev_gains_;
    gn->stdev_params.bottomRightCorner<6,6>() = stdev_offsets_;
    //gn->stdev_initial_state.diagonal()<< 0.017,0.017,0.017, 0.05,0.05,0.05, 0.017,0.017,0.017, 0.05,0.05,0.05, 0.017,0.017,0.017;
    //gn->stdev_params.diagonal()<<0.001, 0.2,0.2,0.2, 0.2,0.2,0.2, 0.05,0.05,0.05, 0.05,0.05,0.05;

    //Rate of avoidance:
    params_loader_.GetDouble("max_ko",max_ko);
    params_loader_.GetInt("max_iters",gn->max_iters);
    params_loader_.GetDouble("gain_ko",gain_ko);

    //Fill tparam (knots) from optimizer:
    state_left_.s = state_right_.s = gn->s;

    //Obstacles:
    //Load left obstacles
    params_loader_.GetVectorXd("obs_left",obs_info);
    obstacles.push_back(obs_info);
    state_left_.obstacles = obstacles;
    //Load right obstacles
    params_loader_.GetVectorXd("obs_right",obs_info);
    obstacles[0] = obs_info;
    state_right_.obstacles = obstacles;
    /*if(params_loader_.Exists("obs2"))
    {
      params_loader_.GetVectorXd("obs2",obs_info);
      obstacles.push_back(obs_info);
    }
    */
    cout<<"Problem Settings: "<<endl;
    cout<<"X0: "<<px0.transpose()<<endl;
    cout<<"Xf: "<<pxf.transpose()<<endl;
    cout<<"Cost Settings: "<<endl;
    cout<<"Q: "<<Q.transpose()<<endl;
    cout<<"Qf: "<<Qf.transpose()<<endl;
    cout<<"R: "<<R.transpose()<<endl;
    cout<<"Parameter Settings: "<<endl;
    cout<<"Initial State(p,v,w,u,R): "<<x0.p.transpose()<<" "<<x0.v.transpose()<<" "<<x0.w.transpose()<<" "<<x0.u.transpose()<<endl<<x0.R<<endl;
    cout<<"Mean Params: "<<meanp.transpose()<<endl;
    cout<<"stdev_init_state: "<<endl<<stdev_initial_state.transpose()<<endl;
    cout<<"stdev_params: "<<endl<<(gn->stdev_params)<<endl;
    cout<<"Number Of Obstacles: "<<obstacles.size()<<endl;
    cout<<"Max Iters: "<<(gn->max_iters)<<endl;

    //Publisher for gcop trajectory:
    traj_publisher_ = nh.advertise<gcop_comm::CtrlTraj>("ctrltraj",2,true);//Latched
}

void QRotorIDModelControl::eigenVectorToGeometryMsgsVector(geometry_msgs::Vector3 &out, const Vector3d &in)
{
    out.x = in[0];
    out.y = in[1];
    out.z = in[2];
}

void QRotorIDModelControl::so3ToGeometryMsgsQuaternion(geometry_msgs::Quaternion &out, const Matrix3d &in)
{
  Vector4d wxyz;
  so3.g2quat(wxyz,in);
  out.w = wxyz[0];
  out.x = wxyz[1];
  out.y = wxyz[2];
  out.z = wxyz[3];
}

/*void QRotorIDModelControl::setGoal(const geometry_msgs::Pose &xf_)
{
    Vector4d wxyz(xf_.orientation.w, xf_.orientation.x, xf_.orientation.y, xf_.orientation.z);
    so3.quat2g(xf.R, wxyz);
    xf.p<<xf_.position.x, xf_.position.y, xf_.position.z;
    xf.v.setZero();
    xf.w.setZero();
    xf.u.setZero();
    //xf = xf_;
}
*/

void QRotorIDModelControl::setInitialState(const geometry_msgs::Vector3 &vel, const geometry_msgs::Vector3 &rpy)
{
  Vector3d rpy_(0, 0, -rpy.z);
  Matrix3d R;
  so3.q2g(R,rpy_);
  //QRotorIDState &x0 = xs[0];
  xs[0].v<<vel.x, vel.y, vel.z;
  xs[0].v = R*xs[0].v;//Rotate the velocity to body frame
  //Set Initial rp based on measurements
  Vector3d rp_(rpy.x, rpy.y, 0);
  so3.q2g(xs[0].R,rp_);
  xs[0].u = rp_;// set commanded rp equal to initial state
  cout<<"Initial Vel: "<<xs[0].v.transpose()<<endl;
  cout<<"Initial rp: "<<rp_.transpose()<<endl;
}

void QRotorIDModelControl::addObstacles(const vector<VectorXd> &obstacles)
{
  vector<Obstacle> obs(state_left_.obstacles.size());
  for(int i = 0; i < obstacles.size(); i++)
    obs[i].set(obstacles[i], (int)obstacles[i][7]);
  gn->AddObstacles(obs);
}

void QRotorIDModelControl::saveSolutions()
{
  addObstacles(state_left_.obstacles);
  //Compute solutions
  ROS_INFO("Computing Left Solution");
  xs[0] = state_left_.x0;
  gn->s = state_left_.s;
  //Iterate
  iterate(false);
  //Save
  state_left_.s = gn->s;
  //Repeat for right
  ROS_INFO("Computing Right Solution");
  addObstacles(state_right_.obstacles);
  xs[0] = state_right_.x0;
  gn->s = state_right_.s;
  iterate(false);
  state_right_.s = gn->s;
}

//Remove Index
// Restore solution based on center y
// Update iterate for non fast iteration
void QRotorIDModelControl::setObstacleCenter(const double &x, const double &y, const double &z)
{
  Eigen::Vector3d center(x,y,z);
  gn->SetObstaclePos(0, center);
  //Update the obstacles also for plotting:
  obstacles[0].segment<3>(1) = center;
  if(y < 0)
  {
    //Choose obstacle right solution
    ROS_INFO("Choosing right side obstacle solution: %f", y);
    gn->s = state_right_.s;
    xs[0] = state_right_.x0;
  }
  else
  {
    ROS_INFO("Choosing left side obstacle solution: %f", y);
    gn->s = state_left_.s;
    xs[0] = state_left_.x0;
  }
}

void QRotorIDModelControl::iterate(bool fast_iterate)
{
    //Copy Params:
    params_mutex.lock();

    gn->stdev_params.topLeftCorner<7,7>() = stdev_gains_;
    gn->stdev_params.bottomRightCorner<6,6>() = stdev_offsets_;
    p_mean = p_mean_copy_;

    params_mutex.unlock();
    if(fast_iterate)
      gn->ko = (max_ko/2);
    else
    {
      gn->ko = 0.01;
    }

    double temp_max_iters = gn->max_iters;
    if(fast_iterate)
      gn->max_iters = 10;
    while(gn->ko < max_ko)
    {
      ros::Time curr_time = ros::Time::now();
      gn->Iterate();
      gn->ko = gain_ko*gn->ko;//Increase ko;
      cout<<"gn.J: "<<(gn->J)<<endl;
      gn->Reset();
      ROS_INFO("Time taken for 1 iter: %f, %f",(ros::Time::now() - curr_time).toSec(), gn->ko);
      cout<<"Final Pos: "<<xs.back().p.transpose()<<endl;
      //ROS_INFO("Press Key to continue...");//DEBUG
      //cout<<"Stdev: "<<(gn->xs_std.transpose())<<endl;
      //cout<<"Stdev Final: "<<(gn->xs_std.rightCols<1>()).transpose()<<endl;
      //getchar();
    }

    SelfAdjointEigenSolver<Matrix3d> es;
    for(int i = 0; i < xs.size(); i++)
    {
      es.computeDirect(gn->xs_invcov[i]);
      eigen_vectors_stdev[i] = es.eigenvectors();
      eigen_values_stdev[i] = es.eigenvalues().cwiseInverse().cwiseSqrt();

      if(eigen_vectors_stdev[i].col(0).transpose()*(eigen_vectors_stdev[i].col(1).cross(eigen_vectors_stdev[i].col(2)))<0)
      {
        eigen_vectors_stdev[i].col(0).swap(eigen_vectors_stdev[i].col(1));//Swap the values
        double temp = eigen_values_stdev[i](1);
        eigen_values_stdev[i](1) = eigen_values_stdev[i](0);
        eigen_values_stdev[i](0) = temp;
      }
    }
    J_ = gn->J;
    if(fast_iterate)
      gn->max_iters = temp_max_iters;//Reset the max iterations of gn
}
void QRotorIDModelControl::logTrajectory(std::string filename)
{
  {
    ofstream trajfile;///< Log trajectory
    trajfile.open(filename.c_str());

    trajfile.precision(10);

    //trajfile<<"#Time X Y Z Vx Vy Vz r p y Wx Wy Wz Ut Ur Up Uy xs_stdx xs_stdy xs_stdz"<<endl;

    logged_trajectory_ = true;
    int N = us.size();
    //Log Trajectory:
    Vector3d rpy;
    so3.g2q(rpy,xs[0].R);
    //Get xs stdev and orientation matrix:
    Vector3d rpy_std;
    so3.g2q(rpy_std,eigen_vectors_stdev[0]);
    trajfile<<ts[0]<<" "<<xs[0].p.transpose()<<" "<<xs[0].v.transpose()<<" "<<rpy.transpose()<<" "<<xs[0].w.transpose()<<" 0 "<<xs[0].u.transpose()<<" "<<(eigen_values_stdev[0].transpose())<<" "<<(rpy_std.transpose())<<endl;
    for(int i = 1; i < N+1;i++)
    {
      so3.g2q(rpy,xs[i].R);
      so3.g2q(rpy_std,eigen_vectors_stdev[i]);
      trajfile<<ts[i]<<" "<<xs[i].p.transpose()<<" "<<xs[i].v.transpose()<<" "<<rpy.transpose()<<" "<<xs[i].w.transpose()<<" "<<us[i-1][0]<<" "<<xs[i].u.transpose()<<" "<<(eigen_values_stdev[i].transpose())<<" "<<(rpy_std.transpose())<<endl;
    }
  }
}

void QRotorIDModelControl::resetControls()
{
    for(int i = 0; i < us.size(); i++)
    {
        us[i]<<(9.81/p_mean[0]),0,0,0;//Set to default values
    }
}

double QRotorIDModelControl::getDesiredObjectDistance(double delay_send_time = 0.2)
{
  //cout<<"Xs[0].v.norm: "<<xs[0].v.norm()<<endl;
  Eigen::Vector3d obs_local_pos = state_left_.obstacles[0].segment<3>(1);
  obs_local_pos[1] = 0; // Set y coordinate to 0
  return obs_local_pos.norm() + xs[0].v.norm()*delay_send_time;
}

const QRotorIDState & QRotorIDModelControl::getInitialState()
{
  return state_left_.x0;
}
void QRotorIDModelControl::getControl(Vector4d &ures)
{
    ures(0) = us[0](0);
    ures(1) = xs[1].u(0);
    ures(2) = xs[1].u(1);
    ures(3) = xs[1].u(2);
}
void QRotorIDModelControl::publishTrajectory(geometry_msgs::Vector3 &pos, geometry_msgs::Vector3 &rpy)
{
    Vector3d yaw(0,0,rpy.z);
    Vector3d pos_(pos.x, pos.y, pos.z);
    Matrix3d yawM;
    so3.q2g(yawM,yaw);
    for(int i = 0; i < obstacles.size(); i++)
    {
      VectorXd obs_data = obstacles[i];
      obs_data.segment<3>(1) = pos_ + yawM*obs_data.segment<3>(1);//Rotate
      obs_data.segment<3>(4) = yawM*obs_data.segment<3>(4);
      visualizer_.publishObstacle(obs_data.data(),i, obs_data[7]);
    }
    this->getCtrlTrajectory(trajectory, yawM, pos_);
    visualizer_.publishTrajectory(trajectory);
    //if(traj_publisher_.getNumSubscribers() > 0)
    traj_publisher_.publish(trajectory);
}

void QRotorIDModelControl::setParametersAndStdev(Vector7d &gains, Matrix7d &stdev_gains, Vector6d *mean_offsets, Matrix6d *stdev_offsets)
{
    params_mutex.lock();
    p_mean_copy_.head<7>() = gains;
    stdev_gains_ = stdev_gains;
    if(mean_offsets)
        p_mean_copy_.tail<6>() = *mean_offsets;
    if(stdev_offsets)
      stdev_offsets_ = *stdev_offsets;
    params_mutex.unlock();
}

void QRotorIDModelControl::getCtrlTrajectory(gcop_comm::CtrlTraj &trajectory, Matrix3d &yawM, Vector3d &pos_)
{
  int N = us.size();
 // printf("US size: %d",N);
  int number_states = std::floor(double(N+1)/skip_publish_segments)+1;
  trajectory.N = number_states-1;

  trajectory.statemsg.resize(number_states);
  trajectory.pos_std.resize(number_states);
  trajectory.ctrl.resize(N);
  trajectory.time.resize(number_states);
  int ind = 0;

  for (int count = 0;count<N+1;count+=skip_publish_segments)
  {
    eigenVectorToGeometryMsgsVector(trajectory.statemsg.at(ind).basepose.translation, yawM*xs[count].p + pos_);
    so3ToGeometryMsgsQuaternion(trajectory.statemsg.at(ind).basepose.rotation, yawM*xs[count].R);
    so3ToGeometryMsgsQuaternion(trajectory.pos_std.at(ind).rot_std, yawM*eigen_vectors_stdev[count]);
    eigenVectorToGeometryMsgsVector(trajectory.pos_std.at(ind).scale_std,4*eigen_values_stdev[count]);
    trajectory.time[ind] = ts[count];
    //eigenVectorToGeometryMsgsVector(trajectory.statemsg[count].basetwist.linear,xs[count].v);
    //eigenVectorToGeometryMsgsVector(trajectory.statemsg[count].basetwist.angular,xs[count].w);
    //double x_std_max =gn->xs_std.col(count).maxCoeff();
    //trajectory.pos_std.at(ind).x = trajectory.pos_std.at(ind).y =trajectory.pos_std.at(ind).z =4*x_std_max;
    ind++;
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
  //final goal:
  eigenVectorToGeometryMsgsVector(trajectory.finalgoal.basepose.translation, yawM*xf.p+pos_);
  so3ToGeometryMsgsQuaternion(trajectory.finalgoal.basepose.rotation, xf.R);
  eigenVectorToGeometryMsgsVector(trajectory.finalgoal.basetwist.linear,xf.v);
  eigenVectorToGeometryMsgsVector(trajectory.finalgoal.basetwist.angular,xf.w);
  //DEBUG:
  Vector3d rpy;
  so3.g2q(rpy, xs[N].R);
  cout<<" "<<xs[N].p.transpose()<<" "<<xs[N].v.transpose()<<" "<<rpy.transpose()<<" "<<xs[N].w.transpose()<<" "<<xs[N].u.transpose()<<endl;
}
