/** This is an example on how to control a rccar using GCOP Library. This creates a 
 * rccar system, rnlq cost and an optimization method (DDP) using  GCOP.
 * There is a dynamic reconfigure interface to change initial and final conditions
 * and iterate the optimization algorithm.
 * An interface to optimize the continuously based on external feedback is also added.
 * The optimized trajectories are published on a topic("CtrlTraj") which has all the controls and states
 *
 * Author: Gowtham Garimella
 */
#include "ros/ros.h"
#include <iomanip>
#include <iostream>
#include <dynamic_reconfigure/server.h>
#include "gcop/ddp.h" //gcop ddp header
#include "gcop/rnlqcost.h" //gcop lqr header
#include "gcop/rccar.h"
#include <gcop_comm/State.h>
#include "gcop_comm/CtrlTraj.h"//msg for publishing ctrl trajectory
#include "gcop_ctrl/DMocInterfaceConfig.h"
#include "geometry_msgs/TransformStamped.h"
#include <tf/transform_listener.h>
#include "gcop/utils.h"////
#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Quaternion.h"
#include "tf/transform_datatypes.h"
#include <gcop_comm/Trajectory_req.h>
//#include "LinearMath/btMatrix3x3.h"


using namespace std;
using namespace Eigen;
using namespace gcop;

typedef Ddp<Vector4d, 4, 2> RccarDdp;

void initCtrls(void);

gcop_ctrl::DMocInterfaceConfig g_config;
//ros messages
gcop_comm::CtrlTraj trajectory; ///< Trajectory message for publishing the optimized trajectory

//Publisher
ros::Publisher trajpub;///< Publisher for optimal trajectory

//Timer
ros::Timer iteratetimer;///< Timer to iterate continously the optimization problem

//Subscriber
ros::Subscriber initialposn_sub;///< Subscriber to receive external feedback on initial position


//Initialize the rccar system
Rccar sys;///< GCOP Rccar system for performing dynamics

//Optimal Controller
RccarDdp *ddp;///< Optimization algorithm from GCOP

//Cost class
RnLqCost<4, 2>*cost;///< Cost function for optimization 

//Define states and controls for system
vector<double> ts;///< time knots for trajectory
vector<Vector4d> xs;///< Discrete states at the discrete times in trajectory
vector<Vector2d> us;///< Discrete controls at the discrete times in trajectory
Vector4d xf = Vector4d::Zero();///< final state
int Nit = 30;///< Number of iterations for ddp
bool usemocap = false;///< If true uses tf for input position. (Passed by parameter)

double tprev = 0;///< Local variable

tf::Vector3 yprev;///< Local variable

/** Publishes the current trajectory
 */
void pubtraj() //N is the number of segments
{
  int N = us.size();

  for (int count = 0;count<N+1;count++)
  {
    for(int count1 = 0;count1 < 4;count1++)
    {
      trajectory.statemsg[count].statevector.resize(4);

      trajectory.statemsg[count].statevector[count1] = xs[count](count1);
    }
  }
  for (int count = 0;count<N;count++)
  {
    for(int count1 = 0;count1 < 2;count1++)
    {
      trajectory.ctrl[count].ctrlvec.resize(2);

      trajectory.ctrl[count].ctrlvec[count1] = us[count](count1);
    }
  }
  trajectory.time = ts;
  //final goal:
  for(int count = 0;count<4;count++)
  {
    trajectory.finalgoal.statevector[count] = xf(count);
  }
  trajpub.publish(trajectory);
}

/** Iterates through the optimization algorithm. Is called by a ros timer
 */
void iterateCallback(void)
{
  struct timeval timer;
  timer_start(timer);
  for (int count = 1;count <= Nit;count++){

    ddp->Iterate();//Updates us and xs after one iteration
  }
  long te = timer_us(timer);
  pubtraj();
}

/** Reconfiguration interface for configuring the optimization problem
 */
void paramreqcallback(gcop_ctrl::DMocInterfaceConfig &config, uint32_t level) 
{
  g_config = config;
  static bool first_time=true;
  if(first_time)
  {
    first_time = false;
    //Set max number of iterations
    Nit = config.Nit;

    //set xf
    xf(0) = config.xN;
    xf(1) = config.yN;
    xf(2) = config.thetaN;
    xf(3) = config.vN;

    //Set x0
    Vector4d x0;
    x0(0) = config.x0;
    x0(1) = config.y0;
    x0(2) = config.theta0;
    x0(3) = config.v0;
    // initial state
    xs[0] = x0;

    //State state cost
    VectorXd Q(4);//Costs
    Q(0) = config.Q1;
    Q(1) = config.Q2;
    Q(2) = config.Q3;
    Q(3) = config.Q4;

    //Set final cost
    VectorXd Qf(4);
    Qf(0) = config.Qf1;
    Qf(1) = config.Qf2;
    Qf(2) = config.Qf3;
    Qf(3) = config.Qf4;

    //set control cost
    VectorXd R(2);
    R(0) = config.R1;
    R(1) = config.R2;

    //Set final time
    cost->tf = config.tf;
    int N = us.size();
    double h = config.tf/N;
    for (int k = 0; k <=N; ++k)
      ts[k] = k*h;

    cost->Q = Q.asDiagonal();
    cost->R = R.asDiagonal();
    cost->Qf = Qf.asDiagonal();

    cost->UpdateGains();

    //update controls based on new tf and xf
    initCtrls();

    return;
  }

  if(config.iterate)
  {
    iterateCallback();
    config.iterate = false;
  }
  else
  {
    ddp->debug = config.ddp_debug;

    //Set max number of iterations
    Nit = config.Nit;

    //set xf
    xf(0) = config.xN;
    xf(1) = config.yN;
    xf(2) = config.thetaN;
    xf(3) = config.vN;

    //Set x0
    Vector4d x0;
    x0(0) = config.x0;
    x0(1) = config.y0;
    x0(2) = config.theta0;
    x0(3) = config.v0;
    // initial state
    xs[0] = x0;

    //State state cost
    VectorXd Q(4);//Costs
    Q(0) = config.Q1;
    Q(1) = config.Q2;
    Q(2) = config.Q3;
    Q(3) = config.Q4;

    //Set final cost
    VectorXd Qf(4);
    Qf(0) = config.Qf1;
    Qf(1) = config.Qf2;
    Qf(2) = config.Qf3;
    Qf(3) = config.Qf4;

    //set control cost
    VectorXd R(2);
    R(0) = config.R1;
    R(1) = config.R2;

    //Set final time
    cost->tf = config.tf;
    int N = us.size();
    double h = config.tf/N;
    for (int k = 0; k <=N; ++k)
      ts[k] = k*h;

    cost->Q = Q.asDiagonal();
    cost->R = R.asDiagonal();
    cost->Qf = Qf.asDiagonal();

    cost->UpdateGains();

    //update controls based on new tf and xf
    initCtrls();

    //change parameters in ddp:
    ddp->mu = config.mu;
    pubtraj();
  }
}

bool cbSrvDdp(gcop_comm::Trajectory_req::Request &req, gcop_comm::Trajectory_req::Response &resp)
{
//  //Update the ddp problem for the new request
//  //Set ts
  int N = us.size();
  cost->tf = (req.itreq.tf - req.itreq.t0);
  double h = cost->tf/N;   // time step
  for (int k = 0; k <=N; ++k)
    ts[k] = k*h;
//
  //set x0
  Vector4d x0;
  x0(0) = req.itreq.x0.statevector[0];
  x0(1) = req.itreq.x0.statevector[1];
  x0(2) = req.itreq.x0.statevector[2];
  x0(3) = req.itreq.x0.statevector[3];
  xs[0] = x0;

  //set xf
  xf(0) = req.itreq.xf.statevector[0];
  xf(1) = req.itreq.xf.statevector[1];
  xf(2) = req.itreq.xf.statevector[2];
  xf(3) = req.itreq.xf.statevector[3];


  if(g_config.enable_debug)
  {
    std::cout<<"*The following ddp planning request is received"<<std::endl;
    std::cout<<"  ddp_debug:"<<ddp->debug<<std::endl;
    std::cout<<"  t0:"<<req.itreq.t0<<" tf:"<<req.itreq.tf<<std::endl;
    std::cout<<"  x0:"<<"\t"<<x0.transpose()<<std::endl;
    std::cout<<"  xf:"<<"\t"<<xf.transpose()<<std::endl;
  }

  initCtrls();

  ddp->Update(true);

  if(g_config.enable_debug)
  {
    std::cout<<"  effect of first controls"<<std::endl;
    for (int k = 0; k <=N; ++k)
      std::cout<<"  "<<xs[k].transpose()<<std::endl;
  }

  //Iterate
  for (int count = 1;count <= Nit;count++)
  {
    if(g_config.enable_debug)
      std::cout<<"  Entering Iteration Number:"<<count<<std::endl;
    if(g_config.getchar)
      getchar();
    ddp->Iterate();//Updates us and xs after one iteration
  }

  //send back the response
  std::cout<<"Creating the traj message"<<std::endl;

  for (int count = 0;count<N+1;count++)
    {
      for(int count1 = 0;count1 < 4;count1++)
      {
        trajectory.statemsg[count].statevector.resize(4);
        trajectory.statemsg[count].statevector[count1] = xs[count](count1);
      }
    }
    for (int count = 0;count<N;count++)
    {
      for(int count1 = 0;count1 < 2;count1++)
      {
        trajectory.ctrl[count].ctrlvec.resize(2);
        trajectory.ctrl[count].ctrlvec[count1] = us[count](count1);
      }
    }
    trajectory.time = ts;
    //final goal:
    for(int count = 0;count<4;count++)
      trajectory.finalgoal.statevector[count] = xf(count);

    resp.traj = trajectory;
    std::cout<<"sending back response"<<std::endl;
}


void initCtrls(void)
{
  if(g_config.enable_debug)
    std::cout<<"*initializing controls"<<endl;
  double tf = cost->tf;
  double thf = xf(2);
  double th0 = xs[0](2);
  double dx = xf(0) - xs[0](0);
  double dy = xf(1) - xs[0](1);
  double dth = atan2(sin(thf-th0), cos(thf-th0));
  double dist = sqrt(dx*dx + dy*dy);
  double u = xs[0](3);
  double v = xf(3);
  double acc = (4*dist - 2*(u+v)*tf)/(tf*tf);
  double torq = acc/sys.r;
  int N = us.size();
  double tanangl = dth*sys.l/(u*tf + acc*tf*tf/4);
  for (int i = 0; i < N/2; ++i) {
    us[i] = Vector2d(torq, tanangl);
    us[N/2+i] = Vector2d(-torq, tanangl);
  }
  if(g_config.enable_debug)
  {
    std::cout<<"  tf:"<<tf<<endl;
    std::cout<<"  dist:"<<dist<<std::endl;
    std::cout<<"  acc:"<<acc<<std::endl;
    std::cout<<"  u:"<<u<<std::endl;
    std::cout<<"  v:"<<v<<std::endl;
    std::cout<<"  th0:"<<th0<<std::endl;
    std::cout<<"  thf:"<<thf<<std::endl;
    std::cout<<"  dth:"<<dth<<std::endl;
    std::cout<<"  tanangl:"<<tanangl<<std::endl;
  }

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "rccarctrl");
  ros::NodeHandle rosddp("/ddp");
  ros::NodeHandle nh;

  //Initialize publisher
  trajpub = rosddp.advertise<gcop_comm::CtrlTraj>("ctrltraj",1);

  //Setup service:
  ros::ServiceServer srvsvr_traj_ = nh.advertiseService("req_ddp_lcl",cbSrvDdp);


  //Setting up the ddp problem
  sys.l = 0.5;
  sys.r = 1;//acceleration of vehicle/unit torque (wheel rad 7cm)
  //define parameters for the system
  int N = 64;        // number of segments

  //resize the states and controls
  ts.resize(N+1);
  xs.resize(N+1);
  us.resize(N);

  std::cout<<"initializing cost"<<std::endl;
  cost = new RnLqCost<4, 2>(sys,0.0,xf);

  //Setting with dynamic reconfigure initialized the control to the right value
  std::cout<<"setting up dynamic reconfigure"<<std::endl;
  dynamic_reconfigure::Server<gcop_ctrl::DMocInterfaceConfig> server;
  dynamic_reconfigure::Server<gcop_ctrl::DMocInterfaceConfig>::CallbackType f;
  f = boost::bind(&paramreqcallback, _1, _2);
  server.setCallback(f);

  ddp = new RccarDdp(sys, *cost, ts, xs, us);
  ddp->mu =  g_config.mu;
  ddp->debug = false;

  //Trajectory message initialization
  trajectory.N = N;
  trajectory.statemsg.resize(N+1);
  trajectory.ctrl.resize(N);
  trajectory.time = ts;
  trajectory.finalgoal.statevector.resize(4);

  pubtraj();
  ros::spin();
  return 0;
}
