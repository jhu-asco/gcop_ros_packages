#include "ros/ros.h"
#include <iomanip>
#include <iostream>
#include <dynamic_reconfigure/server.h>
#include "gcop/ddp.h" //gcop ddp header
#include "gcop/rnlqcost.h" //gcop lqr header
#include "gcop/rccar.h"
#include "gcop_comm/CtrlTraj.h"//msg for publishing ctrl trajectory
#include "gcop_comm/CurrPose.h"
#include "gcop_ctrl/DMocInterfaceConfig.h"
#include "geometry_msgs/TransformStamped.h"
#include <tf/transform_listener.h>
#include "gcop/utils.h"////
#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Quaternion.h"
#include "tf/transform_datatypes.h"
#include <math.h>
//#include "LinearMath/btMatrix3x3.h"


using namespace std;
using namespace Eigen;
using namespace gcop;

typedef Ddp<Vector4d, 4, 2> RccarDdp;

//ros messages
gcop_comm::CtrlTraj trajectory;

//Publisher
ros::Publisher trajpub;

//Timer
ros::Timer iteratetimer;

//Subscriber
ros::Subscriber initialposn_sub;

ros::Subscriber posn_sub;

//Initialize the rccar system
Rccar sys;

//Optimal Controller
RccarDdp *ddp;

//Cost class
RnLqCost<4, 2>*cost;

//Define states and controls for system
	vector<double> ts;
  vector<Vector4d> xs;
  vector<Vector2d> us;
  Vector4d xf = Vector4d::Zero();// final state initialization passed by reference so needs to be global to be changed
	int Nit = 30;//number of iterations for ddp
bool usemocap = false;

double tprev = 0;

tf::Vector3 yprev;

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

void iterateCallback(const ros::TimerEvent & event)
//void iterateCallback()
{
//	ros::Time startime = ros::Time::now();
struct timeval timer; 
timer_start(timer);
	for (int count = 1;count <= Nit;count++){
		 
		ddp->Iterate();//Updates us and xs after one iteration
	}//double te = 1e6*(ros::Time::now() - startime).toSec();
 long te = timer_us(timer);
	cout << "Time taken " << te << " us." << endl;
	
//publish the message
//	ros::Duration d1 = ros::Time::now()-startime;
//	ROS_INFO("Duration %f",d1.toSec());
	//iteratetimer.stop();
	pubtraj();
}

void posnCallback(const geometry_msgs::PoseStamped::ConstPtr &posnmsg)
{

// Convert quaternion to RPY.
        tf::Quaternion q;
        double roll, pitch, yaw;
        tf::quaternionMsgToTF(posnmsg->pose.orientation, q);
        tf::Matrix3x3(q).getRPY(roll, pitch, yaw);
       // ROS_DEBUG("RPY = (%lf, %lf, %lf)", roll, pitch, yaw);

       
        xf(0) = posnmsg->pose.position.x;
	xf(1) = posnmsg->pose.position.y;
	xf(2) = yaw;
        std::cout<<"Yaw final: "<<yaw<<std::endl;

}

void initialposnCallback(const gcop_comm::CurrPose::ConstPtr &currframe)
{
	if(!usemocap)
		return;
	tf::StampedTransform UV_O;
	tf::Quaternion quat;
	tf::quaternionMsgToTF((*currframe).pose.transform.rotation,quat);
	transformStampedMsgToTF((*currframe).pose,UV_O);//converts to the right format 
	//getrpy:
	double roll,pitch,yaw;
        UV_O.getBasis().getRPY(roll,pitch,yaw);
        double tcurr = currframe->pose.header.stamp.toSec();
        tf::Vector3 y = UV_O.getOrigin();

	 tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);
/*
	double dt = (tcurr - tprev);
	tf::Vector3 dydt = (y - yprev)/dt;
	tf::Vector3 yawvec(cos(yaw),sin(yaw),0);
*/
	Vector4d x0 = Vector4d::Zero();// initial state
        x0[0] = y[0];
        x0[1] = y[1];
        x0[2] = yaw;
	x0[3] = currframe->velocity; //xf(3);????//ASK JESSE WHY THIS IS HERE??
  //WHY IS XF(3) not being set anywhere??

        std::cout<<"Yaw initial: "<<yaw<<std::endl;

        float tm1 = sqrt(pow(xf(0)-x0(0),2) + pow(xf(1)-x0(1),2)+ pow(xf(2)-x0(2),2) + pow(xf(3)-x0(3),2));
	
        cost->tf = tm1;
	int N = us.size();
	float h = tm1/N;
	//for (int k = 0; k <=N; ++k)
	//	ts[k] = k*h;
	//ddp->ts = ts;
	xs[0] = x0;
	//ros::TimerEvent e1;
	//iterateCallback(e1);
	return;
}

void paramreqcallback(gcop_ctrl::DMocInterfaceConfig &config, uint32_t level) 
{
	Nit = config.Nit; 
	//int N = config.N;
	//double h = config.tf/N;   // time step
	//Vector4d xf = Vector4d::Zero();// final state
	Vector4d x0 = Vector4d::Zero();// initial state
	VectorXd Q(4);//Costs
	VectorXd R(2);  
	VectorXd Qf(4);

	xf(0) = config.xN;
	xf(1) = config.yN;
	xf(2) = config.thetaN;
	xf(3) = config.vN;

	x0(0) = config.x0;
	x0(1) = config.y0;
	x0(2) = config.theta0;
	x0(3) = config.v0;

	Q(0) = config.Q1;
	Q(1) = config.Q2;
	Q(2) = config.Q3;
	Q(3) = config.Q4;

	Qf(0) = config.Qf1;
	Qf(1) = config.Qf2;
	Qf(2) = config.Qf3;
	Qf(3) = config.Qf4;

	R(0) = config.R1;
	R(1) = config.R2;

	//resize
/*
	ts.resize(N+1);
	xs.resize(N+1);
	us.resize(N);
	trajectory.N = N;
	trajectory.statemsg.resize(N+1);
	trajectory.ctrl.resize(N);
*/	
	// cost

	//RnLqCost<4, 2> cost(config.tf, xf);
	if(!config.usemocap)
		cost->tf = config.tf;
	//cost->xf = xf; //xf is already changed when we assign xf its new values

	cost->Q = Q.asDiagonal();
	cost->R = R.asDiagonal();
	cost->Qf = Qf.asDiagonal();


//	for (int k = 0; k <=N; ++k)
//		ts[k] = k*h;

	// initial state
	if(!config.usemocap)
	{	xs[0] = x0;}
	usemocap = config.usemocap;
	int N = us.size();
	if(config.reinitialize == 1)
	{
	  for (int i = 0; i < N/2; ++i) {
	    us[i] = Vector2d(.01, .0);
	    us[N/2+i] = Vector2d(-.01, .0);
	  } 
	  config.reinitialize = 0;
	}
	else if(config.reinitialize == -1)
	{
	  for (int i = 0; i < N/2; ++i) {
	    us[i] = Vector2d(-.01, .0);
	    us[N/2+i] = Vector2d(.01, .0);
	  }
	  config.reinitialize = 0; 
	}
	//initial controls
	/*for (int i = 0; i < N/2; ++i) {
		us[i] = Vector2d(.01, .0);
		us[N/2+i] = Vector2d(-.01, .0);
	}
	*/
	//change parameters in ddp:
	ddp->ts = ts;
//	ddp->xs = xs;
	//ddp->us = us;
	ddp->mu = config.mu;



	//dont know what to do with the cost


	//destroy previous ddp:
	//delete(ddp);
	//ddp = new RccarDdp(sys, cost, ts, xs, us);  
}

int main(int argc, char** argv)
{
	ros::init(argc, argv, "rccarctrl");
	ros::NodeHandle rosddp("/ddp");
        ros::NodeHandle n2;
	//Initialize publisher
   trajpub = rosddp.advertise<gcop_comm::CtrlTraj>("ctrltraj",1);
	//Subscribe to initial posn from tf
	initialposn_sub = rosddp.subscribe<gcop_comm::CurrPose>("mocap",1,initialposnCallback);

	posn_sub = n2.subscribe<geometry_msgs::PoseStamped>("/target",1,posnCallback);
	

	//define parameters for the system
  int N = 64;        // number of segments
  double tf = 10,mu = 0.01;    // time horizon
  Vector4d x0 = Vector4d::Zero();// initial state
  VectorXd Q(4);//Costs
  VectorXd R(2);  
  VectorXd Qf(4);
	xf.setZero();
//sys.U.lb[0] = -2;
//sys.U.ub[0] = 2;
sys.U.lb[1] =  tan(-M_PI/8);
sys.U.ub[1] = tan(M_PI/8);

/* x0[0] = 1;x0[1] = 1;
  Q(0) = 0.5; Q(1) = 0.5;Q(2) = 0.1; Q(3) = 1;
	R(0)=0.5;R(1) = 0.1;
	Qf(0) = 10;Qf(1) = 10;Qf(2)=10;Qf(3) = 1;*/ 

		//get parameters from ros:
	ros::param::get("/ddp/tf", tf);
	ros::param::get("/ddp/N", N);

	ros::param::get("/ddp/x0", x0(0));
	ros::param::get("/ddp/y0", x0(1));
	ros::param::get("/ddp/vx0", x0(2));
	ros::param::get("/ddp/vy0", x0(3));

	ros::param::get("/ddp/xN", xf(0));
	ros::param::get("/ddp/xN", xf(1));
	ros::param::get("/ddp/vy0", xf(2));
	ros::param::get("/ddp/vyN", xf(3));

	ros::param::get("/ddp/Qf1", Qf(0));
	ros::param::get("/ddp/Qf2", Qf(1));
	ros::param::get("/ddp/Qf3", Qf(2));
	ros::param::get("/ddp/Qf4", Qf(3));

	ros::param::get("/ddp/Q1", Q(0));
	ros::param::get("/ddp/Q2", Q(1));
	ros::param::get("/ddp/Q3", Q(2));
	ros::param::get("/ddp/Q4", Q(3));

	ros::param::get("/ddp/R1", R(0));
	ros::param::get("/ddp/R2", R(1));

	ros::param::get("/ddp/mu", mu);
	
	ros::param::get("/ddp/Nit", Nit);

	
	//resize the states and controls
	ts.resize(N+1);
	xs.resize(N+1);
	us.resize(N);

	  //conversions:
  double h = tf/N;   // time step

	cost = new RnLqCost<4, 2>(sys,tf,xf);

	cost->Q = Q.asDiagonal();
  cost->R = R.asDiagonal();
  cost->Qf = Qf.asDiagonal();

  for (int k = 0; k <=N; ++k)
    ts[k] = k*h;

  // initial state
  xs[0] = x0;
  //initial controls
  for (int i = 0; i < N/2; ++i) {
    us[i] = Vector2d(.01, .0);
    us[N/2+i] = Vector2d(-.01, .0);
  }
 
 
  ddp = new RccarDdp(sys, *cost, ts, xs, us);  
  ddp->mu = mu;
  ddp->debug = false;


  //Trajectory message initialization
	trajectory.N = N;
	trajectory.statemsg.resize(N+1);
	trajectory.ctrl.resize(N);
	trajectory.time = ts;
	trajectory.finalgoal.statevector.resize(4);
	//trajectory.time.resize(N);
//Dynamic Reconfigure setup Callback ! immediately gets called with default values	
	dynamic_reconfigure::Server<gcop_ctrl::DMocInterfaceConfig> server;
	dynamic_reconfigure::Server<gcop_ctrl::DMocInterfaceConfig>::CallbackType f;
	f = boost::bind(&paramreqcallback, _1, _2);
	server.setCallback(f);
	
//	ros::TimerEvent event;

//  iterateCallback(event);
	//create timer for iteration
  iteratetimer = rosddp.createTimer(ros::Duration(0.01), iterateCallback);
	iteratetimer.start();
	ros::spin();
  return 0;
}
