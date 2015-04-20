/** This is a node for external programs to control Multi body systems (mbs). This creates a
  * mbs system from a custom URDF file in params folders.A rnlq cost based on joints and base position is created and an optimization method (DDP) 
  * is used to create optimal reference trajectories.
  * The node accepts iteration request which is initial state of the mbs, final goal, number of iterations etc through a topic (Iteration_request) and 
  * outputs a trajectory on another topic. This is non blocking and allows for the requesting node to work on other stuff which the optimization is happening
  * The optimized trajectories are published on a topic("CtrlTraj") which has all the controls and states
  *
  * Author: Gowtham Garimella
*/

#include "ros/ros.h"

#include <iomanip>
#include <iostream>



#include <gcop/urdf_parser.h>
#include <gcop/se3.h>
#include "gcop/ddp.h" //gcop ddp header
#include "gcop/lqcost.h" //gcop lqr header
#include "gcop/rn.h"
#include "gcop_ctrl/MbsNodeInterfaceConfig.h"

#include "tf/transform_datatypes.h"
//#include <tf/transform_listener.h>
#include "tf2_ros/static_transform_broadcaster.h"
#include <dynamic_reconfigure/server.h>
#include <XmlRpcValue.h>//To access xml arrays in parameters 

#include <visualization_msgs/Marker.h>
#include <gcop_comm/Iteration_req.h>
#include <gcop_comm/CtrlTraj.h>
#include <gcop_comm/Trajectory_req.h>
#include <sensor_msgs/JointState.h>

//#include <signal.h>

using namespace std;
using namespace Eigen;
using namespace gcop;

typedef Ddp<MbsState> MbsDdp;//defining chainddp


//ros messages
visualization_msgs::Marker trajectory_marker;
std::vector<sensor_msgs::JointState> jointstate_vec;
std::vector<geometry_msgs::TransformStamped> basetransform_msg;

//Timer

//Subscriber
ros::Subscriber iterationreq_sub;

//Publisher
ros::Publisher trajectory_pub;
ros::Publisher visualize_trajectory_pub;
std::vector<ros::Publisher> jointpub_vec;

//Pointer for mbs system
boost::shared_ptr<Mbs> mbsmodel;

//Pointer for Optimal Controller
boost::shared_ptr<MbsDdp> mbsddp;

//MbsState final state
boost::shared_ptr<MbsState> xf;

//Cost lqcost
boost::shared_ptr<LqCost<MbsState>> cost;

int Nit = 1;//number of iterations for ddp
int Nrobotstovisualize = 3;//Number of robots to visualize other than the moving robot
string mbstype; // Type of system
Matrix4d gposeroot_i; //inital inertial frame wrt the joint frame
Matrix4d gposei_root; //joint frame wrt inital inertial frame
gcop_comm::CtrlTraj current_traj;
//int traj_size = 0;


//Helper Functions
void q2transform(geometry_msgs::Transform &transformmsg, Vector6d &bpose)
{
	tf::Quaternion q;
	q.setEulerZYX(bpose[2],bpose[1],bpose[0]);
	tf::Vector3 v(bpose[3],bpose[4],bpose[5]);
	tf::Transform tftransform(q,v);
	tf::transformTFToMsg(tftransform,transformmsg);
	//cout<<"---------"<<endl<<transformmsg.position.x<<endl<<transformmsg.position.y<<endl<<transformmsg.position.z<<endl<<endl<<"----------"<<endl;
}

void xml2vec(VectorXd &vec, XmlRpc::XmlRpcValue &my_list)
{
	ROS_ASSERT(my_list.getType() == XmlRpc::XmlRpcValue::TypeArray);
	ROS_ASSERT(my_list.size() > 0);
	vec.resize(my_list.size());

	for (int32_t i = 0; i < my_list.size(); i++) 
	{
				ROS_ASSERT(my_list[i].getType() == XmlRpc::XmlRpcValue::TypeDouble);
				cout<<"my_list["<<i<<"]\t"<<my_list[i]<<endl;
			  vec[i] =  (double)(my_list[i]);
	}
}

inline void msgtogcoptwist(const geometry_msgs::Twist &in, Vector6d &out)
{
	out<<(in.angular.x),(in.angular.y),(in.angular.z),(in.linear.x),(in.linear.y),(in.linear.z);
}

inline void gcoptwisttomsg(const Vector6d &in, geometry_msgs::Twist &out)
{
	out.angular.x = in[0]; out.angular.y = in[1]; out.angular.z = in[2];
	out.linear.x = in[3]; out.linear.y = in[4]; out.linear.z = in[5];
}

void fill_mbsstate(MbsState &mbs_state,const gcop_comm::State &mbs_statemsg)
{
	int nb = mbsmodel->nb;
	//Joint Angles
	if(mbs_statemsg.statevector.size() == 2*(nb-1))//Number of joint angles and velocities check
	{
		for(int count1 = 0;count1 < nb-1;count1++)
		{
			mbs_state.r[count1] = mbs_statemsg.statevector[count1];
			mbs_state.dr[count1] = mbs_statemsg.statevector[nb-1 + count1];//Can also do finite diff here instead of asking for it
		}
	}
	else
	{
		ROS_INFO("Wrong size of joint angles and vel provided");
	}

	//Pose
	Vector7d wquatxyz;
	wquatxyz<<(mbs_statemsg.basepose.rotation.w),(mbs_statemsg.basepose.rotation.x),(mbs_statemsg.basepose.rotation.y),(mbs_statemsg.basepose.rotation.z),(mbs_statemsg.basepose.translation.x),(mbs_statemsg.basepose.translation.y),(mbs_statemsg.basepose.translation.z);
	gcop::SE3::Instance().quatxyz2g(mbs_state.gs[0], wquatxyz);
	mbs_state.gs[0] = gposei_root*mbs_state.gs[0];//To move towards the inertial posn of the base

	//Body Fixed Velocities Have to use adjoint to transform this to inertial Frame TODO (For now no problem for quadcopter)
	msgtogcoptwist(mbs_statemsg.basetwist,mbs_state.vs[0]);
}

void filltraj(gcop_comm::CtrlTraj &trajectory, int N1, bool resize=true) //N1 is the number of segments requested
{
	if(!mbsddp)
		return;
	cout<<"N1: "<<N1<<endl;
	cout<<"Existing Traj_size: "<<trajectory.ctrl.size()<<endl;
	int csize = mbsmodel->U.n;
	cout<<"csize: "<<csize<<endl;
	int nb = mbsmodel->nb;
	cout<<"nb: "<<nb<<endl;
	Vector6d bpose;//Temp variable

	//Prepare trajectory msg:
	//Will not be needed if we use nodelets and somehow pass it by pointer
	//Or create a library to be used directly
	if(resize && (N1 != trajectory.ctrl.size()))//Can add more checks for if resize is needed or not
	{
		trajectory.N = N1;
		trajectory.statemsg.resize(N1+1);
		trajectory.ctrl.resize(N1);
		for(int count1 = 0;count1 <=N1;count1++)
			trajectory.time.push_back(mbsddp->ts[count1]);
		trajectory.rootname = mbsmodel->links[0].name;

		trajectory.finalgoal.statevector.resize(nb-1);
		trajectory.finalgoal.names.resize(nb-1);

		trajectory.statemsg[0].statevector.resize(nb-1);
		trajectory.statemsg[0].names.resize(nb-1);

		for (int i = 0; i < N1; ++i) 
		{
			trajectory.statemsg[i+1].statevector.resize(2*(nb-1));
			trajectory.statemsg[i+1].names.resize(nb-1);
			trajectory.ctrl[i].ctrlvec.resize(mbsmodel->U.n);
		}
	}

	gcop::SE3::Instance().g2q(bpose,gposeroot_i*mbsddp->xs[0].gs[0]);//Can convert to 7dof instead of 6dof to avoid singularities TODO
	q2transform(trajectory.statemsg[0].basepose , bpose);
	gcoptwisttomsg(mbsddp->xs[0].vs[0],trajectory.statemsg[0].basetwist);//No conversion from inertial frame to the visual frame 

	for(int count1 = 0;count1 < nb-1;count1++)
	{
		trajectory.statemsg[0].statevector[count1] = mbsddp->xs[0].r[count1];
		trajectory.statemsg[0].names[count1] = mbsmodel->joints[count1].name;
	}
	trajectory.time[0] = mbsddp->ts[0];
	cout<<"Filled init state"<<endl;

	for (int i = 0; i < N1; ++i) 
	{
		gcop::SE3::Instance().g2q(bpose, gposeroot_i*mbsddp->xs[i+1].gs[0]);
		q2transform(trajectory.statemsg[i+1].basepose,bpose);
		gcoptwisttomsg(mbsddp->xs[i+1].vs[0],trajectory.statemsg[i+1].basetwist);//No conversion from inertial frame to the visual frame 

		for(int count1 = 0;count1 < nb-1;count1++)
		{
			trajectory.statemsg[i+1].statevector[count1] = mbsddp->xs[i+1].r[count1];
			trajectory.statemsg[i+1].statevector[nb-1 + count1] = mbsddp->xs[i+1].dr[count1];
			trajectory.statemsg[i+1].names[count1] = mbsmodel->joints[count1].name;
		}
		for(int count1 = 0;count1 < csize;count1++)
		{
			trajectory.ctrl[i].ctrlvec[count1] = mbsddp->us[i](count1);
		}
		trajectory.time[i+1] = mbsddp->ts[i+1];
	}
	//final goal:
	gcop::SE3::Instance().g2q(bpose, gposeroot_i*xf->gs[0]);
	q2transform(trajectory.finalgoal.basepose,bpose);

	for(int count1 = 0;count1 < nb-1;count1++)
	{
		trajectory.finalgoal.statevector[count1] = xf->r[count1];
		trajectory.finalgoal.names[count1] = mbsmodel->joints[count1].name;
	}
}

void visualize_publish_traj()
{
	int N = mbsddp->us.size();
	int nb = mbsmodel->nb;
	Vector6d bpose;//Temp variable
	static tf2_ros::StaticTransformBroadcaster broadcaster;

	//publish the trajectory
	trajectory_marker.header.stamp  = ros::Time::now();

	for(int i =0;i<(N+1); i++)
	{
		gcop::SE3::Instance().g2q(bpose, gposeroot_i*mbsddp->xs[i].gs[0]);
		trajectory_marker.points[i].x = bpose[3];
		trajectory_marker.points[i].y = bpose[4];
		trajectory_marker.points[i].z = bpose[5];
	}
	visualize_trajectory_pub.publish(trajectory_marker);

	//Publish joint states:
	//Publish Static Transform to visualize the intermediate robots:
	for(int i = 0; i < Nrobotstovisualize;i++)
	{
		jointstate_vec[i].header.stamp = ros::Time::now();
		basetransform_msg[i].header.stamp = ros::Time::now();

		int indextoshow = (N*(i+1))/(Nrobotstovisualize);

		for(int count1 = 0; count1 < (nb-1); count1++)
		{
			jointstate_vec[i].position[count1] = mbsddp->xs[indextoshow].r[count1];
		}

		gcop::SE3::Instance().g2q(bpose, gposeroot_i*mbsddp->xs[indextoshow].gs[0]);
		q2transform(basetransform_msg[i].transform,bpose);

		jointpub_vec[i].publish(jointstate_vec[i]);
		broadcaster.sendTransform(basetransform_msg[i]);
	}

	for(int count1 = 0; count1 < (nb-1); count1++)
		jointstate_vec[Nrobotstovisualize].position[count1] = xf->r[count1];

	basetransform_msg[Nrobotstovisualize].header.stamp = ros::Time::now();
	jointstate_vec[Nrobotstovisualize].header.stamp = ros::Time::now();

	gcop::SE3::Instance().g2q(bpose, gposeroot_i*xf->gs[0]);
	q2transform(basetransform_msg[Nrobotstovisualize].transform,bpose);

	jointpub_vec[Nrobotstovisualize].publish(jointstate_vec[Nrobotstovisualize]);//Final Goal
	broadcaster.sendTransform(basetransform_msg[Nrobotstovisualize]);
}


//Actual Callback Functions
void iteration_request(const gcop_comm::Iteration_req &req)
{
	if(!mbsddp)
		return;
	int N = mbsddp->us.size();

	//Fill the initial state mbs
	fill_mbsstate(mbsddp->xs[0], req.x0);
	//Also need to generate a trajectory which is consistent with the initial conditions and the controls

	//Fill the final state mbs
	fill_mbsstate(*xf, req.xf);

	cost->tf = (req.tf < 0.01)?0.01:req.tf;
	cout<<"tf: "<<(cost->tf)<<endl;

	double h = (cost->tf)/N;   // time step
	for (int k = 0; k <=N; ++k)
		mbsddp->ts[k] = k*h;

	mbsmodel->Rec(mbsddp->xs[0], h);//To fill the transformation matrices

	//Until we figure out how to give the right controls which will not make the system go crazy, we give the controls
	// that cancel the external forces and make system not move at all.
	// This is fine for open loop with lots of iterations in the beginning
	VectorXd u(mbsmodel->U.n);
	u.setZero();

	//States and controls for system
	cout<<"Finding Biases"<<endl;
	int n11 = mbsmodel->nb -1 + 6*(!mbsmodel->fixed);
	VectorXd forces(n11);
	mbsmodel->Bias(forces,0,mbsddp->xs[0]);
	cout<<"Bias computed: "<<forces.transpose()<<endl;

	//Set Controls to cancel the ext forces:
	if(mbstype == "FLOATBASE")
	{
		assert(6 + mbsmodel->nb - 1 == forces.size());
		u.head(6) = forces.head(6);
	}
	else if(mbstype == "AIRBASE")
	{
		u[0] = forces[0];
		u[1] = forces[1];
		u[2] = forces[2];
		u[3] = forces[5];	
	}
	//Add more types when they are here

	//Joint Torques:
	u.tail((mbsmodel->nb)-1) = forces.tail((mbsmodel->nb)-1);

	//Setup the trajectory using the above controls:
	for(int i = 0; i < N;i ++)
	{
		mbsddp->us[i] = u;
		mbsmodel->Step(mbsddp->xs[i+1], i*h, mbsddp->xs[i], u, h);
	}

	//Iterating:
	ros::Time startime = ros::Time::now(); 
	for (int count = 1;count <= Nit;count++)
	{
		mbsddp->Iterate();//Updates us and xs after one iteration
	}
	double te = 1e6*(ros::Time::now() - startime).toSec();
	cout << "Time taken " << te << " us." << endl;

	int N1 = (req.N == 0 || req.N > N)?N:req.N;
	filltraj(current_traj, N1);

	visualize_publish_traj();//Publish to rviz

	//Publish Trajectory
	trajectory_pub.publish(current_traj);

	return;
}

void paramreqcallback(gcop_ctrl::MbsNodeInterfaceConfig &config, uint32_t level) 
{
	if(!mbsddp)
		return;
	cout<<"Level: "<<level<<endl;
	if(level == 0xFFFFFFFF)
	{
		//Initialization:
		config.Nit = Nit;
	}
	//Setting Values
	if(level & 0x00000002)
	{
		Nit = config.Nit; 
	}
	//mbsddp->mu = config.mu;[ONLY SET IN BEGINNING]
}


int main(int argc, char** argv)
{
	ros::init(argc, argv, "chainload");
	ros::NodeHandle n("mbsddp");

	//get parameter for xml_string:
	string xml_string, xml_filename;
	if(!ros::param::get("/robot_description", xml_string))
	{
		ROS_ERROR("Could not fetch xml file name");
		return 0;
	}
	n.getParam("basetype",mbstype);
	assert((mbstype == "FIXEDBASE")||(mbstype == "AIRBASE")||(mbstype == "FLOATBASE"));
	VectorXd xmlconversion;
	//Create Mbs system
	mbsmodel = gcop_urdf::mbsgenerator(xml_string,gposei_root, mbstype);
	//[NOTE]mbsmodel->U.bnd = false;
	gcop::SE3::Instance().inv(gposeroot_i,gposei_root);
	cout<<"Mbstype: "<<mbstype<<endl;
	mbsmodel->ag << 0, 0, -0.05;
	//get ag from parameters
	XmlRpc::XmlRpcValue ag_list;
	if(n.getParam("ag", ag_list))
		xml2vec(xmlconversion,ag_list);
	ROS_ASSERT(xmlconversion.size() == 3);
	mbsmodel->ag = xmlconversion.head(3);

	//Printing the mbsmodel params:
	for(int count = 0;count<(mbsmodel->nb);count++)
	{
		cout<<"Ds["<<mbsmodel->links[count].name<<"]"<<endl<<mbsmodel->links[count].ds<<endl;
		cout<<"I["<<mbsmodel->links[count].name<<"]"<<endl<<mbsmodel->links[count].I<<endl;
	}
	for(int count = 0;count<(mbsmodel->nb)-1;count++)
	{
		cout<<"Joint["<<mbsmodel->joints[count].name<<"].gc"<<endl<<mbsmodel->joints[count].gc<<endl;
		cout<<"Joint["<<mbsmodel->joints[count].name<<"].gp"<<endl<<mbsmodel->joints[count].gp<<endl;
		cout<<"Joint["<<mbsmodel->joints[count].name<<"].a"<<endl<<mbsmodel->joints[count].a<<endl;
	}
	cout<<"mbsmodel damping: "<<mbsmodel->damping.transpose()<<endl;

	int ctrlveclength = (mbstype == "AIRBASE")? 4:
		(mbstype == "FIXEDBASE")? 6:
		(mbstype == "FLOATBASE")? 0:0;
	cout<<"ctrlveclength "<<ctrlveclength<<endl;
	//set bounds on basebody controls
	XmlRpc::XmlRpcValue ub_list;
	if(n.getParam("ulb", ub_list))
	{
		xml2vec(xmlconversion,ub_list);
		ROS_ASSERT(xmlconversion.size() == ctrlveclength);
		mbsmodel->U.lb.head(ctrlveclength) = xmlconversion;
	}
	//upper bound
	if(n.getParam("uub", ub_list))
	{
		xml2vec(xmlconversion,ub_list);
		ROS_ASSERT(xmlconversion.size() == ctrlveclength);
		mbsmodel->U.ub.head(ctrlveclength) = xmlconversion;
	}
	cout<<"mbsmodel U.lb: "<<mbsmodel->U.lb.transpose()<<endl;
	cout<<"mbsmodel U.ub: "<<mbsmodel->U.ub.transpose()<<endl;
	cout<<"mbsmodel X.lb.gs[0]"<<endl<<mbsmodel->X.lb.gs[0]<<endl;
	cout<<"mbsmodel X.ub.gs[0]"<<endl<<mbsmodel->X.ub.gs[0]<<endl;
	cout<<"mbsmodel X.lb.vs[0]: "<<mbsmodel->X.lb.vs[0].transpose()<<endl;
	cout<<"mbsmodel X.ub.vs[0]: "<<mbsmodel->X.ub.vs[0].transpose()<<endl;
	//Initialize after setting everything up
	mbsmodel->Init();


	//Using mbsmodel:
	//define parameters for the system
	int nb = mbsmodel->nb;
	double tf = 20;   // time-horizon
	int N = 100;      // discrete trajectory segments

	n.getParam("tf",tf);
	n.getParam("N",N);

	double h = tf/N; // time-step



	//times
	vector<double> ts(N+1);
	for (int k = 0; k <=N; ++k)
		ts[k] = k*h;


	//Define Final State [This will change based on the service call]
	xf.reset( new MbsState(nb,(mbstype == "FIXEDBASE")));
	xf->gs[0].setIdentity();
	xf->vs[0].setZero();
	xf->dr.setZero();
	xf->r.setZero();

	// Get Xf	 from params
	XmlRpc::XmlRpcValue xf_list;
	if(n.getParam("XN", xf_list))
	{
		xml2vec(xmlconversion,xf_list);
		ROS_ASSERT(xmlconversion.size() == 12);
		xf->vs[0] = xmlconversion.tail<6>();
		gcop::SE3::Instance().rpyxyz2g(xf->gs[0],xmlconversion.head<3>(),xmlconversion.segment<3>(3)); 
	}
	xf->gs[0] = gposei_root*xf->gs[0];//new stuff with transformations
	//list of joint angles:
	XmlRpc::XmlRpcValue xfj_list;
	if(n.getParam("JN", xfj_list))
		xml2vec(xf->r,xfj_list);
	cout<<"xf->r"<<endl<<xf->r<<endl;

	if(n.getParam("JvN", xfj_list))
		xml2vec(xf->dr,xfj_list);
	cout<<"xf->dr"<<endl<<xf->dr<<endl;


	//Define Lqr Cost
	cost.reset(new LqCost<MbsState>(*mbsmodel, tf, *xf));
	cost->Qf.setIdentity();
	/*	if(mbstype != "FIXEDBASE")
			{
			cost->Qf(0,0) = 2; cost->Qf(1,1) = 2; cost->Qf(2,2) = 2;
			cost->Qf(3,3) = 20; cost->Qf(4,4) = 20; cost->Qf(5,5) = 20;
			}
	 */
	//cost.Qf(9,9) = 20; cost.Qf(10,10) = 20; cost.Qf(11,11) = 20;
	//list of final cost :
	XmlRpc::XmlRpcValue finalcost_list;
	if(n.getParam("Qf", finalcost_list))
	{
		cout<<mbsmodel->X.n<<endl;
		xml2vec(xmlconversion,finalcost_list);
		cout<<"conversion"<<endl<<xmlconversion<<endl;
		ROS_ASSERT(xmlconversion.size() == mbsmodel->X.n);
		cost->Qf = xmlconversion.asDiagonal();
		cout<<"Cost.Qf"<<endl<<cost->Qf<<endl;
	}
	//
	XmlRpc::XmlRpcValue statecost_list;
	if(n.getParam("Q", statecost_list))
	{
		cout<<mbsmodel->X.n<<endl;
		xml2vec(xmlconversion,statecost_list);
		cout<<"conversion"<<endl<<xmlconversion<<endl;
		ROS_ASSERT(xmlconversion.size() == mbsmodel->X.n);
		cost->Q = xmlconversion.asDiagonal();
		cout<<"Cost.Q"<<endl<<cost->Q<<endl;
	}
	XmlRpc::XmlRpcValue ctrlcost_list;
	if(n.getParam("R", ctrlcost_list))
	{
		cout<<mbsmodel->U.n<<endl;
		xml2vec(xmlconversion,ctrlcost_list);
		ROS_ASSERT(xmlconversion.size() == mbsmodel->U.n);
		cout<<"conversion"<<endl<<xmlconversion<<endl;
		cost->R = xmlconversion.asDiagonal();
	}
	cout<<"Cost.R"<<endl<<cost->R<<endl;
	//


	//Define the initial state mbs
	MbsState x0(nb,(mbstype == "FIXEDBASE"));
	x0.gs[0].setIdentity();
	x0.vs[0].setZero();
	x0.dr.setZero();
	x0.r.setZero();

	// Get X0	 from params
	XmlRpc::XmlRpcValue x0_list;
	if(n.getParam("X0", x0_list))
	{
		xml2vec(xmlconversion,x0_list);
		ROS_ASSERT(xmlconversion.size() == 12);
		x0.vs[0] = xmlconversion.tail<6>();
		gcop::SE3::Instance().rpyxyz2g(x0.gs[0],xmlconversion.head<3>(),xmlconversion.segment<3>(3)); 
	}
	x0.gs[0] = gposei_root*x0.gs[0];//new stuff with transformations
	//list of joint angles:
	XmlRpc::XmlRpcValue j_list;
	if(n.getParam("J0", j_list))
		xml2vec(x0.r,j_list);
	cout<<"x.r"<<endl<<x0.r<<endl;
	if(n.getParam("Jv0", j_list))
		xml2vec(x0.dr,j_list);
	cout<<"x.dr"<<endl<<x0.dr<<endl;
	mbsmodel->Rec(x0, h);

	// initial controls (e.g. hover at one place)
	VectorXd u(mbsmodel->U.n);
	u.setZero();

	//States and controls for system
	cout<<"Finding Biases"<<endl;
	int n11 = mbsmodel->nb -1 + 6*(!mbsmodel->fixed);
	VectorXd forces(n11);
	mbsmodel->Bias(forces,0,x0);
	cout<<"Bias computed: "<<forces.transpose()<<endl;

	//Set Controls to cancel the ext forces:
	if(mbstype == "FLOATBASE")
	{
		assert(6 + mbsmodel->nb - 1 == forces.size());
		u.head(6) = forces.head(6);
	}
	else if(mbstype == "AIRBASE")
	{
		u[0] = forces[0];
		u[1] = forces[1];
		u[2] = forces[2];
		u[3] = forces[5];	
	}
	//Add more types when they are here

	//Joint Torques:
	u.tail(nb-1) = forces.tail(nb-1);

	vector<VectorXd> us(N,u);
	vector<MbsState> xs(N+1,x0);
	//bool usectrl = true;

	//Creating Optimal Control Problem
	mbsddp.reset(new MbsDdp(*mbsmodel, *cost, ts, xs, us));

	mbsddp->mu = 10.0;
	n.getParam("mu",mbsddp->mu);

	//Debug false for mbs (No more debugging)
	mbsmodel->debug = false;


	// Get number of iterations
	n.getParam("Nit",Nit);
	// Create timer for iterating	
	//iteratetimer = n.createTimer(ros::Duration(0.1), iterateCallback);

	//	Dynamic Reconfigure setup Callback ! immediately gets called with default values or the param values which can be set
	dynamic_reconfigure::Server<gcop_ctrl::MbsNodeInterfaceConfig> server;
	dynamic_reconfigure::Server<gcop_ctrl::MbsNodeInterfaceConfig>::CallbackType f;
	f = boost::bind(&paramreqcallback, _1, _2);
	server.setCallback(f);

	//Setup the trajectory marker:
	trajectory_marker.header.frame_id = "/optitrak";
	trajectory_marker.ns = "traj";
	trajectory_marker.action = visualization_msgs::Marker::ADD;
	trajectory_marker.pose.orientation.w = 1.0;
	trajectory_marker.id = 1;
	trajectory_marker.type = visualization_msgs::Marker::LINE_STRIP;
	trajectory_marker.scale.x = 0.05;
	trajectory_marker.scale.y = 0.05;
	trajectory_marker.scale.z = 0.05;
	trajectory_marker.color.b = 1.0;
	trajectory_marker.color.a = 1.0;
	trajectory_marker.points.resize(N + 1);

//Joint State Variable 	
	sensor_msgs::JointState tempjoint;
	tempjoint.position.resize(nb-1);
	for(int count2 = 0;count2 < (nb-1); count2++)
		tempjoint.name.push_back(mbsmodel->joints[count2].name);

//base transform variable
	geometry_msgs::TransformStamped temp_trans;
	temp_trans.header.frame_id = "/optitrak";

	//Setup ros server for accepting trajectory requests:
	trajectory_pub = n.advertise<gcop_comm::CtrlTraj>("traj_resp",1);
	visualize_trajectory_pub = n.advertise<visualization_msgs::Marker>("desired_traj",1);

	for(int count1 = 1;count1 < Nrobotstovisualize+1; count1++)
	{
		string robotname = "/vizrobot"+ to_string(count1);
		jointpub_vec.push_back(n.advertise<sensor_msgs::JointState>(string(robotname+"/joint_states"),1));

		tempjoint.header.frame_id = robotname;
		jointstate_vec.push_back(tempjoint);

		temp_trans.child_frame_id = robotname + "/" + mbsmodel->links[0].name;
		basetransform_msg.push_back(temp_trans);
	}

	//Add one more joint state publisher  and variabl for goal state
	jointpub_vec.push_back(n.advertise<sensor_msgs::JointState>("/goalrobot/joint_states",1));

	tempjoint.header.frame_id = "goalrobot";
	jointstate_vec.push_back(tempjoint);

	iterationreq_sub = n.subscribe("iteration_req", 1, iteration_request);

	temp_trans.child_frame_id = "/goalrobot/" + mbsmodel->links[0].name;
	basetransform_msg.push_back(temp_trans);

	ros::spin();
	return 0;
}
