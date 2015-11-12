/** This is an example on how to control a Multibodysystem using GCOP Library. This creates a 
  * mbs system from a custom URDF file in params folders.A rnlq cost based on joints and base position is created and an optimization method (DDP) 
  * is used to create optimal reference trajectories.
  * The optimized trajectories are published on a topic("CtrlTraj") which has all the controls and states
  *
  * Author: Gowtham Garimella
*/
#include "ros/ros.h"
#include <iomanip>
#include <iostream>
#include <dynamic_reconfigure/server.h>
#include "gcop_comm/CtrlTraj.h"//msg for publishing ctrl trajectory
#include <gcop/urdf_parser.h>
#include "tf/transform_datatypes.h"
#include <gcop/se3.h>
#include "gcop/lqcost.h" //gcop lqr header
#include "gcop/rn.h"
#include "gcop/mbscontroller.h"
#include "gcop_ctrl/MbsDMocInterfaceConfig.h"
#include <tf/transform_listener.h>
#include <XmlRpcValue.h>
#include <fstream>

#define USE_GN

#ifdef USE_GN
#include "gcop/gndocp.h" //gcop ddp header
#include "gcop/controltparam.h"
#else
#include "gcop/ddp.h" //gcop ddp header
#endif
//#include <signal.h>

using namespace std;
using namespace Eigen;
using namespace gcop;


#ifdef USE_GN
  typedef GnDocp<MbsState> MbsGn;
#else
  typedef Ddp<MbsState> MbsDdp;//defining chainddp
#endif


//ros messages
gcop_comm::CtrlTraj trajectory;

//Publisher
ros::Publisher trajpub;

//Timer
ros::Timer iteratetimer;

//Pointer for mbs system
boost::shared_ptr<Mbs> mbsmodel;

//Pointer for Optimal Controller
#ifdef USE_GN
boost::shared_ptr<MbsGn> mbsopt;
#else
boost::shared_ptr<MbsDdp> mbsopt;
#endif

//MbsState final state
boost::shared_ptr<MbsState> xf;

//Cost lqcost
boost::shared_ptr<LqCost<MbsState>> cost;

boost::shared_ptr<MbsController> ctrl;
int Nit = 1;//number of iterations for ddp
int N = 100;      // discrete trajectory segments
string mbstype; // Type of system
Matrix4d gposeroot_i; //inital inertial frame wrt the joint frame
Matrix4d gposei_root; //inverse of gposeroot_i

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
void pubtraj() //N is the number of segments
{
	if(!mbsopt)
		return;
	int N = mbsopt->us.size();
	//cout<<"N: "<<N<<endl;
	int csize = mbsmodel->U.n;
	//cout<<"csize: "<<csize<<endl;
	int nb = mbsmodel->nb;
	//cout<<"nb: "<<nb<<endl;
	Vector6d bpose;

  ofstream logfile;
  std::string logfile_name;
  bool logfile_open = false;
  ros::param::param<std::string>("/logfile_name", logfile_name, "/home/gowtham/hydro_workspace/src/gcop_ros_packages/gcop_ctrl/log/Trajectory.txt");
  logfile.open(logfile_name);
  if(logfile.is_open())
    logfile_open = true;
  //Write the Header:
  {
    logfile<<"#Time\tX\tY\tZ\tYaw\tVx\tVy\tVz";
    for(int count = 1; count < nb; count++)
      logfile<<"\tJ"<<count;
    for(int count = 1; count < nb; count++)
      logfile<<"\tJv"<<count;
    logfile<<endl;
  }

	gcop::SE3::Instance().g2q(bpose,mbsopt->xs[0].gs[0]*gposeroot_i);
	q2transform(trajectory.statemsg[0].basepose,bpose);

	for(int count1 = 0;count1 < nb-1;count1++)
	{
		trajectory.statemsg[0].statevector[count1] = mbsopt->xs[0].r[count1];
		trajectory.statemsg[0].names[count1] = mbsmodel->joints[count1].name;
	}

  //Logfile 0 state:
  if(logfile_open)
  {
    logfile<<trajectory.time[0];//Time
    logfile<<"\t"<<trajectory.statemsg[0].basepose.translation.x<<"\t"<<trajectory.statemsg[0].basepose.translation.y<<"\t"<<trajectory.statemsg[0].basepose.translation.z;//XYZ
    logfile<<"\t"<<bpose[2];//Yaw
    logfile<<"\t"<<trajectory.statemsg[0].basetwist.linear.x<<"\t"<<trajectory.statemsg[0].basetwist.linear.y<<"\t"<<trajectory.statemsg[0].basetwist.linear.z;
    for(int count1 = 0; count1 < nb -1; count1++)
    {
      logfile<<"\t"<<trajectory.statemsg[0].statevector[count1];
    }
    for(int count1 = 0; count1 < nb -1; count1++)
    {
      logfile<<"\t"<<0;
    }
    logfile<<endl;
  }

	for (int i = 0; i < N; ++i) 
	{
		gcop::SE3::Instance().g2q(bpose, mbsopt->xs[i+1].gs[0]*gposeroot_i);
		q2transform(trajectory.statemsg[i+1].basepose,bpose);
		gcoptwisttomsg(mbsopt->xs[i+1].vs[0],trajectory.statemsg[i+1].basetwist);//No conversion from inertial frame to the visual frame 
		for(int count1 = 0;count1 < nb-1;count1++)
		{
			trajectory.statemsg[i+1].statevector[count1] = mbsopt->xs[i+1].r[count1];
			trajectory.statemsg[i+1].names[count1] = mbsmodel->joints[count1].name;
		}
		for(int count1 = 0;count1 < csize;count1++)
		{
			trajectory.ctrl[i].ctrlvec[count1] = mbsopt->us[i](count1);
		}
    //Logfile
    if(logfile_open)
    {
      logfile<<trajectory.time[i+1];//Time
      logfile<<"\t"<<trajectory.statemsg[i+1].basepose.translation.x<<"\t"<<trajectory.statemsg[i+1].basepose.translation.y<<"\t"<<trajectory.statemsg[i+1].basepose.translation.z;//XYZ
      logfile<<"\t"<<bpose[2];//Yaw
      logfile<<"\t"<<trajectory.statemsg[i+1].basetwist.linear.x<<"\t"<<trajectory.statemsg[i+1].basetwist.linear.y<<"\t"<<trajectory.statemsg[i+1].basetwist.linear.z;
      for(int count1 = 0; count1 < nb -1; count1++)
      {
        logfile<<"\t"<<trajectory.statemsg[i+1].statevector[count1];
      }
      for(int count1 = 0; count1 < nb -1; count1++)
      {
        logfile<<"\t"<<trajectory.statemsg[i+1].statevector[nb-1+count1];
      }
      logfile<<endl;
    }
	}
	//final goal:
	gcop::SE3::Instance().g2q(bpose, xf->gs[0]*gposeroot_i);
	q2transform(trajectory.finalgoal.basepose,bpose);

	for(int count1 = 0;count1 < nb-1;count1++)
	{
		trajectory.finalgoal.statevector[count1] = xf->r[count1];
		trajectory.finalgoal.names[count1] = mbsmodel->joints[count1].name;
	}

	trajectory.time = mbsopt->ts;
	trajpub.publish(trajectory);

}
void iterateCallback(const ros::TimerEvent & event)
{
	if(!mbsopt)
		return;
	ros::Time startime = ros::Time::now(); 
	for (int count = 1;count <= Nit;count++)
	{
		mbsopt->Iterate();//Updates us and xs after one iteration
	}
	double te = 1e6*(ros::Time::now() - startime).toSec();
	cout << "Time taken " << te << " us." << endl;
  cout<< "mbsopt->J "<<(mbsopt->J)<<endl;
	//publish the message
	pubtraj();
}

void paramreqcallback(gcop_ctrl::MbsDMocInterfaceConfig &config, uint32_t level) 
{
	if(!mbsopt)
		return;
	int nb = mbsmodel->nb;
	Nit = config.Nit; 
	//int N = config.N;
	double h = config.tf/N;   // time step

	if(level & 0x00000001)
	{
		Vector3d rpy;
		Vector3d xyz;

		if(config.i_Q > mbsmodel->X.n)
			config.i_Q = mbsmodel->X.n; 
		else if(config.i_J <= 0)
			config.i_Q = 1;

		config.Qfi = cost->Qf(config.i_Q -1,config.i_Q -1);
		config.Qi = cost->Q(config.i_Q -1,config.i_Q -1);

		if(config.i_R > mbsmodel->U.n)
			config.i_R = mbsmodel->U.n; 
		else if(config.i_R <= 0)
			config.i_R = 1;

		config.Ri = cost->R(config.i_R -1,config.i_R -1);

		if(config.final)
		{
			if(mbstype != "FIXEDBASE")
			{
				// overwrite the config with values from final state
				config.vroll = xf->vs[0](0);
				config.vpitch = xf->vs[0](1);
				config.vyaw = xf->vs[0](2);
				config.vx = xf->vs[0](3);
				config.vy = xf->vs[0](4);
				config.vz = xf->vs[0](5);

				gcop::SE3::Instance().g2rpyxyz(rpy,xyz,xf->gs[0]*gposeroot_i);
				config.roll = rpy(0);
				config.pitch = rpy(1);
				config.yaw = rpy(2);
				config.x = xyz(0);
				config.y = xyz(1);
				config.z = xyz(2);
			}

			if(config.i_J > nb-1)
				config.i_J = nb-1; 
			else if(config.i_J < 1)
				config.i_J = 1;
			config.Ji = xf->r[config.i_J-1];     
			config.Jvi = xf->dr[config.i_J-1];     
#ifdef USE_GN
     mbsopt->Reset();
#endif
		}
		else
		{
			if(mbstype != "FIXEDBASE")
			{
				// overwrite the config with values from initial state
				config.vroll = mbsopt->xs[0].vs[0](0);
				config.vpitch = mbsopt->xs[0].vs[0](1);
				config.vyaw = mbsopt->xs[0].vs[0](2);
				config.vx = mbsopt->xs[0].vs[0](3);
				config.vy = mbsopt->xs[0].vs[0](4);
				config.vz = mbsopt->xs[0].vs[0](5);

				gcop::SE3::Instance().g2rpyxyz(rpy,xyz,mbsopt->xs[0].gs[0]*gposeroot_i);
				config.roll = rpy(0);
				config.pitch = rpy(1);
				config.yaw = rpy(2);
				config.x = xyz(0);
				config.y = xyz(1);
				config.z = xyz(2);
			}

			if(config.i_J > nb-1)
				config.i_J = nb-1; 
			else if(config.i_J < 1)
				config.i_J = 1;

			config.Ji = mbsopt->xs[0].r[config.i_J-1];     
			config.Jvi = mbsopt->xs[0].dr[config.i_J-1];     
		}
#ifndef USE_GN
		config.mu = mbsopt->mu;
#endif
		config.tf = cost->tf;
		if(config.iterate)
		{
			config.iterate = false;
			ros::TimerEvent timerevent;
			iterateCallback(timerevent);
		}
		if(config.animate)
		{
			config.animate = false;
			pubtraj();
		}
		return;
	}

	cout<<"Config.tf: "<<config.tf<<endl;

	if(config.final)
	{

		if(mbstype != "FIXEDBASE")
		{
			gcop::SE3::Instance().rpyxyz2g(xf->gs[0], Vector3d(config.roll,config.pitch,config.yaw), Vector3d(config.x,config.y,config.z));
      xf->gs[0] = xf->gs[0]*gposei_root;
			xf->vs[0]<<config.vroll, config.vpitch, config.vyaw, config.vx, config.vy, config.vz;
		}

		if(config.i_J > nb-1)
			config.i_J = nb-1; 
		else if(config.i_J < 1)
			config.i_J = 1;

		xf->r[config.i_J-1] = config.Ji;
		xf->dr[config.i_J-1] = config.Jvi;
	}
	else
	{
		if(mbstype != "FIXEDBASE")
		{
			gcop::SE3::Instance().rpyxyz2g(mbsopt->xs[0].gs[0], Vector3d(config.roll,config.pitch,config.yaw), Vector3d(config.x,config.y,config.z));
      mbsopt->xs[0].gs[0] = mbsopt->xs[0].gs[0]*gposei_root;
			mbsopt->xs[0].vs[0]<<config.vroll, config.vpitch, config.vyaw, config.vx, config.vy, config.vz;
		}

		if(config.i_J > nb-1)
			config.i_J = nb-1; 
		else if(config.i_J <= 0)
			config.i_J = 1;

		mbsopt->xs[0].r[config.i_J -1] = config.Ji;
		mbsopt->xs[0].dr[config.i_J -1] = config.Jvi;
		mbsmodel->Rec(mbsopt->xs[0], h);
	}
	if(config.ureset)
  {
    //#TODO Use Biases to reset controls
    //cout<<"Hello"<<endl;
    VectorXd u(mbsmodel->U.n);
    u.setZero();
    if(mbstype == "AIRBASE")
    {
      for(int count = 0;count < nb;count++)
        u[3] += (mbsmodel->links[count].m)*(-mbsmodel->ag(2));
      cout<<"u[3]: "<<u[3]<<endl;
    }
    else if(mbstype == "FLOATBASE")
    {
      for(int count = 0;count < nb;count++)
        u[5] += (mbsmodel->links[count].m)*(-mbsmodel->ag(2));
      cout<<"u[5]: "<<u[5]<<endl;
    }
    int size = mbsopt->us.size();
    double t1;

    for(int count = 0;count < size;count++)
    {
      mbsopt->us[count] = u;
      mbsmodel->Step(mbsopt->xs[count+1], count*h, mbsopt->xs[count], mbsopt->us[count], h);
    }
  }


	if(config.i_Q > mbsmodel->X.n)
		config.i_Q = mbsmodel->X.n; 
	else if(config.i_J <= 0)
		config.i_Q = 1;

	cost->Qf(config.i_Q -1,config.i_Q -1) = config.Qfi;
	cost->Q(config.i_Q -1,config.i_Q -1) = config.Qi;

	if(config.i_R > mbsmodel->U.n)
		config.i_R = mbsmodel->U.n; 
	else if(config.i_R <= 0)
		config.i_R = 1;

	cost->R(config.i_R -1,config.i_R -1) = config.Ri;

  //Setting Values
	for (int k = 0; k <=N; ++k)
	mbsopt->ts[k] = k*h;

	cost->tf = config.tf;
	
}


int main(int argc, char** argv)
{
	ros::init(argc, argv, "chainload");
	ros::NodeHandle n("mbsddp");
	//Initialize publisher
	trajpub = n.advertise<gcop_comm::CtrlTraj>("ctrltraj",1);
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
  if(n.hasParam("p0"))
    mbsmodel = gcop_urdf::mbsgenerator(xml_string, mbstype, 6);
  else
    mbsmodel = gcop_urdf::mbsgenerator(xml_string, mbstype);

	gposei_root = mbsmodel->pose_inertia_base;
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


	//Using it:
	//define parameters for the system
	int nb = mbsmodel->nb;
	double tf = 20;   // time-horizon

	n.getParam("tf",tf);
	n.getParam("N",N);

	double h = tf/N; // time-step



	//times
	vector<double> ts(N+1);
	for (int k = 0; k <=N; ++k)
		ts[k] = k*h;


	//Define Final State
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
	xf->gs[0] = xf->gs[0]*gposei_root;//new stuff with transformations
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
  cost->UpdateGains();
	//

  //Define external parameters:
  VectorXd p0(6);///< Initial Guess for external forces on end effector
  p0<<0,0,0,0,0,0;//Initialization
  XmlRpc::XmlRpcValue param_list;
  if(n.getParam("p0", param_list))
  {
		xml2vec(p0,param_list);
		ROS_ASSERT(p0.size() == 6);
		cout<<"parameter: "<<endl<<p0<<endl;
  }

  //If external parameter is provided, end effector frame name should also be provided
  if(n.hasParam("frame_name"))
  {
    n.getParam("frame_name",(mbsmodel->end_effector_name));
    ROS_INFO("End effector name: %s",(mbsmodel->end_effector_name.c_str()));
  }

  if((n.hasParam("p0") ^ n.hasParam("frame_name"))!=0)
  {
    ROS_WARN("Provided external parameter but no end effector name parameter: frame_name provided");
  }


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
	x0.gs[0] = x0.gs[0]*gposei_root;//new stuff with transformations
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
  if(n.hasParam("p0"))
    mbsmodel->Bias(forces,0,x0, &p0);
  else
    mbsmodel->Bias(forces,0,x0);
  //Base Controls are to be given in base frame:
  Matrix6d M_base_inertia;
  gcop::SE3::Instance().Ad(M_base_inertia, gposeroot_i);
  forces.head<6>() = M_base_inertia.transpose()*forces.head<6>();
	cout<<"Bias computed: "<<forces.transpose()<<endl;

	//Set Controls to cancel the forces:
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
	cout<<"Nb: "<<nb<<endl;

	//Joint Torques:
	u.tail(nb-1) = forces.tail(nb-1);

	vector<VectorXd> us(N,u);
	vector<MbsState> xs(N+1,x0);

#ifdef USE_GN
  int Nk = 10;//Number of spline segments.
  n.getParam("Nk",Nk);
  //VectorXd tks(Nk+1);
  vector<double> tks(Nk+1);
  for(int k = 0; k <= Nk; ++k)
  {
    tks[k] = k*(tf/Nk);
  }
  int degree = 2;
  n.getParam("degree", degree);
  ControlTparam<MbsState> tp(*mbsmodel, tks);//Spline degree
#endif

  if(n.hasParam("p0"))
  {
#ifdef USE_GN
    mbsopt.reset(new MbsGn(*mbsmodel, *cost, tp, ts, xs, us, &p0));
#else
    mbsopt.reset(new MbsDdp(*mbsmodel, *cost, ts, xs, us, &p0));
#endif
  }
  else
  {
#ifdef USE_GN
    mbsopt.reset(new MbsGn(*mbsmodel, *cost, tp, ts, xs, us));
#else
    mbsopt.reset(new MbsDdp(*mbsmodel, *cost, ts, xs, us));
#endif
  }

#ifndef USE_GN
	mbsopt->mu = 10.0;
	n.getParam("mu",mbsopt->mu);
#else
  mbsopt->numdiff_stepsize = 1e-16;
	n.getParam("stepsize",(mbsopt->numdiff_stepsize));
  cout<<"step size: "<<(mbsopt->numdiff_stepsize)<<endl;
#endif

	//Trajectory message initialization
	trajectory.N = N;
	trajectory.statemsg.resize(N+1);
	trajectory.ctrl.resize(N);
	trajectory.time = ts;
	trajectory.rootname = mbsmodel->links[0].name;

	trajectory.finalgoal.statevector.resize(nb-1);
	trajectory.finalgoal.names.resize(nb-1);

	trajectory.statemsg[0].statevector.resize(nb-1);
	trajectory.statemsg[0].names.resize(nb-1);

	for (int i = 0; i < N; ++i) 
	{
		trajectory.statemsg[i+1].statevector.resize(nb-1);
		trajectory.statemsg[i+1].names.resize(nb-1);
		trajectory.ctrl[i].ctrlvec.resize(mbsmodel->U.n);
	}
	//Debug true for mbs

	//mbsmodel->debug = true;


	// Get number of iterations
	n.getParam("Nit",Nit);
	// Create timer for iterating	
	iteratetimer = n.createTimer(ros::Duration(0.1), iterateCallback);
	string mode = "continuous";//or user
	n.getParam("mode",mode);//otherwise user based
	cout<<"mode: "<<mode<<endl;
	if(mode == "continuous")
		iteratetimer.start();
	else
		iteratetimer.stop();
	//	Dynamic Reconfigure setup Callback ! immediately gets called with default values	
	dynamic_reconfigure::Server<gcop_ctrl::MbsDMocInterfaceConfig> server;
	dynamic_reconfigure::Server<gcop_ctrl::MbsDMocInterfaceConfig>::CallbackType f;
	f = boost::bind(&paramreqcallback, _1, _2);
	server.setCallback(f);
	ros::spin();
	return 0;
}


	




