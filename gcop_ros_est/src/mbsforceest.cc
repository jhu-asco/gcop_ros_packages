/** This is an example on simulating a Multi body system with given controls. There is a reconfiguration interface for changing the controls and is 
  * used primarily for checking if the created MBS system from urdf file is working properly
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
#include <gcop/mbscontroller.h>
#include <gcop/lqsensorcost.h>
#include <gcop/sensor.h>
//#include "gcop_ctrl/MbsSimInterfaceConfig.h"
#include <tf/transform_listener.h>
#include <geometry_msgs/WrenchStamped.h>
#include <XmlRpcValue.h>

#ifdef USE_STOCHASTIC_DYNAMICS
#include <gcop/gndoep.h>
#else
#include <gcop/deterministicgndoep.h>
#endif


using namespace std;
using namespace Eigen;
using namespace gcop;

/*
 * Procedure: 
 * 1. Add Force to end effector in mbs.h
 * 2. Setup costs etc here
 * 3. Only add iterate and ability to change desired parameters(Forces) to reconfigure for now 
 * 4. Visualize force and estimated force in Rviz
 * Make one more variation where body3d problem is created from a mbs system's mass matrix at a given instance of time
 * But that is I think the same as a regular body3dsystem problem
 */


//ros messages
gcop_comm::CtrlTraj trajectory;

//Publisher
ros::Publisher trajpub;
ros::Publisher wrenchpub;
ros::Publisher desiredwrenchpub;

//Timer
ros::Timer iteratetimer;

//Subscriber
//ros::Subscriber initialposn_sub;

//Pointer for mbs system
boost::shared_ptr<Mbs> mbsmodel;

//Pointer to lqsensor cost
boost::shared_ptr<LqSensorCost<MbsState, Dynamic, Dynamic, Dynamic, Dynamic, MbsState, Dynamic> > cost;

//Pointer to sensor
boost::shared_ptr<Sensor<MbsState> > sensor;


//States and controls for system
vector<VectorXd> us;
vector<MbsState> xs;
vector<double> ts;

string mbstype; // Type of system
double tfinal = 20;   // time-horizon
Matrix4d gposeroot_i; //inital inertial frame wrt the joint frame

string endeffector_framename;//Frame name of the end effector

VectorXd p0(6);///< Initial Guess for external forces
VectorXd pd(6);///< True External forces

void rpy2transform(geometry_msgs::Transform &transformmsg, Vector6d &bpose)
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
	//cout<<my_list.size()<<endl;
	//ROS_ASSERT(vec.size() <= my_list.size()); //Desired size

	for (int32_t i = 0; i < my_list.size(); i++) 
	{
				ROS_ASSERT(my_list[i].getType() == XmlRpc::XmlRpcValue::TypeDouble);
				cout<<"my_list["<<i<<"]\t"<<my_list[i]<<endl;
			  vec[i] =  (double)(my_list[i]);
	}
}

void publishtraj()
{
	int N = us.size();
	double h = tfinal/N; // time-step
	int csize = mbsmodel->U.n;
	int nb = mbsmodel->nb;
	Vector6d bpose;
	gcop::SE3::Instance().g2q(bpose,gposeroot_i*xs[0].gs[0]);
	rpy2transform(trajectory.statemsg[0].basepose,bpose);
	for (int i = 0; i < N; ++i) 
  {
    //Fill trajectory message:
    gcop::SE3::Instance().g2q(bpose, gposeroot_i*xs[i+1].gs[0]);
    rpy2transform(trajectory.statemsg[i+1].basepose,bpose);
    for(int count1 = 0;count1 < nb-1;count1++)
    {
      trajectory.statemsg[i+1].statevector[count1] = xs[i+1].r[count1];
      trajectory.statemsg[i+1].names[count1] = mbsmodel->joints[count1].name;
    }
    for(int count1 = 0;count1 < csize;count1++)
    {
      trajectory.ctrl[i].ctrlvec[count1] = us[i](count1);
    }
  }
	trajectory.time = ts;

	trajpub.publish(trajectory);
  //Also publish the desired and current wrenches:
  geometry_msgs::WrenchStamped wrench_msg;
  wrench_msg.header.stamp = ros::Time::now();
  wrench_msg.header.frame_id = endeffector_framename;
  //Current_wrench:
  wrench_msg.wrench.torque.x = p0[0];
  wrench_msg.wrench.torque.y = p0[1];
  wrench_msg.wrench.torque.z = p0[2];
  wrench_msg.wrench.force.x  = p0[3];
  wrench_msg.wrench.force.y  = p0[4];
  wrench_msg.wrench.force.z  = p0[5];
  wrenchpub.publish(wrench_msg);
  //Desired wrench:
  wrench_msg.wrench.torque.x = pd[0];
  wrench_msg.wrench.torque.y = pd[1];
  wrench_msg.wrench.torque.z = pd[2];
  wrench_msg.wrench.force.x  = pd[3];
  wrench_msg.wrench.force.y  = pd[4];
  wrench_msg.wrench.force.z  = pd[5];
  desiredwrenchpub.publish(wrench_msg);
}

void simtraj(vector<MbsState> &zs, vector<double> &ts_sensor, VectorXd &p) //N is the number of segments
{
  //Add Zs and ts_sensor support: #TODO
	cout<<"Sim Traj called"<<endl;
	int N = us.size();
	double h = tfinal/N; // time-step
	//cout<<"N: "<<N<<endl;
	int csize = mbsmodel->U.n;
	//cout<<"csize: "<<csize<<endl;
	int nb = mbsmodel->nb;
	//cout<<"nb: "<<nb<<endl;
	Vector6d bpose;

  int sensor_index = 0;

	gcop::SE3::Instance().g2q(bpose,gposeroot_i*xs[0].gs[0]);
	rpy2transform(trajectory.statemsg[0].basepose,bpose);

	for(int count1 = 0;count1 < nb-1;count1++)
	{
		trajectory.statemsg[0].statevector[count1] = xs[0].r[count1];
		trajectory.statemsg[0].names[count1] = mbsmodel->joints[count1].name;
	}

	for (int i = 0; i < N; ++i) 
	{
//		cout<<"i "<<i<<endl;
		mbsmodel->Step(xs[i+1], i*h, xs[i], us[i], h, &p);

    //Fill Sensor output:
    if((ts_sensor[sensor_index] - ts[i])>= 0 && (ts_sensor[sensor_index] - ts[i+1]) < 0)
    {
      int near_index = (ts_sensor[sensor_index] - ts[i]) > -(ts_sensor[sensor_index] - ts[i+1])?(i+1):i;
      (*sensor)(zs[sensor_index], ts[near_index], xs[near_index], us[near_index]);
      sensor_index = sensor_index < (ts_sensor.size()-1)?sensor_index+1:sensor_index;
      //Print pose of zs to see if something meaningful is done:
      gcop::SE3::Instance().g2q(bpose, gposeroot_i*xs[i+1].gs[0]);
      cout<<"zs["<<sensor_index<<"]: "<<bpose.transpose()<<endl;
    }

    //Fill trajectory message:
		gcop::SE3::Instance().g2q(bpose, gposeroot_i*xs[i+1].gs[0]);
		rpy2transform(trajectory.statemsg[i+1].basepose,bpose);
		for(int count1 = 0;count1 < nb-1;count1++)
		{
			trajectory.statemsg[i+1].statevector[count1] = xs[i+1].r[count1];
			trajectory.statemsg[i+1].names[count1] = mbsmodel->joints[count1].name;
		}
		for(int count1 = 0;count1 < csize;count1++)
		{
			trajectory.ctrl[i].ctrlvec[count1] = us[i](count1);
		}
	}

	trajectory.time = ts;

	trajpub.publish(trajectory);

}


/*void paramreqcallback(gcop_ctrl::MbsSimInterfaceConfig &config, uint32_t level) 
{
	int nb = mbsmodel->nb;

	if(level == 0xffffffff)
		config.tf = tfinal;

	tfinal = config.tf;

	if(level & 0x00000001)
	{

		if(config.i_J > nb-1)
			config.i_J = nb-1; 
		else if(config.i_J < 1)
			config.i_J = 1;

		config.Ji = xs[0].r[config.i_J-1];     
		config.Jvi = xs[0].dr[config.i_J-1];     

		if(config.i_u > mbsmodel->U.n)
			config.i_u = mbsmodel->U.n;
		else if(config.i_u < 1)
			config.i_u = 1;
		config.ui = us[0][config.i_u-1];
	}

	int N = us.size();
	double h = tfinal/N;
	//Setting Values
	for (int k = 0; k <=N; ++k)
		ts[k] = k*h;

  if((mbstype == "FLOATBASE")||(mbstype == "AIRBASE"))
	{
		gcop::SE3::Instance().rpyxyz2g(xs[0].gs[0], Vector3d(config.roll,config.pitch,config.yaw), Vector3d(config.x,config.y,config.z));
		xs[0].vs[0]<<config.vroll, config.vpitch, config.vyaw, config.vx, config.vy, config.vz;
	}

	xs[0].r[config.i_J -1] = config.Ji;
	xs[0].dr[config.i_J -1] = config.Jvi;
	mbsmodel->Rec(xs[0], h);

	for(int count = 0;count <us.size();count++)
		us[count][config.i_u-1] = config.ui; 

	return;
}
*/


int main(int argc, char** argv)
{
	ros::init(argc, argv, "chainload");
	ros::NodeHandle n("mbsdoep");
	//Initialize publisher
	trajpub = n.advertise<gcop_comm::CtrlTraj>("ctrltraj",1);
  wrenchpub = n.advertise<geometry_msgs::WrenchStamped>("current_wrench",1);
  desiredwrenchpub = n.advertise<geometry_msgs::WrenchStamped>("desired_wrench",1);
	//get parameter for xml_string:
	string xml_string, xml_filename;
	if(!ros::param::get("/robot_description", xml_string))
	{
		ROS_ERROR("Could not fetch xml file name");
		return 0;
	}
  //Get End effector name:
  n.getParam("frame_name",endeffector_framename);
  //Get Base type:
	n.getParam("basetype",mbstype);
	assert((mbstype == "FIXEDBASE")||(mbstype == "AIRBASE")||(mbstype == "FLOATBASE"));
	VectorXd xmlconversion;
	Matrix4d gposei_root; //inital inertial frame wrt the joint frame
	//Create Mbs system
	mbsmodel = gcop_urdf::mbsgenerator(xml_string,gposei_root, mbstype, 6);
	mbsmodel->U.bnd = false;
	gcop::SE3::Instance().inv(gposeroot_i,gposei_root);
	cout<<"Mbstype: "<<mbstype<<endl;
	mbsmodel->ag << 0, 0, -0.05;
  //get ag from parameters
	XmlRpc::XmlRpcValue ag_list;
	if(n.getParam("ag", ag_list))
		xml2vec(xmlconversion,ag_list);
	ROS_ASSERT(xmlconversion.size() == 3);
	//mbsmodel->ag = gposei_root.topLeftCorner(3,3).transpose()*xmlconversion.head(3);
	mbsmodel->ag = xmlconversion.head(3);

	cout<<"mbsmodel->ag: "<<endl<<mbsmodel->ag<<endl;
	//mbsmodel->ag = xmlconversion.head(3);

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
		(mbstype == "FLOATBASE")? 6:
		(mbstype == "FIXEDBASE")? 0:0;
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

  //Create Sensor
  sensor.reset(new Sensor<MbsState>(mbsmodel->X));//Create a default sensor which just copies the MbsState over

  //Create Sensor Cost:
  //<Tx, nx, nu, nres, np, Tz, nz>
  cost.reset(new LqSensorCost<MbsState, Dynamic, Dynamic, Dynamic, Dynamic, MbsState, Dynamic>(*mbsmodel, sensor->Z));

  //Get parameters for cost:
  {
    cost->R.setIdentity();

    //list of Sensor out cost :
    XmlRpc::XmlRpcValue sensorcost_list;
    if(n.getParam("R", sensorcost_list))
    {
      cout<<(sensor->Z.n)<<endl;
      xml2vec(xmlconversion,sensorcost_list);
      cout<<"conversion"<<endl<<xmlconversion<<endl;
      ROS_ASSERT(xmlconversion.size() == sensor->Z.n);
      cost->R = xmlconversion.asDiagonal();
      cout<<"Cost.R"<<endl<<cost->R<<endl;
    }
    // Noise cost list:
    XmlRpc::XmlRpcValue noisecost_list;
    if(n.getParam("S", noisecost_list))
    {
      cout<<mbsmodel->X.n<<endl;
      xml2vec(xmlconversion,noisecost_list);
      cout<<"conversion"<<endl<<xmlconversion<<endl;
      ROS_ASSERT(xmlconversion.size() == mbsmodel->X.n);
      cost->S = xmlconversion.asDiagonal();
      cout<<"Cost.S"<<endl<<cost->S<<endl;
    }
    else
    {
      cout<<"Mbsmodel->X.n"<<mbsmodel->X.n<<endl;
      xmlconversion.resize(mbsmodel->X.n);
      xmlconversion.setConstant(1000);
      cost->S = xmlconversion.asDiagonal();
      cout<<"Cost.S"<<endl<<cost->S<<endl;
    }

    //Parameter cost list:
    XmlRpc::XmlRpcValue paramcost_list;
    if(n.getParam("P", paramcost_list))
    {
      cout<<mbsmodel->P.n<<endl;
      xml2vec(xmlconversion,paramcost_list);
      ROS_ASSERT(xmlconversion.size() == mbsmodel->P.n);
      cout<<"conversion"<<endl<<xmlconversion<<endl;
      cost->P = xmlconversion.asDiagonal();
    }
    cout<<"Cost.P"<<endl<<cost->P<<endl;

  }
  //Update Gains after entering the costs above
  cost->UpdateGains();

  VectorXd mup(6);//Initial Prior for parameters

  p0<<0,0,0,0,0,-0.05;//Initialization
  pd<<0,0,0.05,0,0.1,-0.1;//Initialization
  {
    XmlRpc::XmlRpcValue p_list;
    if(n.getParam("p0", p_list))
      xml2vec(p0,p_list);
    ROS_ASSERT(p0.size() == 6);

    if(n.getParam("pd", p_list))
      xml2vec(pd,p_list);
    ROS_ASSERT(pd.size() == 6);
  }
  mup = p0;//Copy the initial guess to be the same as prior for the parameters


	//Using it:
	//define parameters for the system
	int nb = mbsmodel->nb;

	//times
	int N = 100;      // discrete trajectory segments
	n.getParam("tf",tfinal);
	n.getParam("N",N);
	double h = tfinal/N; // time-step

	//Define the initial state mbs
	MbsState x(nb,(mbstype == "FIXEDBASE"));
	x.gs[0].setIdentity();
	x.vs[0].setZero();
	x.dr.setZero();
	x.r.setZero();

	// Get X0	 from params
	XmlRpc::XmlRpcValue x0_list;
	if(n.getParam("X0", x0_list))
	{
		xml2vec(xmlconversion,x0_list);
		ROS_ASSERT(xmlconversion.size() == 12);
		x.vs[0] = xmlconversion.tail<6>();
		gcop::SE3::Instance().rpyxyz2g(x.gs[0],xmlconversion.head<3>(),xmlconversion.segment<3>(3)); 
	}
	x.gs[0] = gposei_root*x.gs[0];//new stuff with transformations
	cout<<"x.gs[0]"<<endl<<x.gs[0]<<endl;
  //list of joint angles:
	XmlRpc::XmlRpcValue j_list;
	if(n.getParam("J0", j_list))
	{
		xml2vec(x.r,j_list);
	}
	cout<<"x.r"<<endl<<x.r<<endl;

	if(n.getParam("Jv0", j_list))
	{
		xml2vec(x.dr,j_list);
		cout<<"x.dr"<<endl<<x.dr<<endl;
	}
	mbsmodel->Rec(x, h);

	// initial controls (e.g. hover at one place)
	VectorXd u(mbsmodel->U.n);
	u.setZero();
	
	cout<<"Finding Biases"<<endl;
	int n11 = mbsmodel->nb -1 + 6*(!mbsmodel->fixed);
	VectorXd forces(n11);
	mbsmodel->Bias(forces,0,x, &pd);
	cout<<"Bias computed: "<<forces.transpose()<<endl;

	//forces = -1*forces;//Controls should be negative of the forces


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
	//Joint Torques:
	u.tail(nb-1) = forces.tail(nb-1);



	//Create states and controls
	xs.resize(N+1,x);
	us.resize(N,u);

	ts.resize(N+1);
	for (int k = 0; k <=N; ++k)
		ts[k] = k*h;


  //Create Sensor:
  //Sensor
  vector<MbsState> zs(N/2);//Same as ts_sensor
  vector<double> ts_sensor(N/2);

  //Set sensor times:
  for (int k = 0; k <(N/2); ++k)
    ts_sensor[k] = 2*k*h;
 
	//Trajectory message initialization
	trajectory.N = N;
	trajectory.statemsg.resize(N+1);
	trajectory.ctrl.resize(N);
	trajectory.time = ts;
	trajectory.rootname = mbsmodel->links[0].name;

	trajectory.statemsg[0].statevector.resize(nb-1);
	trajectory.statemsg[0].names.resize(nb-1);

	for (int i = 0; i < N; ++i) 
	{
		trajectory.statemsg[i+1].statevector.resize(nb-1);
		trajectory.statemsg[i+1].names.resize(nb-1);
		trajectory.ctrl[i].ctrlvec.resize(mbsmodel->U.n);
	}
  ROS_INFO("Press any key to continue...");
  getchar();
  ROS_INFO("Desired trajectory");
  simtraj(zs, ts_sensor, pd);//Fills zs with the right sensor data and publishes a trajectory
  ros::spinOnce();
  cost->SetReference(&zs, &mup);//Set reference for zs
  getchar();
  //Create GN Optimal estimation class:	
  //<Tx, nx, nu, np, nres, Tz, nz, T1=T, nx1=nx>
  GnDoep<MbsState, Dynamic, Dynamic, Dynamic, Dynamic, MbsState, Dynamic, MbsState, Dynamic> gn(*mbsmodel, *sensor, *cost, ts, xs, us, p0, ts_sensor);  
  gn.debug = false;
  //Publish the initial guessed trajectory:
  ROS_INFO("Intial guess trajectory");
  publishtraj();
  ros::spinOnce();
  getchar();
  while(ros::ok())
  {
    cout<<"Iterating..."<<endl;
    gn.Iterate();
    cout<<"Done Iterating"<<endl;
    cout<<"Cost: "<<(gn.J)<<endl;
    cout << "Parameter: "<< p0 << endl;
    publishtraj();
    ros::spinOnce();
    char c;
    cin>>c;
    while (c == 'a')
    {
      publishtraj();
      ros::spinOnce();
      cin>>c;
    }
  }
	return 0;
}


	




