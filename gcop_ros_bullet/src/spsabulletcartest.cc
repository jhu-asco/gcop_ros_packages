/** This example computes optimal trajectories for a rccar using Cross Entropy based sampling method. 
 * Gcop library is used for creating rccar system and algorithm etc.
 * Rviz is used for display
 */
//System stuff
#include <iostream>

//Gcop Stuff
#include <gcop/spsa.h>
#include <gcop/controltparam.h>
#include <gcop/rnlqcost.h>
#include <gcop/bulletrccar.h>
#include <gcop/bulletworld.h>
//#include "utils.h"
//#include "controltparam.h"

// Ros stuff
#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <gcop_ros_bullet/CEInterfaceConfig.h>
#include <visualization_msgs/Marker.h>
#include <sensor_msgs/JointState.h>
#include <XmlRpcValue.h>
#include <tf/transform_broadcaster.h>
#include <std_msgs/Float64MultiArray.h>
#include "gcop_comm/CtrlTraj.h"//msg for publishing ctrl trajectory


using namespace std;
using namespace Eigen;
using namespace gcop;

typedef SPSA<Vector4d, 4, 2> RccarSPSA;

//Global variable:
boost::shared_ptr<RccarSPSA> spsa;///<Cross entropy based solver
boost::shared_ptr<Bulletrccar> sys;///Bullet rccar system
boost::shared_ptr<BaseSystem> base_sys;///Bullet rccar system
vector<Vector4d> xs;///< State trajectory of the system
vector<Vector2d> us;///< Controls for the trajectory of the system
vector<double> ts;///< Times for trajectory
vector<double> zs;///< Height of the car along the trajectory (same as xs) only used for visualization
bool sendtrajectory;///< Send the gcop trajectory 
int Nreq;///< Number of segments requested for gcop trajectory

//ros publisher and subscribers:
ros::Publisher joint_pub;///<Rccar model joint publisher for animation
ros::Publisher traj_pub;///<Rviz best trajectory publisher
ros::Publisher sampletraj_pub;///<Rviz sample trajectories publisher
ros::Publisher gcoptraj_pub;///<Publisher for gcop trajectory (States, times, Controls)

//Messages:
visualization_msgs::Marker line_strip;///<Best trajectory message
visualization_msgs::Marker sampleline_strip;///<Sample trajectory message
sensor_msgs::JointState joint_state;///< Joint states for wheels in animation
gcop_comm::CtrlTraj trajectory;

//tf
tf::TransformBroadcaster *broadcaster;

/** Helper function for converting ros vector list into eigen vector
 * @param vec  output from my_list
 * @param my_list input ros list obtained from nh.getParam
 *
 *  The list should be an array and each value should be a doule
 */
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

/**  Interface for configuring /iterating through the solver
 * @param config Configuration class autogenerated from dynamic reconfiguration
 * @param level  Each change in reconfiguration is added to level through a bitwise and operation
 */
void ParamreqCallback(gcop_ros_bullet::CEInterfaceConfig &config, uint32_t level) 
{
  if(config.iterate)
  {
    ros::Time currtime;
    cout<<"Iterating: "<<endl;
    spsa->Nit = config.Nit;
    spsa->stepc.A = 0.5*(spsa->Nit);//50 percent of total number of iterations
    for (int i = 0; i < config.Nit; ++i) {
      currtime = ros::Time::now();
      spsa->Iterate();
      cout << "Iteration #" << i << " took: " << (ros::Time::now() - currtime).toSec()*1e3 << " ms." << endl;
      //cout << "Cost=" << spsa->J << endl;#TODO Add this to docp very important
      //cout<<"xsN: "<<xs.back().transpose()<<endl;

      //Publish rviz Trajectory for visualization:
      line_strip.header.stamp  = ros::Time::now();

      for(int i =0;i<xs.size(); i++)
      {
        //geometry_msgs::Point p;
        line_strip.points[i].x = xs[i][0];
        line_strip.points[i].y = xs[i][1];
        line_strip.points[i].z = zs[i];//Need to add  this to state or somehow get it #TODO
      }
      traj_pub.publish(line_strip);
    }
    
    for(int i =0;i < us.size();i++)
    {
      cout<<"us["<<i<<"]: "<<us[i].transpose()<<endl;
      cout<<"xs["<<i+1<<"]: "<<xs[i+1].transpose()<<endl;
    }//#DEBUG

    //Publish control trajectory when parameter is set:
    if(sendtrajectory)
    {
      for (int count = 0;count<Nreq;count++)
      {
        for(int count1 = 0;count1 < 2;count1++)
        {
          trajectory.ctrl[count].ctrlvec[count1] = us[count](count1);
        }
      }
      gcoptraj_pub.publish(trajectory);
    }

    config.iterate = false;
  }

  if(config.animate)
  {
    //Run the system:
    sys->Reset(xs[0],ts[0]);
    for(int count1 = 0;count1 < us.size();count1++)
    {
      base_sys->Step(us[count1], ts[count1+1]-ts[count1]);
      //Set the car joint stuff:
      joint_state.header.stamp = ros::Time::now();
      //Back wheel
			joint_state.position[2] = 0;
			//steering angle
			joint_state.position[0] = sys->gVehicleSteering;
			joint_state.position[1] = sys->gVehicleSteering;
			//send joint state
			joint_pub.publish(joint_state);
      //Set the pose for the car:
      btTransform chassistransform = sys->m_carChassis->getWorldTransform();
      btQuaternion chassisquat = chassistransform.getRotation();
      btVector3 chassisorig = chassistransform.getOrigin();
      tf::Transform tf_chassistransform;
      tf_chassistransform.setRotation(tf::Quaternion(chassisquat.x(), chassisquat.y(), chassisquat.z(), chassisquat.w()));
      tf_chassistransform.setOrigin(tf::Vector3(chassisorig.x(), chassisorig.y(), chassisorig.z()));
      broadcaster->sendTransform(tf::StampedTransform(tf_chassistransform, ros::Time::now(), "world", "base_link"));
      cout<<"spsa->sys.x" <<ts[count1+1]<<"\txs: "<<(sys->x).transpose()<<endl;//#DEBUG
      usleep((ts[count1+1] - ts[count1])*1e6);//microseconds
    }
      config.animate = false;
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "spsacartest");

  ros::NodeHandle nh("~");

  //setup tf
	broadcaster = new tf::TransformBroadcaster();

  //Setup publishers and subscribers:
	traj_pub = nh.advertise<visualization_msgs::Marker>("best_traj", 1);
	sampletraj_pub = nh.advertise<visualization_msgs::Marker>("sample_traj", 100);
	joint_pub = nh.advertise<sensor_msgs::JointState>("/joint_states", 1);
  gcoptraj_pub = nh.advertise<gcop_comm::CtrlTraj>("ctrltraj",1);

  /* Parametersf for solver */
  int N = 32;        // number of segments
  double tf = 5;    // time horizon

  nh.getParam("N",N);
  ROS_ASSERT(2*int(N/2) == N);//May not be really necessary #CHECK
  nh.getParam("tf",tf);
  cout<<"N: "<<N<<endl;
  cout<<"tf: "<<tf<<endl;

  double h = tf/N;   // time step

  //Create Bullet world and rccar system:
  BulletWorld world(true);//Set the up axis as z for this world

  zs.resize(N+1);//<Resize the height vector and pass it to the rccar system
  sys.reset(new Bulletrccar(world, &zs));
  base_sys = boost::static_pointer_cast<BaseSystem>(sys);
  sys->initialz = 0.12;
  sys->gain_cmdvelocity = 1;
  sys->kp_steer = 0.0125;
  sys->kp_torque = 15;
  sys->steeringClamp = 15.0*(M_PI/180.0);
  sys->U.lb[0] = -(sys->steeringClamp);
  sys->U.ub[1] = (sys->steeringClamp);
  sys->U.bnd = true;

  sys->offsettrans.setIdentity();
  sys->offsettransinv.setIdentity();

  //Load Ground
  {
    std::string filename;
    nh.getParam("mesh",filename);
    cout<<"Filename: "<<filename<<endl;
    btCollisionShape *groundShape;
    if(filename.compare("plane") == 0)
      groundShape = world.CreateGroundPlane(20,20);
    else
      groundShape= world.CreateMeshFromSTL(filename.c_str());//20 by 20 long plane

    btTransform tr;
    tr.setOrigin(btVector3(0, 0, 0));
    tr.setRotation(btQuaternion(0,0,0));
    world.LocalCreateRigidBody(0,tr, groundShape);
  }

  // initial state
  Vector4d x0(1,1,0,0);

  // final state
  Vector4d xf(0,0,0,0);

  // cost
  RnLqCost<4, 2,Dynamic, 6> cost(*sys, tf, xf);

  {
    VectorXd temp;
    XmlRpc::XmlRpcValue list;

    //Initial state
    if(nh.getParam("x0", list))
      xml2vec(temp,list);
    ROS_ASSERT(temp.size() == 4);

    x0 = temp;

    if(nh.getParam("xf", list))
      xml2vec(temp,list);
    ROS_ASSERT(temp.size() == 4);

    xf = temp;


    if(nh.getParam("Q", list))
      xml2vec(temp,list);
    ROS_ASSERT(temp.size() == 4);

    cost.Q = temp.asDiagonal();

    if(nh.getParam("Qf", list))
      xml2vec(temp,list);
    ROS_ASSERT(temp.size() == 4);

    cost.Qf = temp.asDiagonal();

    if(nh.getParam("R", list))
      xml2vec(temp,list);
    ROS_ASSERT(temp.size() == 2);

    cost.R = temp.asDiagonal();

    cost.UpdateGains();//#TODO Make this somehow implicit in the cost function otherwise becomes a coder's burden !

    cout<<"x0: "<<x0.transpose()<<endl;
    cout<<"xf: "<<xf.transpose()<<endl;
    cout<<"Q: "<<endl<<cost.Q<<endl;
    cout<<"Qf: "<<endl<<cost.Qf<<endl;
    cout<<"R: "<<endl<<cost.R<<endl;
  }

  // times
  ts.resize(N+1);
  for (int k = 0; k <=N; ++k)
    ts[k] = k*h;

  // states
  xs.resize(N+1);
  // initial state

  // initial controls [ If more complicated control is needed hardcode them here]
  us.resize(N);

  for (int i = 0; i < N/2; ++i) {
    us[i] = Vector2d(0.5, 0);
    us[N/2+i] = Vector2d(0.5, 0);    
  }
  //Set initial state:
  xs[0] = x0;

 /* int Nk = 10;
  nh.getParam("Nk", Nk);
  assert(Nk > 0);

  vector<double> tks(Nk+1);
  for (int k = 0; k <=Nk; ++k)
    tks[k] = k*(tf/Nk);
  ControlTparam<Vector4d, 4, 2> ctp(*sys, tks);
    */
  

  //spsa.reset(new RccarGn(*sys, cost, ctp, ts, xs, us));  
  spsa.reset(new RccarSPSA(*sys, cost, ts, xs, us));  

  {
    VectorXd temp;
    XmlRpc::XmlRpcValue list;

    if(nh.getParam("stepcoeffs", list))
      xml2vec(temp,list);
    ROS_ASSERT(temp.size() == 4);

    spsa->stepc.a = temp[0];
    spsa->stepc.c1 = temp[1];
    spsa->stepc.alpha = temp[2];
    spsa->stepc.gamma = temp[3];//Set coeffs based on parameters
    cout<<"Step Gains: "<<(spsa->stepc.a)<<"\t"<<(spsa->stepc.c1)<<"\t"<<(spsa->stepc.alpha)<<"\t"<<(spsa->stepc.gamma)<<endl;
  }

  spsa->debug = true; 
  //Add Visualize option through external function passing #TODO

  //Set up dynamic reconfigure:
	dynamic_reconfigure::Server<gcop_ros_bullet::CEInterfaceConfig> server;
	dynamic_reconfigure::Server<gcop_ros_bullet::CEInterfaceConfig>::CallbackType f = boost::bind(&ParamreqCallback, _1, _2);
	server.setCallback(f);

  //Print the current trajectory :
  cout<<"us_size"<<us.size()<<endl;
  for(int i =0;i < us.size();i++)
  {
    cout<<"us["<<i<<"]: "<<us[i].transpose()<<endl;
    cout<<"xs["<<i+1<<"]: "<<xs[i+1].transpose()<<endl;
  }

  //Setup Trajectory variable for visualization:
  line_strip.header.frame_id = "/world";
	line_strip.ns = "traj";
	line_strip.action = visualization_msgs::Marker::ADD;
	line_strip.pose.orientation.w = 1.0;
	line_strip.id = 1;
	line_strip.type = visualization_msgs::Marker::LINE_STRIP;
	line_strip.scale.x = 0.1;
	line_strip.color.r = 1.0;
	line_strip.color.a = 1.0;
  line_strip.points.resize(xs.size());
  //Setup sample trajectory
  sampleline_strip.header.frame_id = "/world";
	sampleline_strip.ns = "sampletraj";
	sampleline_strip.action = visualization_msgs::Marker::ADD;
	sampleline_strip.pose.orientation.w = 1.0;
	sampleline_strip.id = 1;
	sampleline_strip.type = visualization_msgs::Marker::LINE_STRIP;
	sampleline_strip.scale.x = 0.01;
	sampleline_strip.color.b = 1.0;
	sampleline_strip.color.a = 1.0;
  sampleline_strip.points.resize(xs.size());

  //initializing joint msg for animation
	joint_state.name.resize(3);
	//joint_state.header.frame_id = "movingcar";//no namespace for now since only one car present
	joint_state.position.resize(3);
	joint_state.name[0] = "base_to_frontwheel1";
	joint_state.name[1] = "base_to_frontwheel2";
	joint_state.name[2] = "base_to_backwheel1";
  // Initialize control trajectory for publishing:
  sendtrajectory = false;
  nh.getParam("sendtrajectory",sendtrajectory);
  if(sendtrajectory)
  {
    cout<<"Send Trajectory SET "<<endl;
    Nreq = N;
    nh.getParam("Nreq",Nreq);
    assert((Nreq <= N) && (Nreq >0));
    trajectory.N = Nreq;
    trajectory.ctrl.resize(Nreq);
    trajectory.time.assign(ts.begin(), ts.begin()+Nreq+1);
    for(int count1 = 0;count1 < Nreq; count1++)
    {
      trajectory.ctrl[count1].ctrlvec.resize(2);//2 controls
    }
  }

  while(ros::ok())
  {
    ros::spinOnce();
    usleep(10000);//100 Hz roughly spinning
  }
  //spsa.reset();//Clear spsa
  
  return 0;
}
