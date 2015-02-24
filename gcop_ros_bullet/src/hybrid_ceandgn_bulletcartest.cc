/** This example computes optimal trajectories for a rccar using Cross Entropy based sampling method. 
 * Gcop library is used for creating rccar system and algorithm etc.
 * Rviz is used for display
 */
//System stuff
#include <iostream>
#include <iomanip>

//Gcop Stuff
#include <gcop/systemce.h>
#include <gcop/gndocp.h>
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
#include <std_msgs/Float64.h>

#define USE_SPLINEPARAM

#ifdef USE_SPLINEPARAM
#include <gcop/uniformsplinetparam.h>
#else
#include <gcop/controltparam.h>
#endif



using namespace std;
using namespace Eigen;
using namespace gcop;

typedef SystemCe<Vector4d, 4, 2, Dynamic> RccarCe;
typedef GnDocp<Vector4d, 4, 2, Dynamic, 6> RccarGn;
typedef Matrix<double, 6, 1> Vector6d;//#DEBUG

//Global variable:
boost::shared_ptr<RccarCe> ce;///<Cross entropy based solver
boost::shared_ptr<RccarGn> gn;///<Cross entropy based solver
boost::shared_ptr<Bulletrccar> sys;///Bullet rccar system
#ifdef USE_SPLINEPARAM
boost::shared_ptr<UniformSplineTparam<Vector4d, 4, 2> > ctp;//Parametrization
#else
boost::shared_ptr<ControlTparam<Vector4d, 4, 2> > ctp;//Parametrization
#endif
vector<Vector4d> xs;///< State trajectory of the system
vector<Vector2d> us;///< Controls for the trajectory of the system
vector<double> ts;///< Times for trajectory
vector<double> zs;///< Height of the car along the trajectory (same as xs) only used for visualization
int Nreq;///< Number of segments requested for gcop trajectory
Vector4d xf(0,0,0,0);///< final state
double marker_height;///< Height of the final arrow

//ros publisher and subscribers:
ros::Publisher joint_pub;///<Rccar model joint publisher for animation
ros::Publisher traj_pub;///<Rviz best trajectory publisher
ros::Publisher sampletraj_pub;///<Rviz sample trajectories publisher
ros::Publisher gcoptraj_pub;///<Publisher for gcop trajectory (States, times, Controls)
ros::Publisher costlog_pub;///<Publish the cost after every iteration for logging

//Messages:
visualization_msgs::Marker line_strip;///<Best trajectory message
visualization_msgs::Marker sampleline_strip;///<Sample trajectory message
visualization_msgs::Marker goal_arrow;///<Best trajectory message
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

/**
 * External render function for displaying sample trajectories from CE method
 */
 void render_trajectory(int id, vector<Vector4d> &xss)
 {
   sampleline_strip.id = id;
   //Can use pragma here OpenMP
   for(int count1 = 0;count1 < xs.size(); count1++)
   {
     sampleline_strip.points[count1].x = xss[count1][0];
     sampleline_strip.points[count1].y = xss[count1][1];
     sampleline_strip.points[count1].z = zs[count1];
     //cout<<"count,x,y,z: "<<count1<<"\t"<<xss[count1][0]<<"\t"<<xss[count1][1]<<"\t"<<zs[count1]<<endl;
   }
   sampletraj_pub.publish(sampleline_strip);
   //getchar();//#DEBUG
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

    //Set rviz goal model also
    goal_arrow.pose.position.x = xf(0);
    goal_arrow.pose.position.y = xf(1);
    goal_arrow.pose.position.z = marker_height;
    tf::quaternionTFToMsg(tf::createQuaternionFromYaw(xf(2)+M_PI/2), goal_arrow.pose.orientation);
    traj_pub.publish(goal_arrow);

    //Publishing initial trajectory:
    line_strip.header.stamp  = ros::Time::now();
    for(int i =0;i<xs.size(); i++)
    {
      //geometry_msgs::Point p;
      line_strip.points[i].x = xs[i][0];
      line_strip.points[i].y = xs[i][1];
      line_strip.points[i].z = zs[i];//Need to add  this to state or somehow get it #TODO
    }
    traj_pub.publish(line_strip);

    //Publish initial trajectory cost:
    //Publish the cost of the initial trajectory:
    {
      double cost1 = 0;
      int N = us.size();
      int count_cost = 0;
      for(count_cost = 0;count_cost < N;count_cost++)
      {
        double h = (ts[count_cost+1] - ts[count_cost]);
        cost1 += (ce->cost).L(ts[count_cost], xs[count_cost], us[count_cost], h, 0);
      }
      cost1 += (ce->cost).L(ts[count_cost], xs[count_cost], us[count_cost-1], 0, 0);
      cout<<"Initial cost: "<<cost1<<endl;

      std_msgs::Float64 costmsg;///<Message with the current cost after every iteration
      costmsg.data = cost1;
      costlog_pub.publish(costmsg);
    }

    cout<<"Iterating: "<<endl;
    for (int i = 0; i < config.Nit; ++i) {
      currtime = ros::Time::now();
      ce->Iterate();
      cout << "Iteration #" << i << " took: " << (ros::Time::now() - currtime).toSec()*1e3 << " ms." << endl;
      cout << "Cost=" << ce->J << endl;

      /*std_msgs::Float64 costmsg;///<Message with the current cost after every iteration
      costmsg.data = ce->J;
      costlog_pub.publish(costmsg);
      costmsg.data = ce->nofevaluations;
      costlog_pub.publish(costmsg);

      //Set the Gn guess to be the ce guess:
      if(abs(ce->ce.Jmin - ce->J)<1e-3)
        gn->s = ce->ce.zmin;//Current mean
      */
      int Nelite = ce->ce.zps.size();//Number zps pairs left
      cout<<"Elite samples: "<<Nelite<<endl;
      //set the min Cost for the elite samples as very high;
      ce->ce.Jmin = std::numeric_limits<double>::max();

      //For each of the elite samples run GN method:
      for(int count_e =0;count_e < Nelite; count_e++)
      {
        gn->s = ce->ce.zps[count_e].first;//Current parameter guess
        if(gn->lm)//Recreate gn problem everytime
        {
          delete (gn->lm);
          gn->lm = 0;
        }
        //cout<<"Cost before iteration: "<<(ce->ce.cs[count_e])<<endl;
        for(int it_gn = 0;it_gn < 5; it_gn++)
        {
          gn->Iterate();//Iterate GN
          if(gn->info == 2)
            break;
        }
        cout << "Cost_after_before: " << (ce->ce).cs[count_e]<<"\t"<<gn->J << endl;//GN Cost after one iteration
        ce->ce.zps[count_e].first = gn->s;//Copy Back the sample to CE
        ce->ce.zps[count_e].second = gn->J;
        //Verify if new gn cost is less than ce J:
        if((gn->J) < (ce->ce.Jmin))
        {
          ce->ce.zmin = (gn->s);
          ce->ce.Jmin = gn->J;
        }
        //getchar();
      }
      //With new samples Fit CE again:
      if (!(ce->ce.Fit())) {
        cout << "[W] TrajectoryPrmSample::Sample: ce.Fit failed!" << endl;
      }
      if((ce->ce.Jmin) < ce->J)//For the updated samples, if there is a sample which has a lower cost than the current min, then update xs, us and the min cost
      {
        ctp->From(ts, xs, us, (ce->ce.zmin),ce->p);
        //cout<<"Update ce cost: "<<ce->Update(xs, us)<<endl;//#DEBUG
        ce->Update(xs,us,false);
        ce->J = (ce->ce).Jmin;
        ce->zmin = (ce->ce).zmin;
      }
      else
      {
        ctp->From(ts, xs, us, ce->zmin,ce->p);
      }
      cout<<"Final ce->J: "<<(ce->J)<<endl;
//Publish the line strip
      for(int i =0;i<xs.size(); i++)
      {
        //geometry_msgs::Point p;
        line_strip.points[i].x = xs[i][0];
        line_strip.points[i].y = xs[i][1];
        line_strip.points[i].z = zs[i];//Need to add  this to state or somehow get it #TODO
      }
      traj_pub.publish(line_strip);
        


      
      /*for(int count_gn = 0;count_gn < 2;count_gn++)
      {
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
        
        
        //cout<<"Current Param guess: "<<(gn->s).transpose()<<endl;

        gn->Iterate();//Iterate Gauss newton 2 times for every ce
        cout << "Cost=" << gn->J << endl;

        std_msgs::Float64 costmsg;///<Message with the current cost after every iteration
        costmsg.data = gn->J;
        costlog_pub.publish(costmsg);
        costmsg.data = gn->nofevaluations;
        costlog_pub.publish(costmsg);
        //getchar();
      }
      //Set the ce mu to the one from gauss newton optimization:
      ce->ce.gmm.ns[0].mu = gn->s;
      ce->ce.zmin = gn->s;
      ce->J = gn->J;
      */
      /*{
        double cost1 = 0;
        int N = us.size();
        Vector6d g;
        int count_cost = 0;
        for(count_cost = 0;count_cost < N;count_cost++)
        {
          double h = (ts[count_cost+1] - ts[count_cost]);
          cost1 += (ce->cost).L(ts[count_cost], xs[count_cost], us[count_cost], h, 0);
          //gn->cost.Res(g, ts[count_cost], xs[count_cost], us[count_cost], h, 0);
          //cost2 += 0.5*(g.transpose()*g);
        }
        cost1 += (ce->cost).L(ts[count_cost], xs[count_cost], us[count_cost-1], 0, 0);

        ce->J =  cost1;
      }
      */
      //cout<<"xsN: "<<xs.back().transpose()<<endl;
    }
    
    /*for(int i =0;i < us.size();i++)
    {
      cout<<"us["<<i<<"]: "<<us[i].transpose()<<endl;
      cout<<"xs["<<i+1<<"]: "<<xs[i+1].transpose()<<endl;
    }//#DEBUG
    */

    //Publish control trajectory when parameter is set:
    for (int count = 0;count<Nreq;count++)
    {
	    for(int count1 = 0;count1 < 2;count1++)
	    {
		    trajectory.ctrl[count].ctrlvec[count1] = us[count](count1);
	    }
    }
 
    config.iterate = false;
  }
  if(config.send_traj)
  {
    ROS_INFO("Publishing Trajectory");
    gcoptraj_pub.publish(trajectory);
    config.send_traj = false;
  }
  if(config.animate)
  {
    //Run the system:
    sys->reset(xs[0],ts[0]);
    for(int count1 = 0;count1 < us.size();count1++)
    {
      sys->Step2(us[count1], ts[count1+1]-ts[count1]);
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
      btTransform chassistransform = (sys->m_carChassis)->getWorldTransform();
      btQuaternion chassisquat = chassistransform.getRotation();
      btVector3 chassisorig = chassistransform.getOrigin();
      tf::Transform tf_chassistransform;
      tf_chassistransform.setRotation(tf::Quaternion(chassisquat.x(), chassisquat.y(), chassisquat.z(), chassisquat.w()));
      tf_chassistransform.setOrigin(tf::Vector3(chassisorig.x(), chassisorig.y(), chassisorig.z()));
      broadcaster->sendTransform(tf::StampedTransform(tf_chassistransform, ros::Time::now(), "world", "base_link"));
      cout<<"ce->sys.x" <<ts[count1+1]<<"\txs: "<<(sys->x).transpose()<<endl;//#DEBUG
      usleep((ts[count1+1] - ts[count1])*1e6);//microseconds
    }
      config.animate = false;
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "cecartest");

  cout<<std::setprecision(20)<<endl;

  ros::NodeHandle nh("~");

  //setup tf
	broadcaster = new tf::TransformBroadcaster();

  //Setup publishers and subscribers:
	traj_pub = nh.advertise<visualization_msgs::Marker>("best_traj", 1);
	sampletraj_pub = nh.advertise<visualization_msgs::Marker>("sample_traj", 100);
	joint_pub = nh.advertise<sensor_msgs::JointState>("/joint_states", 1);
  gcoptraj_pub = nh.advertise<gcop_comm::CtrlTraj>("ctrltraj",1);
  costlog_pub = nh.advertise<std_msgs::Float64>("cost",10);

  /* Parametersf for solver */
  int N = 32;        // number of segments
  double tf = 5;    // time horizon

  nh.getParam("N",N);
  ROS_ASSERT(2*int(N/2) == N);
  nh.getParam("tf",tf);

  if(!nh.getParam("marker_height",marker_height))
    marker_height = 1.0;

  cout<<"N: "<<N<<endl;
  cout<<"tf: "<<tf<<endl;
  cout<<"marker_height: "<<marker_height<<endl;

  double h = tf/N;   // time step

  //Create Bullet world and rccar system:
  BulletWorld world(true);//Set the up axis as z for this world

  zs.resize(N+1);//<Resize the height vector and pass it to the rccar system
  sys.reset(new Bulletrccar(world, &zs));
  sys->initialz = 0.12;
  sys->gain_cmdvelocity = 1;
  sys->kp_steer = 0.2;
  sys->kp_torque = 100;
  sys->gain_cmdvelocity = 1.04;
  //sys->kp_torque = 25;
  sys->steeringClamp = 15.0*(M_PI/180.0);
  sys->U.lb[0] = -(sys->steeringClamp);
  sys->U.ub[1] = (sys->steeringClamp);
  sys->U.bnd = true;

  nh.getParam("initialz", (sys->initialz));

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
  Vector2d us_firsthalf(0.1,0.0);
  Vector2d us_secondhalf(0.1,0.0);

  // cost
  RnLqCost<4, 2, Dynamic, 6> cost(*sys, tf, xf);

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

    if(nh.getParam("us1", list))
      xml2vec(temp,list);
    ROS_ASSERT(temp.size() == 2);

    us_firsthalf = temp;

    if(nh.getParam("us2", list))
      xml2vec(temp,list);
    ROS_ASSERT(temp.size() == 2);

    us_secondhalf = temp;

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
    //us[i] = Vector2d(0.5, 0);
    //us[N/2+i] = Vector2d(0.5, 0);    
    us[i] = us_firsthalf;
    us[N/2+i] = us_secondhalf;    
    //us[i] = Vector2d(1, -.1);
    //us[N/2+i] = Vector2d(1, -.1);    
  }
  //Set initial state:
  xs[0] = x0;
  /*Initialize xs to x0
  for(int i = 0; i <= N; ++)
  {
    xs[i] = x0;
  }
  */

  Vector2d du(.2, .1);///<Hardcoded if needed change the du and de here

  Vector2d e(.001, .001);

  vector<Vector2d> dus(N, du);
  vector<Vector2d> es(N, e);

  //Create Parametrization:
  int Nk = 10;
  nh.getParam("Nk", Nk);

#ifdef USE_SPLINEPARAM
  VectorXd tks(Nk+1);
#else
  vector<double> tks(Nk+1);
#endif
  for (int k = 0; k <=Nk; ++k)
  {
    tks[k] = k*(tf/Nk);
  }

#ifdef USE_SPLINEPARAM
  int degree = 2;
  nh.getParam("degree",degree);
  assert(degree>0);
  ctp.reset(new UniformSplineTparam<Vector4d, 4, 2>(*sys, tks, degree));// Create Linear Parametrization of controls
#else
  ctp.reset(new ControlTparam<Vector4d, 4, 2>(*sys, tks));// Create Linear Parametrization of controls
#endif



  gn.reset(new RccarGn(*sys, cost, *ctp, ts, xs, us));  
  gn->numdiff_stepsize = 1e-6;
  gn->debug = false;

  ce.reset(new RccarCe(*sys, cost, *ctp, ts, xs, us, 0, dus, es));//Can pass custom parameters here too
  ce->ce.mras = false;///<#TODO Find out what these are (This is a different type of CE sampling method which uses all the samples instead of only the elite samples)
  ce->ce.inc = false;///<#TODO Find out what these are
  ce->external_render = &render_trajectory;

  nh.getParam("Ns", ce->Ns);


  ce->debug = true; 

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

  //Create a cylinder marker:
  goal_arrow.header.frame_id="/world";
  goal_arrow.ns = "goalmarker";
  goal_arrow.action = visualization_msgs::Marker::ADD;
  goal_arrow.id = 1000;//Some big number which should not match wit others
  goal_arrow.type = visualization_msgs::Marker::ARROW;
  goal_arrow.pose.orientation.w = 1.0;
  goal_arrow.scale.x = 1.5;
  goal_arrow.scale.y = 0.1;
  goal_arrow.scale.z = 0.1;
  goal_arrow.color.r = 1.0;
  goal_arrow.color.a = 1.0;

  //initializing joint msg for animation
	joint_state.name.resize(3);
	//joint_state.header.frame_id = "movingcar";//no namespace for now since only one car present
	joint_state.position.resize(3);
	joint_state.name[0] = "base_to_frontwheel1";
	joint_state.name[1] = "base_to_frontwheel2";
	joint_state.name[2] = "base_to_backwheel1";
  // Initialize control trajectory for publishing:

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

  while(ros::ok())
  {
    ros::spinOnce();
    usleep(10000);//100 Hz roughly spinning
  }
  //ce.reset();//Clear ce
  
  return 0;
}
