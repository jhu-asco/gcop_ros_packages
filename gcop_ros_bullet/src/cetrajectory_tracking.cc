/** This example computes optimal trajectories for a rccar using Cross Entropy based sampling method. 
 * Gcop library is used for creating rccar system and algorithm etc.
 * Rviz is used for display
 */
//System stuff
#include <iostream>
#include <fstream>

//Gcop Stuff
#include <gcop/systemce.h>
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

#include <boost/thread.hpp>


using namespace std;
using namespace Eigen;
using namespace gcop;

typedef SystemCe<Vector4d, 4, 2, Dynamic> RccarCe;

//Global variable:
boost::shared_ptr<RccarCe> ce;///<Cross entropy based solver
boost::shared_ptr<Bulletrccar> sys;///Bullet rccar system
boost::shared_ptr<BaseSystem> base_sys;///Bullet rccar system
vector<Vector4d> xs;///< State trajectory of the system
vector<Vector2d> us;///< Controls for the trajectory of the system
vector<Vector2d> dus;///< Variance of the controls  in CE
vector<Vector2d> es;///< Variance which keeps CE from getting stuck into local minimum

vector<double> ts;///< Times for trajectory
vector<double> zs;///< Height of the car along the trajectory (same as xs) only used for visualization
bool sendtrajectory;///< Send the gcop trajectory 
int Nreq;///< Number of segments requested for gcop trajectory
Vector4d xf(0,0,0,0);///< final state
double marker_height;///< Height of the final arrow
ofstream costlogfile("/home/gowtham/hydro_workspace/src/gcop_ros_packages/gcop_ros_bullet/results/costs/ce.dat");
ofstream optimaltrajlogfile("/home/gowtham/hydro_workspace/src/gcop_ros_packages/gcop_ros_bullet/results/costs/cetraj.dat");

//Animation thread variables:
boost::mutex control_mutex;
boost::mutex truestate_mutex;
bool close_thread = false, true_state_set = false, controls_updated = false;
std::string mesh_filename;

//ros publisher and subscribers:
ros::Publisher joint_pub;///<Rccar model joint publisher for animation
ros::Publisher traj_pub;///<Rviz best trajectory publisher
ros::Publisher sampletraj_pub;///<Rviz sample trajectories publisher
ros::Publisher gcoptraj_pub;///<Publisher for gcop trajectory (States, times, Controls)
ros::Publisher costlog_pub;///<Publish the cost after every iteration for logging

//Messages:
visualization_msgs::Marker line_strip;///<Best trajectory message
visualization_msgs::Marker sampleline_strip;///<Sample trajectory message
//visualization_msgs::Marker goal_cylinder;///<Best trajectory message
visualization_msgs::Marker goal_arrow;///<Best trajectory message
sensor_msgs::JointState joint_state;///< Joint states for wheels in animation
gcop_comm::CtrlTraj trajectory;///<Trajectory with the optimal control efforts

//tf
tf::TransformBroadcaster *broadcaster;

//CarState (Real car state which is propagated with noisy control)
Bulletrccar::CarState true_initialstate;

#ifdef USE_SPLINEPARAM
boost::shared_ptr<UniformSplineTparam<Vector4d, 4, 2> > ctp;//Parametrization
#else
boost::shared_ptr<ControlTparam<Vector4d, 4, 2> > ctp;//Parametrization
#endif

boost::shared_ptr< RnLqCost<4, 2> > cost;//Cost class

int Ns;//Number of samples

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

/** Animation thread which runs in parallel to optimization thread
*/
void Animate()
{
  BulletWorld parallel_world(true);//Set the up axis as z for this parallel_world
  boost::shared_ptr<Bulletrccar> parallel_sys;///Bullet rccar system
  boost::shared_ptr<BaseSystem> base_parallel_sys;///Bullet rccar system
  parallel_sys.reset(new Bulletrccar(parallel_world));
  base_parallel_sys = boost::static_pointer_cast<BaseSystem>(parallel_sys);
  parallel_sys->initialz = 0.18;
  parallel_sys->gain_cmdvelocity = 1.04;
  parallel_sys->kp_steer = 0.2;
  parallel_sys->kp_torque = 100;
  parallel_sys->steeringClamp = 15.0*(M_PI/180.0);
  parallel_sys->U.lb[0] = -(parallel_sys->steeringClamp);
  parallel_sys->U.ub[1] = (parallel_sys->steeringClamp);
  parallel_sys->U.bnd = true;

  parallel_sys->initialz = sys->initialz;

  parallel_sys->offsettrans.setIdentity();
  parallel_sys->offsettransinv.setIdentity();

  //Load Ground
  {
    cout<<"Filename: "<<mesh_filename<<endl;
    btCollisionShape *groundShape;
    if(mesh_filename.compare("plane") == 0)
      groundShape = parallel_world.CreateGroundPlane(20,20);
    else
      groundShape= parallel_world.CreateMeshFromSTL(mesh_filename.c_str());//20 by 20 long plane

    btTransform tr;
    tr.setOrigin(btVector3(0, 0, 0));
    tr.setRotation(btQuaternion(0,0,0));
    parallel_world.LocalCreateRigidBody(0,tr, groundShape);
  }

  double propagate_time = 0.5;
  double timestep = ts[1] - ts[0];
  int propagate_steps = round(propagate_time/timestep);//nofsteps to propagate
  assert(propagate_steps > 0);
  vector<Vector2d> us_parallel(propagate_steps);///< Controls for propagation of system 

  truestate_mutex.lock();
  parallel_sys->Reset(xs[0],ts[0]);
  parallel_sys->setinitialstate(xs[0]);//This is used to fill the initial state matrix in the system based on the current state of the car. It also fills xs[0]
  true_initialstate = *(parallel_sys->initialstate);
  true_state_set = true;
  truestate_mutex.unlock();

  Eigen::Vector4d x0temp;
  while(!close_thread)
  {
    {
      int counter = 0;
      while(1)
      {
        control_mutex.lock();
        if(controls_updated == true || counter > 1000)
        {
          //Copy controls to propagate forward:
          for(int propagate_count = 0; propagate_count < propagate_steps; propagate_count++)
          {
            us_parallel[propagate_count] = us[propagate_count];
          }

          controls_updated = false;
          control_mutex.unlock();
          break;
        }
        else
        {
          cout<<"Waiting for controls to be updated..."<<endl;
          counter++;
        }
        control_mutex.unlock();
        usleep(10000);
      }
      if(counter > 1000)
      {
        cout<<"Timeout: controls were not updated"<<endl;
        return;
      }
    }

    for(int propagate_count = 0; propagate_count<propagate_steps; propagate_count++)
    {
      //Add noise
      base_parallel_sys->Step(us_parallel[propagate_count], timestep);
      //Visualization:
      //Set the car joint stuff:
      joint_state.header.stamp = ros::Time::now();
      //Back wheel
      joint_state.position[2] = 0;
      //steering angle
      joint_state.position[0] = parallel_sys->gVehicleSteering;
      joint_state.position[1] = parallel_sys->gVehicleSteering;
      //send joint state
      joint_pub.publish(joint_state);
      //Set the pose for the car:
      btTransform chassistransform = (parallel_sys->m_carChassis)->getWorldTransform();
      btQuaternion chassisquat = chassistransform.getRotation();
      btVector3 chassisorig = chassistransform.getOrigin();
      tf::Transform tf_chassistransform;
      tf_chassistransform.setRotation(tf::Quaternion(chassisquat.x(), chassisquat.y(), chassisquat.z(), chassisquat.w()));
      tf_chassistransform.setOrigin(tf::Vector3(chassisorig.x(), chassisorig.y(), chassisorig.z()));
      broadcaster->sendTransform(tf::StampedTransform(tf_chassistransform, ros::Time::now(), "world", "base_link"));
      //ros::spinOnce();
      cout<<"ce->parallel_sys.x" <<ts[propagate_count+1]<<"\txs: "<<(parallel_sys->x).transpose()<<endl;//#DEBUG
      usleep(timestep*1e6);//microseconds
    }

    parallel_sys->setinitialstate(x0temp);//fill initialstate with current car state

    truestate_mutex.lock();

    true_initialstate = *(parallel_sys->initialstate);//store this state
    true_state_set = true;

    truestate_mutex.unlock();
  }
  close_thread = false;//Put back to default state
}

/**
 * Set xf on a circle
 */
 void setxf(double dt)
 {
   double r = sqrt(xf(0)*xf(0) + xf(1)*xf(1));
   cout<<"r: "<<r<<endl;
   if(r < 0.001)
     r = 0.001;
   double omega = xf(3)/r;
   //double omega = 0.6;
   xf(2) += omega*dt;
   /*if((xf(2)) > M_PI)
   {
     cout<<"xf: "<<xf.transpose()<<endl;
     cout<<"x0: "<<xs[0].transpose()<<endl;
     getchar();
     xf(2) -= 2*M_PI;
   }
   else if((xf(2)) < -M_PI)
   {
     cout<<"xf: "<<xf.transpose()<<endl;
     cout<<"x0: "<<xs[0].transpose()<<endl;
     getchar();
     xf(2) += 2*M_PI;
   }
   */

   xf(0) = r*cos(xf(2));
   xf(1) = r*sin(xf(2));
   cout<<"xf: "<<xf.transpose()<<endl;
   cout<<"omega:"<<omega<<endl;
   return;
 }

/** Publish rviz trajectory
*/
void Publishrviztrajectory()
{
  //Publish rviz Trajectory for visualization:
  line_strip.header.stamp  = ros::Time::now();

  for(int i =0;i<xs.size(); i++)
  {
    //geometry_msgs::Point p;
    line_strip.points[i].x = xs[i][0];
    line_strip.points[i].y = xs[i][1];
    line_strip.points[i].z = zs[i]+0.1;//Need to add  this to state or somehow get it #TODO
  }
  traj_pub.publish(line_strip);
  return;
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
     sampleline_strip.points[count1].z = zs[count1]+0.1;
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
    Publishrviztrajectory();

    //Publish initial trajectory cost:
    //Publish the cost of the initial trajectory:
    {
      double cost1 = 0;
      int N = us.size();
      int count_cost = 0;
      for(count_cost = 0;count_cost < N;count_cost++)
      {
        double h = (ts[count_cost+1] - ts[count_cost]);
        cost1 += (cost)->L(ts[count_cost], xs[count_cost], us[count_cost], h, 0);
      }
      cost1 += (cost)->L(ts[count_cost], xs[count_cost], us[count_cost-1], 0, 0);
      cout<<"Initial cost: "<<cost1<<endl;

      //std_msgs::Float64 costmsg;///<Message with the current cost after every iteration
      //costmsg.data = cost1;
      //costlog_pub.publish(costmsg);
      costlogfile<<cost1<<"\t"<<1<<endl;

    }
    cout<<"Iterating: "<<endl;
    cout<<"xf: "<<xf.transpose()<<endl;
    //Evaluate average frequency:
    double total_iteration_time = 0.0;
    //We are tracking a circle or radius tracking_rad and speed = 2 m/s
    //We do 3 iterations before updating the trajectory
    //We update the optimal trajectory every 2Hz => the closedloop control component is for 0.5 seconds based on gains and disturbance
    //Predict without disturbance where the car will be after 0.5 seconds (Can also predict dx there and input that to SDDP #TODO) and optimize the next trajectory 

    //For pure simulation, we run 3 iterations; Then follow the trajectory with/without noise. Use the without noise for optimizing new trajectory for 3 iterations
    //Use with noise for closedloop control

    //Propagate the true position we see to 0.5 seconds forward, run 3 iterations of SDDP, True Execution(run with noise closed loop(true position we see))
    //Added noise to input when following the trajectory.

    int propagate_iterations = 2;//Number of optimization iterations that can be done in the above time
    double propagate_time = 0.5;
    int propagate_steps = round(propagate_time/(ts[1] - ts[0]));//nofsteps to propagate
    assert(propagate_steps > 0);
    int trajectory_steps = round(cost->tf/(ts[1]-ts[0]));//total nofsteps in trajectory

    //Do Initial iteration
    sys->Reset(xs[0],ts[0]);
    //sys->setinitialstate(xs[0]);//This is used to fill the initial state matrix in the system based on the current state of the car. It also fills xs[0]
    //true_initialstate = *(sys->initialstate);
    for(int i = 0; i < propagate_iterations; ++i)
    {
      ce->Iterate();
      Publishrviztrajectory();
    }
    controls_updated = true;//For the first time controls are updated by initial number of iterations
    //Create a boost thread for running parallel system:
		boost::thread animate_thread(Animate);//Passing in function to serial_recv

    for (int i = 0; i < config.Nit*ceil(trajectory_steps/propagate_steps); ++i) {

    //  sys->setinitialstate(true_initialstate, xs[0]);


//////////////////////////////
     // sys->setinitialstate(xs[0]);//fill initialstate with current car state
      //Bulletrccar::CarState temp_true_initialstate = *(sys->initialstate);//store this state

 
      //Project without Noise to predict where we will be starting for the next optimization run
      // *(sys->initialstate) = true_initialstate;
      //wait till true_initialstate is set with a timeout of 1second
      {
        int counter = 0;
        while(1)
        {
          truestate_mutex.lock();
          if(true_state_set == true || counter > 1000)
          {
            true_state_set = false;
            truestate_mutex.unlock();
            break;
          }
          else
          {
            counter++;
            cout<<"Waiting for true_state to be updated..."<<endl;
          }
          truestate_mutex.unlock();
          usleep(10000);
        }
        if(counter > 1000)
        {
          cout<<"Timeout: initialstate was not set"<<endl;
          return;
        }
      }
      sys->setinitialstate(true_initialstate, xs[0]);
      sys->Reset(xs[0],ts[0]);

      //Wait for controls to be copied before changing them
      {
        int counter = 0;
        while(1)
        {
          control_mutex.lock();
          if(controls_updated == false || counter > 1000)
          {
            control_mutex.unlock();
            break;
          }
          else
          {
            cout<<"Waiting for controls to be copied..."<<endl;
            counter++;
          }
          control_mutex.unlock();
          usleep(10000);
        }
        if(counter > 1000)
        {
          cout<<"Timeout: initialstate was not set"<<endl;
          return;
        }
      }

      for(int propagate_count = 0; propagate_count<propagate_steps; propagate_count++)
      {
        //No noise
        base_sys->Step(us[propagate_count], ts[propagate_count+1]-ts[propagate_count]);
      }

      sys->setinitialstate(xs[0]);//Set x0 with predicted state      

      if(i == 0)
        xf(2) = atan2(xf(1),xf(0));//Set the goal angle for circle
      cout<<"xf: "<<xf.transpose()<<endl;
      setxf(propagate_time);//propagate xf forward
      ///Publish goal:
      goal_arrow.pose.position.x = xf(0);
      goal_arrow.pose.position.y = xf(1);
      goal_arrow.pose.position.z = marker_height;
      tf::quaternionTFToMsg(tf::createQuaternionFromYaw(xf(2)+M_PI/2), goal_arrow.pose.orientation);
      traj_pub.publish(goal_arrow);
      //(ce->cost).tf -= propagate_time;
      //if((ce->cost).tf < propagate_time)
        //break;

      //Set us, dus and update the trajectory:
      //sys->Reset(xs[0],ts[0]);
      //double cost_newtrajectory = 0;
      /////DEBUG HERE///// 
      for(int us_count = 0; us_count<trajectory_steps; us_count++)
      {
        int copy_index = (us_count+propagate_steps)>(trajectory_steps-1)?(trajectory_steps-1):(us_count+propagate_steps);
        us[us_count] = us[copy_index];
        /*if(us_count+propagate_steps>trajectory_steps-1)
          {
          dus[us_count] = Vector2d(0.4,0.2);//du hardcoded value change only for new controls
          }
          else
          dus[us_count] = dus[copy_index];
         */
      //  base_sys->Step(xs[us_count+1],us[us_count],ts[us_count+1]-ts[us_count]);//Update states based on these controls
       // cost_newtrajectory += (ce->cost).L(ts[us_count], xs[us_count], us[us_count], (ts[us_count+1]-ts[us_count]), 0);
      }

      ce.reset(new RccarCe(*sys, *cost, *ctp, ts, xs, us, 0, dus, es));//Create a CE problem with the new us and xs #TODO Create a RESET function in Ce which can evaluate a trajectory and update dus etc
      ce->Ns = Ns;
      ce->ce.inc = false;///<#TODO Find out what these are
      //ce->external_render = &render_trajectory;
      //Do Tparam conversion for us, dus also
      //cost_newtrajectory += (ce->cost).L(ts[trajectory_steps], xs[trajectory_steps], us[trajectory_steps-1], 0, 0);
      //ce->J = cost_newtrajectory;
      Publishrviztrajectory();
      //getchar();


      //Iterate 3 times get new us
      for(int iterate_count = 0; iterate_count < propagate_iterations; iterate_count++)
      {
        currtime = ros::Time::now();
        ce->Iterate();
        total_iteration_time += (ros::Time::now() - currtime).toSec();
        cout << "Iteration #" << i << " took: " << (ros::Time::now() - currtime).toSec()*1e3 << " ms." << endl;
        cout << "Cost=" << ce->J << endl;
        //cout<<"xsN: "<<xs.back().transpose()<<endl;

        /*std_msgs::Float64 costmsg;///<Message with the current cost after every iteration
          costmsg.data = ce->J;
          costlog_pub.publish(costmsg);
          costmsg.data = ce->nofevaluations;
          costlog_pub.publish(costmsg);
         */
        costlogfile<<(ce->J)<<"\t"<<(ce->nofevaluations)<<endl;

        //Publish rviz Trajectory for visualization:
        Publishrviztrajectory();
      //  getchar();
      }
      //Let the other thread know we are done with optimization
      control_mutex.lock();
        controls_updated = true;
      control_mutex.unlock();
    }
    close_thread = true;
    cout<<"Average Frequency: "<<(config.Nit/total_iteration_time)<<endl;
    
    for(int i =0;i < us.size();i++)
    {
      cout<<"us["<<i<<"]: "<<us[i].transpose()<<endl;
      cout<<"xs["<<i+1<<"]: "<<xs[i+1].transpose()<<endl;
      optimaltrajlogfile<<ts[i]<<"\t"<<us[i].transpose()<<"\t"<<xs[i].transpose()<<endl;
    }//#DEBUG
    optimaltrajlogfile<<ts[us.size()]<<"\t"<<us[us.size()-1].transpose()<<"\t"<<xs[us.size()].transpose()<<endl;

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
  if(config.send_traj)
  {
    gcoptraj_pub.publish(trajectory);
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

  ros::NodeHandle nh("~");

  //setup tf
	broadcaster = new tf::TransformBroadcaster();

  //Setup publishers and subscribers:
	traj_pub = nh.advertise<visualization_msgs::Marker>("best_traj", 1);
	sampletraj_pub = nh.advertise<visualization_msgs::Marker>("sample_traj", 100);
	joint_pub = nh.advertise<sensor_msgs::JointState>("/joint_states", 1);
  gcoptraj_pub = nh.advertise<gcop_comm::CtrlTraj>("ctrltraj",1);
  costlog_pub = nh.advertise<std_msgs::Float64>("cost",1);

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
  base_sys = boost::static_pointer_cast<BaseSystem>(sys);
  sys->initialz = 0.12;
  sys->gain_cmdvelocity = 1.04;
  sys->kp_steer = 0.2;
  sys->kp_torque = 100;
  sys->steeringClamp = 15.0*(M_PI/180.0);
  sys->U.lb[0] = -(sys->steeringClamp);
  sys->U.ub[1] = (sys->steeringClamp);
  sys->U.bnd = true;

  nh.getParam("initialz", (sys->initialz));

  sys->offsettrans.setIdentity();
  sys->offsettransinv.setIdentity();

  //Load Ground
  {
    nh.getParam("mesh",mesh_filename);
    cout<<"Filename: "<<mesh_filename<<endl;
    btCollisionShape *groundShape;
    if(mesh_filename.compare("plane") == 0)
      groundShape = world.CreateGroundPlane(20,20);
    else
      groundShape= world.CreateMeshFromSTL(mesh_filename.c_str());//20 by 20 long plane

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
  cost.reset(new RnLqCost<4, 2>(*sys, tf, xf));

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

    cost->Q = temp.asDiagonal();

    if(nh.getParam("Qf", list))
      xml2vec(temp,list);
    ROS_ASSERT(temp.size() == 4);

    cost->Qf = temp.asDiagonal();

    if(nh.getParam("R", list))
      xml2vec(temp,list);
    ROS_ASSERT(temp.size() == 2);

    cost->R = temp.asDiagonal();

    cost->UpdateGains();//#TODO Make this somehow implicit in the cost function otherwise becomes a coder's burden !

    cout<<"x0: "<<x0.transpose()<<endl;
    cout<<"xf: "<<xf.transpose()<<endl;
    cout<<"Q: "<<endl<<cost->Q<<endl;
    cout<<"Qf: "<<endl<<cost->Qf<<endl;
    cout<<"R: "<<endl<<cost->R<<endl;
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
    //us[i] = Vector2d(1, 0);
    //us[N/2+i] = Vector2d(1, 0);    
    us[i] = us_firsthalf;
    us[N/2+i] = us_secondhalf;    
  }
  //Set initial state:
  xs[0] = x0;
  /*Initialize xs to x0
  for(int i = 0; i <= N; ++)
  {
    xs[i] = x0;
  }
  */

  Vector2d du(.4, .2);///<Hardcoded if needed change the du and de here

  Vector2d e(.0001, .0001);
  {
    VectorXd temp;
    XmlRpc::XmlRpcValue list;

    //Initial state
    /*if(nh.getParam("du_scale", list))
    {
      xml2vec(temp,list);
      ROS_ASSERT(temp.size() == 2);
      du = temp;
    }

    if(nh.getParam("e", list))
    {
      xml2vec(temp,list);
      ROS_ASSERT(temp.size() == 2);
      e = temp;
    }
    */
  }

  //vector<Vector2d> dus(N, du);
  dus.resize(N,du);
  es.resize(N, e);

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
  ctp.reset(new UniformSplineTparam<Vector4d, 4, 2> (*sys, tks,degree));
#else
  ctp.reset(new ControlTparam<Vector4d, 4, 2> (*sys, tks));
#endif

  ce.reset(new RccarCe(*sys, *cost, *ctp, ts, xs, us, 0, dus, es));//Can pass custom parameters here too
  //ce->ce.mras = true;///<#TODO Find out what these are
  ce->ce.inc = false;///<#TODO Find out what these are
  ce->external_render = &render_trajectory;

  nh.getParam("Ns", Ns);
  ce->Ns = Ns;

  if(!nh.getParam("mras", ce->ce.mras))
    ce->ce.mras = true;

  cout<<"mras: "<<(ce->ce.mras)<<endl;


  ce->debug = true; 
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
  //ce.reset();//Clear ce
  
  return 0;
}
