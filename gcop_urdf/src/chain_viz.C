/** This node takes the ctrl trajectory data for visualization. It publishes the joint message
* to the robot state publisher and also publishes the baselink transformation needed to move the robot.
*/
#include "ros/ros.h"
#include <sensor_msgs/JointState.h>
#include <tf/transform_broadcaster.h>
#include <visualization_msgs/Marker.h>
#include "gcop_comm/CtrlTraj.h"


ros::Subscriber traj_sub;
ros::Publisher traj_pub;
ros::Publisher joint_pub;
ros::Publisher finaljoint_pub;
tf::TransformBroadcaster *broadcaster;


// message declarations
geometry_msgs::TransformStamped global_trans;
geometry_msgs::TransformStamped finalgoal_trans;
sensor_msgs::JointState joint_state;
sensor_msgs::JointState finaljoint_state;
visualization_msgs::Marker line_strip;

bool recvdmsg = false;
bool sim = false;

void append(std::vector<std::string> &names, std::string prefix)
{
	for(int count = 0;count < names.size();count++)
	{
		names[count] = (prefix +"/") +  names[count];
		std::cout<<names[count]<<std::endl;
	}
	
}


void joint_publish(const gcop_comm::CtrlTraj::ConstPtr& trajectory)
{

	double dt = 0;
	if(!recvdmsg)
	{
		global_trans.child_frame_id = "/movingrobot/" + trajectory->rootname;
		finalgoal_trans.child_frame_id = "/goalrobot/" + trajectory->rootname;
		finaljoint_state.name = trajectory->finalgoal.names;
		//append(finaljoint_state.name, "goalrobot");
	}

	//publish zero transform 
	global_trans.header.stamp = ros::Time::now();
	global_trans.transform = trajectory->statemsg[0].basepose;

	broadcaster->sendTransform(global_trans);//send 0 posn

  //Resize line strip
	line_strip.header.stamp  = ros::Time::now();
	line_strip.points.resize(trajectory->N + 1);

  line_strip.points[0].x = trajectory->statemsg[0].basepose.translation.x;
  line_strip.points[0].y = trajectory->statemsg[0].basepose.translation.y;
  line_strip.points[0].z = trajectory->statemsg[0].basepose.translation.z;

	//std::cout<<"Simulation parameter"<<sim<<std::endl;

	//publish final goal:
	if(!sim)
	{
		finalgoal_trans.header.stamp = ros::Time::now();
		finalgoal_trans.transform = trajectory->finalgoal.basepose;

		broadcaster->sendTransform(finalgoal_trans);//send 0 posn

		finaljoint_state.header.stamp = ros::Time::now();
		finaljoint_state.position = trajectory->finalgoal.statevector;
		finaljoint_pub.publish(finaljoint_state);
	}

	for(int i = 1;i<(trajectory->N)+1; i++)
	{
		//joint_state.name.resize(3);
		//joint_state.position.resize(3);
		joint_state.header.stamp = ros::Time::now();
		joint_state.position = trajectory->statemsg[i].statevector;
		if(!recvdmsg)
		{
			joint_state.name = trajectory->statemsg[i].names;
			//append(joint_state.name, "movingrobot");
		}
		//send joint state
		joint_pub.publish(joint_state);

		dt = trajectory->time[i] - trajectory->time[i-1];// Have to change later to not sleep
		ros::Duration(dt).sleep();//sleeps for dt time

		// update transform
		global_trans.header.stamp = ros::Time::now();
		global_trans.transform = trajectory->statemsg[i].basepose;

		//send the transform
		broadcaster->sendTransform(global_trans);

    //assign line strip:
    line_strip.points[i].x = trajectory->statemsg[i].basepose.translation.x;
    line_strip.points[i].y = trajectory->statemsg[i].basepose.translation.y;
    line_strip.points[i].z = trajectory->statemsg[i].basepose.translation.z;
	}
	traj_pub.publish(line_strip);
	recvdmsg = true;
}
int main(int argc, char** argv) {
	ros::init(argc, argv, "state_publisher");
	ros::NodeHandle n;

	//get parameter saying whether we are simulating or not:
	ros::param::get("/mbssim",sim);//for getting parameter for simulation

	broadcaster = new tf::TransformBroadcaster();

	//initializing joint msg
	

	traj_sub = n.subscribe("/mbsddp/ctrltraj",1, joint_publish);

	joint_pub = n.advertise<sensor_msgs::JointState>("/movingrobot/joint_states", 1);
	finaljoint_pub = n.advertise<sensor_msgs::JointState>("/goalrobot/joint_states", 1);
	traj_pub = n.advertise<visualization_msgs::Marker>("optimal_traj", 1);

	//initialize global transform 
	global_trans.header.frame_id = "world";
	global_trans.child_frame_id = "/movingrobot/baselink";
	joint_state.header.frame_id = "movingrobot";


	//initialize final goal:
	finalgoal_trans.header.frame_id = "world";
	finalgoal_trans.child_frame_id = "/goalrobot/baselink";
	finaljoint_state.header.frame_id = "goalrobot";
  
  //initialize Base trajectory:
  line_strip.header.frame_id = "/world";
	line_strip.ns = "traj";
	line_strip.action = visualization_msgs::Marker::ADD;
	line_strip.pose.orientation.w = 1.0;
	line_strip.id = 1;
	line_strip.type = visualization_msgs::Marker::LINE_STRIP;
	line_strip.scale.x = 0.05;
	line_strip.color.b = 1.0;
	line_strip.color.a = 1.0;

	ros::Rate loop_rate(100);
	while(ros::ok())
	{
		if(recvdmsg)
		{
			//publish zero transform 
			joint_state.header.stamp = ros::Time::now();
			global_trans.header.stamp = ros::Time::now();
			joint_pub.publish(joint_state);
			broadcaster->sendTransform(global_trans);
			if(!sim)
				broadcaster->sendTransform(finalgoal_trans);
			//publish final transform
		}
		ros::spinOnce();
		loop_rate.sleep();
	}


	
		return 0;
}
