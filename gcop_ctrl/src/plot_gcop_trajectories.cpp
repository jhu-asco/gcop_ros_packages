#include <ros/ros.h>
#include <iostream>
#include <gcop_comm/gcop_trajectory_visualizer.h>
#include <tf/transform_datatypes.h>
#include <fstream>

using namespace std;

ros::Timer oneshot_timer;

GcopTrajectoryVisualizer *visualizer_;

bool new_dataset;

void getmeasurementCtrlTrajectory(gcop_comm::CtrlTraj &trajectory, string trajfile, int skip_segments)
{
  //Number of cols = 14; We care abt first 7
  ifstream ifile(trajfile);
  string temp;
  tf::Vector3 current_pos, origin;
  geometry_msgs::Vector3 xs_std;
  tf::Matrix3x3 origin_yaw;
  trajectory.N = 0;
  double temp_d;
  while(!ifile.eof())
  {
      gcop_comm::State current_state;
      for(int count = 0; count < skip_segments+1; count++)
      {
        if(!getline(ifile,temp))
            goto RETURN_FUNC2;
        if(trajectory.N == 0)
          break;
      }
      stringstream ss(temp);
      ss>>temp_d;//time;
      if(temp_d < 0)
      {
        cout<<"Temp_d"<<temp_d<<endl;
        continue;
      }
      
      trajectory.N++;
      trajectory.time.push_back(temp_d);
      ss>>current_pos[0]>>current_pos[1]>>current_pos[2];
      if(trajectory.N == 1)
      {
          origin = current_pos;
          for(int i =0; i < 3; i++)
            ss>>temp_d;
          cout<<"Yaw: "<<temp_d<<endl;
          origin_yaw.setEulerZYX(-temp_d,0,0);
      }
      current_pos = origin_yaw*(current_pos - origin);
      tf::vector3TFToMsg(current_pos,current_state.basepose.translation);
      trajectory.statemsg.push_back(current_state);
      //DEBUG:
      cout<<"xs: "<<current_pos[0]<<" "<<current_pos[1]<<" "<<current_pos[2]<<endl;
  }
RETURN_FUNC2:
  trajectory.N= trajectory.N - 1;
}

void getMPCCtrlTrajectory(gcop_comm::CtrlTraj &trajectory, string trajfile, int skip_segments)
{
  ifstream ifile(trajfile);
  string temp;
  gcop_comm::Stdev xs_std;
  geometry_msgs::Vector3 rpy_std;
  tf::Vector3 rpy;
  tf::Quaternion quat;
  trajectory.N = 0;
  double temp_d;
  while(!ifile.eof())
  {
      gcop_comm::State current_state;
      for(int count = 0; count < skip_segments+1; count++)
      {
        if(!getline(ifile,temp))
            goto RETURN_FUNC;
        if(trajectory.N == 0)
          break;
      }
      stringstream ss(temp);
      ss>>temp_d;//time;
      trajectory.N++;
      trajectory.time.push_back(temp_d);
      ss>>current_state.basepose.translation.x>>current_state.basepose.translation.y>>current_state.basepose.translation.z;
      for(int i = 0; i < 3; i++)
        ss>>temp_d;
      //RPY:
      for(int i = 0; i < 3; i++)
        ss>>rpy[i];
      quat.setEulerZYX(rpy[2],rpy[1],rpy[0]);
      tf::quaternionTFToMsg(quat,current_state.basepose.rotation);
      for(int i = 0; i < 7; i++)
          ss>>temp_d;

      ss>>xs_std.scale_std.x>>xs_std.scale_std.y>>xs_std.scale_std.z;
      xs_std.scale_std.x *= 4;
      xs_std.scale_std.y *= 4;
      xs_std.scale_std.z *= 4;
      if(new_dataset)
      {
        ss>>rpy_std.x>>rpy_std.y>>rpy_std.z;
        tf::Quaternion qt;
        qt.setEulerZYX(rpy_std.z, rpy_std.y, rpy_std.x);//Ypr
        tf::quaternionTFToMsg(qt,xs_std.rot_std);
      }
      else
      {
        xs_std.rot_std.w = 1.0;
        xs_std.rot_std.x = xs_std.rot_std.y = xs_std.rot_std.z = 0.0;//Aligned with global axis
      }
      trajectory.statemsg.push_back(current_state);
      trajectory.pos_std.push_back(xs_std);
      cout<<"Data: "<<trajectory.N<<" "<<current_state.basepose.translation.x<<" "<<current_state.basepose.translation.y<<" "<<current_state.basepose.translation.z<<" "<<xs_std.scale_std.x<<" "<<xs_std.scale_std.y<<" "<<xs_std.scale_std.z<<endl;
  }
RETURN_FUNC:
  trajectory.N = trajectory.N - 1;
}

void timerCallback(const ros::TimerEvent &event, string trajfile, bool mpcmode, int skip_segments)
{
  double obs[8] = {0.8, 1.5,0,0, 0,0,1,0};
  //double obs[8] = {0.5, 1,0,0, 0,0,1,0};
  //double obs[8] = {0.3, 2,1.1,0, 0,0,1,0};
  visualizer_->publishObstacle(obs,1,obs[7]);
  gcop_comm::CtrlTraj trajectory;
  if(mpcmode)
  {
    getMPCCtrlTrajectory(trajectory,trajfile,skip_segments);
    cout<<"traj.N: "<<trajectory.N<<endl;
    visualizer_->publishLineStrip(trajectory);
    visualizer_->publishStdev(trajectory);
  }
  else
  {
    getmeasurementCtrlTrajectory(trajectory,trajfile,skip_segments);
    visualizer_->publishLineStrip(trajectory);
  }
}

int main(int argc, char** argv)
{
    //Initialization
    ros::init(argc, argv, "plot_traj");
    ros::NodeHandle nh;
    visualizer_ = new GcopTrajectoryVisualizer(nh,"world",false);
    //Params
    string trajfile, dirname;
    int skip_segments, id;
    bool mpcmode;
    //Get Params for files
    nh.getParam("/trajfile",trajfile);
    nh.getParam("/dirname",dirname);
    nh.param<int>("/skip_segments", skip_segments,5);
    nh.param<bool>("/mpcmode",mpcmode,true);
    nh.param<bool>("/new_dataset",new_dataset,true);
    nh.param<int>("/id",id,1);
    trajfile = dirname + "/"+trajfile;
    if(!mpcmode)
    {
      double r,g,b;
      nh.param<double>("r",r,1);
      nh.param<double>("g",g,1);
      nh.param<double>("b",b,0);
      visualizer_->setColorLineStrip(r,g,b);
      visualizer_->setID(id);
    }
    nh.setParam("/id",id+1);//Increase id

    oneshot_timer = nh.createTimer(ros::Duration(2),boost::bind(&timerCallback,_1,trajfile,mpcmode,skip_segments),true);
    oneshot_timer.start();
    ros::spin();
}

