# Gcop Controller Package
This package provides nodes to control different systems provided by GCOP using ROS interface. The examples in the package are multibody systems with Quadcopters/Fixed serial chains/Rccar models. The package provides different nodes which are described as follows:

mbstest   
--------

Node to test and plan multi body systems using a simple reconfiguration interface. To test the node run the launch file mbs.launch. It should load a quadcopter example with a 2DOF arm. Open rqt_reconfigure in a separate window and you should be able hit iterate to plan a trajectory from start to goal. You can animate it multiple times by pressing animate. There are other utilities in reconfigure to change initial and final states of the quadcopter as well as the arms

Cost function 
![cost function] (./render.png)


## Publishers:
* '/mbsddp/ctrltraj': [gcop_comm/CtrlTraj] Optimal trajectory containing the optimal controls and states published whenever iteration is done through reconfigure interface

## Parameters:
* 'mbsddp/X0' :[vector] Initial state of the multibody state see examples in params folder
* 'mbsddp/XN' :[vector] Final state of the multibody system
* 'mbsddp/Qf' :[Vector] Terminal Cost 
msbnode
-------

Node to plan multi body systems through ros topics. This allows external nodes to use the controller to plan optimal trajectories between start and goal states. This node is launched through mbsnode.launch

## Publishers:
* '/mbsddp/traj_resp': [gcop_comm/CtrlTraj] Response Optimal trajectory containing the optimal controls and states
* '/mbsddp/desired_traj': [visualization_msgs/Marker] Trajectory published to rviz for visualization
* '/msddp/robotname[]/joint_states': [sensor_msgs/JointState] Publishes joint states at different stages in the trajectory for visualization in rviz

## Subscribers:
* '/mbsddp/iteration_req': [gcop_comm/Iteration_req] Request for iteration of the optimization algorithm from external node


