# Gcop Controller Package
This package provides nodes to control different systems provided by GCOP using ROS interface. The examples in the package are multibody systems with Quadcopters/Fixed serial chains/Rccar models. The package provides different nodes which are described as follows:

mbstest   
--------

Node to test and plan multi body systems using a simple reconfiguration interface. To test the node run the launch file mbs.launch. It should load a quadcopter example with a 2DOF arm. Open rqt_reconfigure in a separate window and you should be able hit iterate to plan a trajectory from start to goal. You can animate it multiple times by pressing animate. There are other utilities in reconfigure to change initial and final states of the quadcopter as well as the arms

![cost function] (./render.png " Cost function ")


## Publishers:
* '/mbsddp/ctrltraj': [gcop_comm/CtrlTraj] Optimal trajectory containing the optimal controls and states published whenever iteration is done through reconfigure interface

## Parameters:
* 'mbsddp/X0' :[vector] Initial state of the multibody state see examples in params folder
* 'mbsddp/XN' :[vector] Final state of the multibody system
* 'mbsddp/Qf' :[Vector] Terminal Cost
* 'mbsddp/Q'  :[Vector] State Cost
* 'mbsddp/R'  :[Vector] Control Cost
* 'mbsddp/ag' :[Vector] Gravity vector
* 'mbsddp/J0' :[Vector] Initial joint angles
* 'mbsddp/JN' :[Vector] Final Joint angles
* 'mbsddp/ulb/uub' :[Vector] Bounds on the control of the base body (Accelerations/torque bounds)
* 'mbsddp/mu' :[double] regularization parameter for DDP
* 'mbsddp/Nit': [int] Number of iterations to be done
* 'mbsddp/mode' :[string] If in "user" mode then you have to press iterate on reconfigure to iterate. Otherwise("continous" mode), the optimization is run using a timer and keeps running using the previous goal position even if you do not give a new goal position

Many of these parameters can also be modified using dynamic reconfigure interface

msbnode
-------

Node to plan multi body systems through ros topics. This allows external nodes to use the controller to plan optimal trajectories between start and goal states. This node is launched through mbsnode.launch

## Publishers:
* '/mbsddp/traj_resp': [gcop_comm/CtrlTraj] Response Optimal trajectory containing the optimal controls and states
* '/mbsddp/desired_traj': [visualization_msgs/Marker] Trajectory published to rviz for visualization
* '/msddp/robotname[]/joint_states': [sensor_msgs/JointState] Publishes joint states at different stages in the trajectory for visualization in rviz

## Subscribers:
* '/mbsddp/iteration_req': [gcop_comm/Iteration_req] Request for iteration of the optimization algorithm from external node

This node shares the same parameters as the above node but cannot reconfigure the parameters. 


rccartest
----------

This node provides an example of optimizing rccar system using DDP. This is similar to mbstest in the sense that it provides all the configuration parameters through rqt_reconfigure and changes the optimal trajectory based on the input goal and car states. This node keeps optimizing the trajectory continously based on timer callback using the current goal state even when new goal states have been not provided.

## Publishers:
* '/ddp/ctrltraj': [gcop_comm/CtrlTraj] Provides the current optimal trajectory whenever iteration is done

## Subscribers:
* '/ddp/mocap': [geometry_msgs/TransformStamped] If use mocap option is turned on in rqt_reconfigure, optimizes the trajectory based on input state from external source in the given format. The message should provide the base link pose in the world frame. This is a 2D car simulation hence will not use roll, pitch, z coordinates

## Parameters:
* 'ddp/tf' :[double] Final time
* 'ddp/N'  :[int] Number of segments in the trajectory
* 'ddp/x0' :[double] Initial state of the car (x position)
* 'ddp/y0' :[double] Initial state of the car (y position)
* 'ddp/vx0' :[double] Initial state of the car (theta initial yaw)
* 'ddp/vy0' :[double] Initial state of the car (v body velocity)
* 'ddp/xN' :[double] Final state of the car (x position)
* 'ddp/yN' :[double] Final state of the car (y position)
* 'ddp/vxN' :[double] Final state of the car (theta initial yaw)
* 'ddp/vyN' :[double] Final state of the car (v body velocity)
* 'ddp/Q[]' :[double] Diagonal entries of the state cost matrix Q (diagonal).
* 'ddp/Qf[]' :[double] Diagonal entries of the terminal cost matrix Q (diagonal).
* 'ddp/R[]' :[double] Diagonal entries of the control cost matrix R (diagonal).

rccarsub
------------

This node provides the optimization of rccar system through a topic based interface similar to mbsnode. This can be used by external nodes to call optimization request with specific target pose and receive optimal trajectories connecting the initial state and target pose

## Publishers:
* '/ddp/ctrltraj': [gcop_comm/CtrlTraj] Publish the optimal trajectory based on optimization using DDP

## Subscribers:
* '/ddp/mocap': [gcop_comm/CurrPose] Loads the initial state of the car  from external nodes which publish this such as odometry etc
* '/ddp/target': [geometry_msgs::PoseStamped] Loads the target pose of the car to optimize to

Parameters same as rccartest
