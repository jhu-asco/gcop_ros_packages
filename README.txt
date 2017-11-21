For Building these packages, you need to install gcop into the system.

For documentation of individual packages look into their respective folders

For compiling gcop_bullet_systems, you have to specify the GCOP_SOURCE_DIR in the file gcop_ros_bullet/cmake/GcopBullet.cmake

Change the variable to the actual source directory where GCOP is installed (For example $HOME/projects/gcop)

If you do not want to compile gcop_ros_bullet, add a CATKIN_IGNORE file in the folder gcop_ros_bullet and the package will not be compiled

For hrotor_ovs, you must build cv_bridge from source using OpenCV 2.4.11

Outdated Packages:
1. gcop_ros_est, gcop_ros_utils - Because of dependency on enu
