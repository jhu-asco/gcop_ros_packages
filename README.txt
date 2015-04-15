For Building these packages, you need to install gcop into the system.
For compiling gcop_bullet_systems, you have to specify the GCOP_SOURCE_DIR in the file gcop_ros_bullet/cmake/GcopBullet.cmake
Change the variable to the actual source directory where GCOP is installed (For example ~/projects/gcop)
If you do not want to compile gcop_ros_bullet, add a CATKIN_IGNORE file in the folder gcop_ros_bullet and the package will not be compiled
