# Gcop Bullet package
This package provides different nodes for optimizing an rccar model on 3D terrain/plane using different optimization algorithms.

# Installation
 To compile this package, you have to compile GCOP with USE_BULLET Option and modify the file cmake/GcopConfig.cmake to specify the variable GCOP_SOURCE_DIR 
Change the variable to the actual source directory where GCOP is installed (For example $HOME/projects/gcop). This will allow the package to link to specific version of bullet used by GCOP.

# Nodes and Launch files:

* cecartest.launch: Launch Bullet rccar model on a plane with Cross Entropy based optimization method
* gncartest.launch: Launch Bullet rccar model on a plane with Gauss Newton based optimization method
* sddp_planar_cartest.launch: Launch Bullet rccar model on plane with Sampling based Differential Dynamic Programming (SDDP) optimization method
* ceterraincartest.launch: Launch Bullet rccar model on a pre-built terrain and optimize using CE based optimization method
