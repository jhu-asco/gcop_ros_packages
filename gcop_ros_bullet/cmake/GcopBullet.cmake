###############SET This to GCOP Source Directory###############
set(GCOP_SOURCE_DIR /home/subhransu/projects/libraries/gcop/gcop)

set(BULLET_PHYSICS_SOURCE_DIR ${GCOP_SOURCE_DIR}/bullet3)

set(BULLET_INCLUDE_DIR ${BULLET_PHYSICS_SOURCE_DIR}/src)
message("BulletINCLUDE: ${BULLET_INCLUDE_DIR}")

set(BULLET_LIBRARIES "${BULLET_PHYSICS_SOURCE_DIR}/build3/src/BulletDynamics/libBulletDynamics.so;${BULLET_PHYSICS_SOURCE_DIR}/build3/src/BulletCollision/libBulletCollision.so;${BULLET_PHYSICS_SOURCE_DIR}/build3/src/LinearMath/libLinearMath.so") 

include_directories(${BULLET_INCLUDE_DIR})

link_directories( ${BULLET_PHYSICS_SOURCE_DIR}/build3/btgui/OpenGLWindow
                  ${BULLET_PHYSICS_SOURCE_DIR}/build3/btgui/Gwen
                  ${BULLET_PHYSICS_SOURCE_DIR}/build3/btgui/Bullet3AppSupport
                  ${BULLET_PHYSICS_SOURCE_DIR}/build3/btgui/lua-5.2.3
                  ${BULLET_PHYSICS_SOURCE_DIR}/build3/Demos/OpenGL
                )
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBT_USE_DOUBLE_PRECISION")#Compiling Bullet with double precision



