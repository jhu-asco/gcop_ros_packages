#include <iostream>
#include <fstream>
#include <gcop/bulletrccar.h>
#include <gcop/bulletworld.h>

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <std_msgs/Float64.h>
#include <sensor_msgs/JointState.h>
#include <visualization_msgs/Marker.h>
#include <XmlRpcValue.h>
#include <tf/transform_broadcaster.h>

using namespace std;
using namespace Eigen;
using namespace gcop;

boost::shared_ptr<Bulletrccar> sys;///Bullet rccar system
boost::shared_ptr<BaseSystem> base_sys;///Bullet rccar system

int main(int argc, char** argv)
{
  ros::init(argc, argv, "cecartest");

  ros::NodeHandle nh("~");

  ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("mesh",1);
	ros::Publisher joint_pub = nh.advertise<sensor_msgs::JointState>("/joint_states", 1);

  //tf:
  tf::TransformBroadcaster broadcaster;

  sensor_msgs::JointState joint_state;///< Joint states for wheels in animation
  //initializing joint msg for animation
  joint_state.name.resize(3);
  //joint_state.header.frame_id = "movingcar";//no namespace for now since only one car present
  joint_state.position.resize(3);
  joint_state.name[0] = "base_to_frontwheel1";
  joint_state.name[1] = "base_to_frontwheel2";
  joint_state.name[2] = "base_to_backwheel1";

  //Create a visualization Marker for Mesh:
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time();
  marker.id = 0;
  marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = 0;
  marker.pose.position.y = 0;
  marker.pose.position.z = 0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 1;
  marker.scale.y = 1;
  marker.scale.z = 1;
  marker.color.a = 1.0; // Don't forget to set the alpha!
  marker.color.r = 0.096;
  marker.color.g = 0.199;
  marker.color.b = 0.0;

  //Create Bullet world and rccar system:
  BulletWorld world(true);//Set the up axis as z for this world
  //Create Bullet rccar:
  sys.reset(new Bulletrccar(world));
  base_sys = boost::static_pointer_cast<BaseSystem>(sys);

  //Load Ground
  std::string filename;
  nh.getParam("mesh",filename);
  cout<<"Filename: "<<filename<<endl;
  btTriangleIndexVertexArray *vertexarray;
  //btCollisionShape* groundShape;
  if(filename.compare("plane") == 0)
  {
    cout<<"We want a mesh file"<<endl;
    return 0;
  }
  else
  {
    vertexarray = world.CreateVertexArrayFromSTL(filename.c_str());
  }
  //Get IndexedMesh array of the vertex array:
  //Print the number of subparts:
  cout<<"Number of subparts in mesh: "<<vertexarray->getNumSubParts()<<endl;
  //btAlignedObjectArray< btIndexedMesh >  &indexedmesharray = vertexarray->getIndexedMeshArray();

  //Create static triangle mesh shape with the array:
  btVector3 mesh_min(-5,-5,-5);
  btVector3 mesh_max(5,5,5);
  btBvhTriangleMeshShape *groundShape = new btBvhTriangleMeshShape(vertexarray, true,mesh_min, mesh_max);

  //Add Rigid Body with above collision shape
  btTransform tr;
  tr.setOrigin(btVector3(0, 0, 0));
  tr.setRotation(btQuaternion(0,0,0));
  world.LocalCreateRigidBody(0,tr, (btCollisionShape*)groundShape);

  Vector4d x0(1,1,0,0);
  sys->Reset(x0,0);
  Vector2d u(0,0);
  double deltat = 0.1, finaltime = 1;
  for(int count1 = 0; count1 < (finaltime/deltat); count1++)
  {
    base_sys->Step(u, deltat);//0.1 seconds per step so running for 1 second
    cout<<"ce->sys.x" <<(deltat*(count1+1))<<"\txs: "<<(sys->x).transpose()<<endl;//#DEBUG
  }
  //Reset the car and modify the ground to see what happens:

  //Modifying the ground using the indexed mesh array:
  //btIndexedMesh &indexmesh = indexedmesharray[0];

    for(int count_trials = 0; count_trials < 5; count_trials++)
    {
      //Get Locked access to vertex array for deforming the mesh:
      unsigned char * vertexbase[1];
      int numverts;
      int vertexStride;
      PHY_ScalarType verticestype;
      unsigned char * indexbase[1];
      int indexstride;
      int numfaces;
      PHY_ScalarType indicestype;
      int subpart = 0;

      vertexarray->getLockedVertexIndexBase (vertexbase, numverts, verticestype, vertexStride, indexbase, indexstride, numfaces, indicestype, subpart);
      unsigned char *vertexbase_sub = vertexbase[0];//Base corresponding to the actual vertex base
      unsigned char *indexbase_sub = indexbase[0];//Base corresponding to the actual index base
      //Print stuff:
      cout<<"Number of vertices: "<<numverts<<endl;
      cout<<"vertexStride: "<<vertexStride<<endl;
      cout<<"indexstride: "<<indexstride<<endl;
      cout<<"numfaces: "<<numfaces<<endl;
      marker.points.resize(3*numfaces);
      getchar();

      //Looping through vertices:
      btAlignedObjectArray< btScalar > vertex_array;;
      for(int count_vertices = 0; count_vertices < numverts; count_vertices++)
      {
        //Get the current vertex(x,y,z):
        btScalar vert_temp[3];
        ///////This part is not needed for true mesh as we are given vert_temp[3]///
        memcpy(vert_temp,&vertexbase_sub[count_vertices*3*sizeof(btScalar)],3*sizeof(btScalar));// 3 doubles per vertex
        cout<<"Current vertex: "<<vert_temp[0]<<"\t"<<vert_temp[1]<<"\t"<<vert_temp[2]<<"\t"<<endl;
        vert_temp[2] -= 0.4;//Reduce height of mesh and see what happens
        //////
        //Copy back the new vertices into vertexBase:
        memcpy(&vertexbase_sub[count_vertices*3*sizeof(btScalar)], vert_temp, 3*sizeof(btScalar));// 3 doubles per vertex

        vertex_array.push_back(vert_temp[0]);
        vertex_array.push_back(vert_temp[1]);
        vertex_array.push_back(vert_temp[2]);
      }
      //Display Loop:
      for(int count_faces = 0; count_faces < numfaces; count_faces++)
      {
        int pointindex[3];
        memcpy(pointindex,&indexbase_sub[3*count_faces*sizeof(int)],3*sizeof(int));
        //create point array:
        for(int count_vert = 0; count_vert < 3; count_vert++)
        {
          marker.points[3*count_faces+count_vert].x = vertex_array[3*pointindex[count_vert]   ];
          marker.points[3*count_faces+count_vert].y = vertex_array[3*pointindex[count_vert]+1 ];
          marker.points[3*count_faces+count_vert].z = vertex_array[3*pointindex[count_vert]+2 ];
        }
      }

      vertexarray->unLockVertexBase(subpart);
      marker_pub.publish(marker);
      ros::spinOnce();
      //Refit the static mesh shape:
      groundShape->partialRefitTree(mesh_min, mesh_max);

      cout<<"Running system after modifying mesh"<<endl;
      sys->Reset(x0,0);
      for(int count1 = 0; count1 < (finaltime/deltat); count1++)
      {
        base_sys->Step(u, deltat);
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
        broadcaster.sendTransform(tf::StampedTransform(tf_chassistransform, ros::Time::now(), "world", "base_link"));
        cout<<"ce->sys.x" <<(deltat*(count1+1))<<"\txs: "<<(sys->x).transpose()<<endl;//#DEBUG
        ros::spinOnce();
        usleep(deltat*1e6);//microseconds
      }
    }

  /*    btTransform tr;
        tr.setOrigin(btVector3(0, 0, 0));
        tr.setRotation(btQuaternion(0,0,0));
        world.LocalCreateRigidBody(0,tr, groundShape);
   */
  return 0;
}


