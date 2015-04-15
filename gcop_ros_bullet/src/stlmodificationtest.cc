#include <iostream>
#include <fstream>
#include <gcop/bulletrccar.h>
#include <gcop/bulletworld.h>

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <std_msgs/Float64.h>
#include <XmlRpcValue.h>

using namespace std;
using namespace Eigen;
using namespace gcop;

boost::shared_ptr<Bulletrccar> sys;///Bullet rccar system

int main(int argc, char** argv)
{
  ros::init(argc, argv, "cecartest");

  ros::NodeHandle nh("~");

  //Create Bullet world and rccar system:
  BulletWorld world(true);//Set the up axis as z for this world
  //Create Bullet rccar:
  sys.reset(new Bulletrccar(world));

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
  sys->reset(x0,0);
  Vector2d u(0,0);
  for(int count1 = 0; count1 < 10; count1++)
  {
    sys->Step_internaloutput(u, 0.1);//0.1 seconds per step so running for 1 second
    cout<<"ce->sys.x" <<(0.1*(count1+1))<<"\txs: "<<(sys->x).transpose()<<endl;//#DEBUG
  }
  //Reset the car and modify the ground to see what happens:

  //Modifying the ground using the indexed mesh array:
  //btIndexedMesh &indexmesh = indexedmesharray[0];

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
  //Print stuff:
  cout<<"Number of vertices: "<<numverts<<endl;
  cout<<"vertexStride: "<<vertexStride<<endl;
  cout<<"indexstride: "<<indexstride<<endl;

  //Looping through vertices:
  for(int count_vertices = 0; count_vertices < numverts; count_vertices++)
  {
    //Get the current vertex(x,y,z):
    btScalar vert_temp[3];
    ///////This part is not needed for true mesh as we are given vert_temp[3]///
    memcpy(vert_temp,&vertexbase_sub[count_vertices*3*sizeof(btScalar)],3*sizeof(btScalar));// 3 doubles per vertex
    cout<<"Current vertex: "<<vert_temp[0]<<"\t"<<vert_temp[1]<<"\t"<<vert_temp[2]<<"\t"<<endl;
    vert_temp[2] -= 0.5;//Reduce height of mesh and see what happens
    //////
    //Copy back the new vertices into vertexBase:
    memcpy(&vertexbase_sub[count_vertices*3*sizeof(btScalar)], vert_temp, 3*sizeof(btScalar));// 3 doubles per vertex
  }
  
  vertexarray->unLockVertexBase(subpart);
  //Refit the static mesh shape:
  groundShape->partialRefitTree(mesh_min, mesh_max);

  cout<<"Running system after modifying mesh"<<endl;
  sys->reset(x0,0);
  for(int count1 = 0; count1 < 10; count1++)
  {
    sys->Step_internaloutput(u, 0.1);//0.1 seconds per step so running for 1 second
    cout<<"ce->sys.x" <<(0.1*(count1+1))<<"\txs: "<<(sys->x).transpose()<<endl;//#DEBUG
  }

  /*    btTransform tr;
        tr.setOrigin(btVector3(0, 0, 0));
        tr.setRotation(btQuaternion(0,0,0));
        world.LocalCreateRigidBody(0,tr, groundShape);
   */
  return 0;
}


