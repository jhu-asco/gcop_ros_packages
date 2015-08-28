/*
 * yaml_eig_cov.h
 *
 *  Created on: Aug 27, 2015
 *      Author: subhransu
 */

#ifndef GCOP_ROS_EST_INCLUDE_YAML_EIG_COV_H_
#define GCOP_ROS_EST_INCLUDE_YAML_EIG_COV_H_
#include<yaml-cpp/yaml.h> //using yaml version > 0.5
#include <Eigen/Dense>

using namespace Eigen;

namespace YAML
{
template<typename T>
struct convert<pair<string,T>>
{
  static Node encode(const pair<string,T>& string_mat) {
    Node node;
    node.push_back(string_mat.first);
    T& mat = string_mat.second;
    for (int i = 0; i < mat.rows(); i++)
      for(int j=0; j<mat.cols();j++)
        node.push_back(mat(i,j));
    return node;
  }

  static bool decode(const Node& node, pair<string,T>& string_mat)
  {

    if(!node.IsSequence() )
      return false;
    else if(node.size()==1)
    {
      string_mat.first = node[0].as<string>();
      string_mat.second.setZero();
      return true;
    }
    else if(node.size()-1 != string_mat.second.size())
      return false;

    string_mat.first = node[0].as<string>();
    T& mat = string_mat.second;
    for (int i = 0; i < mat.rows(); i++)
    {
      for(int j = 0; j < mat.cols(); j++)
      {
        int k = j+ i*mat.cols();
        mat(i,j) =  node[k+1].as<double>();
      }
    }
    return true;
  }
};

template<>
struct convert<VectorXd>
{
  static Node encode(const VectorXd& vec)
  {
    Node node;
    for (int32_t i = 0; i < vec.size(); i++)
      node.push_back(vec(0));
    return node;
  }

  static bool decode(const Node& node, VectorXd& vec)
  {
    vec.resize(node.size());
    for (int32_t i = 0; i < node.size(); i++)
        vec[i] =  node[i].as<double>();
    return true;
  }
};


template<int m, int n>
struct convert<Matrix<double,m,n>>
{
  static Node encode(const Matrix<double,m,n>& mat)
  {
    Node node;
    for (int i = 0; i < mat.rows(); i++)
      for(int j=0; j<mat.cols();j++)
        node.push_back(mat(i,j));
    return node;

  }

  static bool decode(const Node& node, Matrix<double,m,n>& mat)
  {

    if(!node.IsSequence() )
      return false;
    else if(node.size()!= mat.size())
      return false;

    for (int i = 0; i < mat.rows(); i++)
    {
      for(int j = 0; j < mat.cols(); j++)
      {
        int k = j+ i*mat.cols();
        mat(i,j) =  node[k].as<double>();
      }
    }
    return true;
  }
};
}


#endif /* GCOP_ROS_EST_INCLUDE_YAML_EIG_COV_H_ */
