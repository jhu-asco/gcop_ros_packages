/*
 * xmlrcp_eig_conv.h
 *
 *  Created on: Aug 27, 2015
 *      Author: subhransu
 */

#ifndef XMLRCP_EIG_CONV_H_
#define XMLRCP_EIG_CONV_H_
#include <XmlRpcValue.h>
#include <Eigen/Dense>

/**
 * converts XmlRpc::XmlRpcValue to Eigen::Matrix<double,r,c> type
 * @param mat: is an Eigen::Matrix(with static rows and cols)
 * @param my_list
 */
template<typename T>
void xml2Mat(T &mat, XmlRpc::XmlRpcValue &my_list)
{
  assert(my_list.getType() == XmlRpc::XmlRpcValue::TypeArray);
  assert(my_list.size() > 0);
  assert(mat.size()==my_list.size());

  for (int i = 0; i < mat.rows(); i++)
  {
    for(int j=0; j<mat.cols();j++)
    {
      int k = j+ i*mat.cols();
      assert(my_list[k].getType() == XmlRpc::XmlRpcValue::TypeDouble);
      mat(i,j) =  (double)(my_list[k]);
    }
  }
}

template<typename T>
string xml2StringMat(T &mat, XmlRpc::XmlRpcValue &my_list)
{
  assert(my_list.getType() == XmlRpc::XmlRpcValue::TypeArray);
  assert(my_list.size() > 0);
  if(my_list.size()==1)
    return static_cast<string>(my_list[0]);
  else if(mat.size()!=my_list.size()-1)
    return "invalid";
  else
  {
    for (int i = 0; i < mat.rows(); i++)
    {
      for(int j=0; j<mat.cols();j++)
      {
        int k = j+ i*mat.cols();
        assert(my_list[k+1].getType() == XmlRpc::XmlRpcValue::TypeDouble);
        mat(i,j) =  (double)(my_list[k+1]);
      }
    }
    return static_cast<string>(my_list[0]);
  }
}
/**
 * Converts a XmlRpc::XmlRpcValue to a Eigen::Vector(dynamic type)
 * @param vec
 * @param my_list
 */
void xml2vec(VectorXd &vec, XmlRpc::XmlRpcValue &my_list)
{
  assert(my_list.getType() == XmlRpc::XmlRpcValue::TypeArray);
  assert(my_list.size() > 0);
  vec.resize(my_list.size());

  for (int32_t i = 0; i < my_list.size(); i++)
  {
    assert(my_list[i].getType() == XmlRpc::XmlRpcValue::TypeDouble);
    vec[i] =  (double)(my_list[i]);
  }
}




#endif /* XMLRCP_EIG_CONV_H_ */
