/*
 * llh_enu_cov.h
 *
 *  Created on: Aug 20, 2015
 *      Author: subhransu
 */

#ifndef LLH_ENU_COV_H_
#define LLH_ENU_COV_H_

extern "C" {
  #include "libswiftnav/coord_system.h"
}
#include <cmath>
#include <Eigen/Dense>
using namespace Eigen;

void llhSI2EnuSI(Vector3d& enu,const Vector3d& llh,const Vector3d& llh0)
{
  double ecef[3];  wgsllh2ecef(llh.data(), ecef);
  double ecef0[3]; wgsllh2ecef(llh0.data(), ecef0);
  double ned[3];   wgsecef2ned_d(ecef, ecef0, ned);

  enu << ned[1],ned[0],-ned[2];
}

void enuSI2llhSI(Vector3d& llh_rrm, const Vector3d& llh0_rrm, const Vector3d& enu)
{
  double ned[3] = { enu(1), enu(0), -enu(2) };
  double ecef[3]; wgsned2ecef_d(ned, llh0_rrm.data(), ecef);

  wgsecef2llh(ecef, llh_rrm.data());
}

void enuSI2llhDDM(Vector3d& llh_ddm, const Vector3d& llh0_ddm, const Vector3d& enu)
{

  Vector3d llh0_rrm; llh0_rrm.head<2>() = llh0_ddm.head<2>() * M_PI/180.0;
  Vector3d llh_rrm;  enuSI2llhSI(llh_rrm,llh0_rrm,enu);
  llh_ddm.head<2>() = llh_rrm.head<2>()*180.0/M_PI;
  llh_ddm(2) = llh_rrm(2);
}

#endif /* LLH_ENU_COV_H_ */
