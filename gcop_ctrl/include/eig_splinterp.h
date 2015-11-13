/*
 * eig_splinterp.h
 *
 *  Created on: Nov 13, 2015
 *      Author: subhransu
 */

#ifndef EIG_SPLINTERP_H_
#define EIG_SPLINTERP_H_
#include <Eigen/Core>
#include <unsupported/Eigen/Splines>

#include <iostream>

class SplineFunction {
public:
  SplineFunction(Eigen::VectorXd const &x_vec,
                 Eigen::VectorXd const &y_vec,int degree)
    : x_min(x_vec.minCoeff()),
      x_max(x_vec.maxCoeff()),
      // Spline fitting here. X values are scaled down to [0, 1] for this.
      spline_(Eigen::SplineFitting<Eigen::Spline<double, 1>>::Interpolate(
                y_vec.transpose(),
                 // No more than cubic spline, but accept short vectors.

                std::min<int>(x_vec.rows() - 1, degree),
                scaled_values(x_vec)))
  { }

  double operator[](double x) const {
    // x values need to be scaled down in extraction as well.
    return spline_(scaled_value(x))(0);
  }

private:
  // Helpers to scale X values down to [0, 1]
  double scaled_value(double x) const {
    return (x - x_min) / (x_max - x_min);
  }

  Eigen::RowVectorXd scaled_values(Eigen::VectorXd const &x_vec) const {
    return x_vec.unaryExpr([this](double x) { return scaled_value(x); }).transpose();
  }

  double x_min;
  double x_max;

  // Spline of one-dimensional "points."
  Eigen::Spline<double, 1> spline_;
};

#endif /* EIG_SPLINTERP_H_ */
