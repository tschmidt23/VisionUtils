#pragma once

/***
 * © Tanner Schmidt 2018
 */

#include <Eigen/SVD>

#include <sophus/se3.hpp>

#include <NDT/Tensor.h>

#include <vu/EigenHelpers.h>

namespace vu {

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 3> ComputeFundamentalMatrix(const NDT::Vector <Vec2<Scalar>> & sourcePoints,
                                                     const NDT::Vector <Vec2<Scalar>> & targetPoints);

template <typename Scalar>
Sophus::SE3<Scalar> RelativePoseFromFundamentalMatrix(const Eigen::Matrix<Scalar, 3, 3> & F,
                                                      const Eigen::Matrix<Scalar, 3, 3> & K,
                                                      const NDT::Vector<Vec2<Scalar> > & raysSource,
                                                      const NDT::Vector<Vec2<Scalar> > & raysDestination);

} // namespace vu