#pragma once

#include <sophus/se3.hpp>

#include <NDT/Tensor.h>

#include <vu/EigenHelpers.h>

namespace vu {

template <typename Scalar, int D>
void Transform(const Sophus::SE3<Scalar> & transform,
               const NDT::ConstTensor<D, Vec3<Scalar> > & source,
               NDT::Tensor<D, Vec3<Scalar> > & destination);


template <typename Scalar, int D>
void Transform(const Sophus::SE3<Scalar> & transform,
               const NDT::ConstDeviceTensor<D, Vec3<Scalar> > & source,
               NDT::DeviceTensor<D, Vec3<Scalar> > & destination);

} // namespace vu