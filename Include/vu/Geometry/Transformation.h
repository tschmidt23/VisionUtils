#pragma once

#include <NDT/Tensor.h>

#include <vu/EigenHelpers.h>

namespace vu {

template <typename TransformT, typename Scalar, int D>
void Transform(const TransformT & transform,
               const NDT::ConstTensor<D, Vec3<Scalar> > & source,
               NDT::Tensor<D, Vec3<Scalar> > & destination);


template <typename TransformT, typename Scalar, int D>
void Transform(const TransformT & transform,
               const NDT::ConstDeviceTensor<D, Vec3<Scalar> > & source,
               NDT::DeviceTensor<D, Vec3<Scalar> > & destination);

} // namespace vu