#pragma  once

/***
 * © Tanner Schmidt 2018
 */

#include <NDT/Tensor.h>

namespace vu {

template <typename Scalar>
NDT::ManagedVector<Scalar> && GenerateHalfGaussianKernel(const Scalar sigma, const int nSigmas);

template <typename Scalar>
void RadiallySymmetricBlur(const NDT::Image<Scalar> & input,
                           const NDT::Vector<Scalar> & halfKernel,
                           NDT::Image<Scalar> & tmp,
                           NDT::Image<Scalar> & output);

} // namespace vu