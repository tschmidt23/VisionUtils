#pragma  once

/***
 * © Tanner Schmidt 2018
 */

#include <NDT/Tensor.h>

namespace vu {

template <typename Scalar>
void GenerateHalfGaussianKernel(NDT::ManagedVector<Scalar> & kernel, const Scalar sigma, const int nSigmas = 2);

template <typename Scalar, typename KernelScalar>
void RadiallySymmetricBlur(const NDT::Image<Scalar> & input,
                           const NDT::Vector<KernelScalar> & halfKernel,
                           NDT::Image<Scalar> & tmp,
                           NDT::Image<Scalar> & output);

} // namespace vu