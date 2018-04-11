/***
 * © Tanner Schmidt 2018
 */

#include <vu/Image/Filtering.h>

#include <vu/EigenHelpers.h>

namespace vu {

template <typename Scalar>
void GenerateHalfGaussianKernel(NDT::ManagedVector<Scalar> & kernel, const Scalar sigma, const int nSigmas) {

    kernel.Resize(1 + std::ceil(nSigmas * sigma));

    Scalar sum(0);

    const Scalar oneOverSigmaSquared = 1 / (sigma * sigma);

    for (int i = 0; i < kernel.Length(); ++i) {

        kernel(i) = std::exp(- i * i * oneOverSigmaSquared);

        sum += kernel(i);

    }

    const Scalar oneOverSum = 1 / sum;

    std::transform(kernel.Data(), kernel.Data() + kernel.Length(), kernel.Data(), [oneOverSum](const Scalar val) {
        return val * oneOverSum;
    });

}

template void GenerateHalfGaussianKernel(NDT::ManagedVector<float> &, const float, const int);

template <typename Scalar, typename KernelScalar>
void RadiallySymmetricBlur(const NDT::Image<Scalar> & input,
                           const NDT::Vector<KernelScalar> & halfKernel,
                           NDT::Image<Scalar> & tmp,
                           NDT::Image<Scalar> & output) {

    for (int y = 0; y < input.Height(); ++y) {

        for (int x = 0; x < input.Width(); ++x) {

            Scalar sum = halfKernel(0) * input(x, y);

            for (int k = 1; k < halfKernel.Length(); ++k) {

                sum += halfKernel(k) * (input(std::max(0, x - k), y) + input(std::min(static_cast<int>(input.Width()) - 1, x + k), y));

            }

            tmp(x, y) = sum;

        }

    }

    for (int y = 0; y < input.Height(); ++y) {

        for (int x = 0; x < input.Width(); ++x) {

            Scalar sum = halfKernel(0) * tmp(x, y);

            for (int k = 1; k < halfKernel.Length(); ++k) {

                sum += halfKernel(k) * (tmp(x, std::max(0, y - k)) + tmp(x, std::min(static_cast<int>(input.Height()) - 1, y + k)));

            }

            output(x, y) = sum;

        }

    }

}

template void
RadiallySymmetricBlur(const NDT::Image<float> &, const NDT::Vector<float> &, NDT::Image<float> &, NDT::Image<float> &);

template void
RadiallySymmetricBlur(const NDT::Image<Vec2<float> > &, const NDT::Vector<float> &, NDT::Image<Vec2<float> > &, NDT::Image<Vec2<float> > &);

template void
RadiallySymmetricBlur(const NDT::Image<Vec3<float> > &, const NDT::Vector<float> &, NDT::Image<Vec3<float> > &, NDT::Image<Vec3<float> > &);


} // namespace vu