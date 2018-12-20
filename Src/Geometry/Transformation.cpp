#include <vu/Geometry/Transformation.h>

namespace vu {

template <typename Scalar, int D>
void Transform(const Sophus::SE3<Scalar> & transform,
               const NDT::ConstTensor<D, Vec3<Scalar> > & source,
               NDT::Tensor<D, Vec3<Scalar> > & destination) {

    std::transform(source.Data(), source.Data() + source.Count(),
                   destination.Data(),
        [&transform](const Vec3<Scalar> & vertex) {
        return transform * vertex;
    });

}

template void Transform<float, 1>(const Sophus::SE3f &, const NDT::ConstVector<Vec3<float> > &, NDT::Vector<Vec3<float> > &);
template void Transform<double, 1>(const Sophus::SE3d &, const NDT::ConstVector<Vec3<double> > &, NDT::Vector<Vec3<double> > &);

template void Transform<float, 2>(const Sophus::SE3f &, const NDT::ConstImage<Vec3<float> > &, NDT::Image<Vec3<float> > &);
template void Transform<double, 2>(const Sophus::SE3d &, const NDT::ConstImage<Vec3<double> > &, NDT::Image<Vec3<double> > &);


} // namespace vu