#include <vu/Geometry/Transformation.h>

#include <sophus/se3.hpp>

namespace vu {

template <typename TransformT, typename Scalar, int D>
void Transform(const TransformT & transform,
               const NDT::ConstTensor<D, Vec3<Scalar> > & source,
               NDT::Tensor<D, Vec3<Scalar> > & destination) {

    std::transform(source.Data(), source.Data() + source.Count(),
                   destination.Data(),
        [&transform](const Vec3<Scalar> & vertex) {
        return transform * vertex;
    });

}

// for points
template void Transform<Sophus::SE3f, float, 1>(const Sophus::SE3f &, const NDT::ConstVector<Vec3<float> > &, NDT::Vector<Vec3<float> > &);
template void Transform<Sophus::SE3d, double, 1>(const Sophus::SE3d &, const NDT::ConstVector<Vec3<double> > &, NDT::Vector<Vec3<double> > &);

template void Transform<Sophus::SE3f, float, 2>(const Sophus::SE3f &, const NDT::ConstImage<Vec3<float> > &, NDT::Image<Vec3<float> > &);
template void Transform<Sophus::SE3d, double, 2>(const Sophus::SE3d &, const NDT::ConstImage<Vec3<double> > &, NDT::Image<Vec3<double> > &);

// for vectors
template void Transform<Sophus::SO3f, float, 1>(const Sophus::SO3f &, const NDT::ConstVector<Vec3<float> > &, NDT::Vector<Vec3<float> > &);
template void Transform<Sophus::SO3d, double, 1>(const Sophus::SO3d &, const NDT::ConstVector<Vec3<double> > &, NDT::Vector<Vec3<double> > &);

template void Transform<Sophus::SO3f, float, 2>(const Sophus::SO3f &, const NDT::ConstImage<Vec3<float> > &, NDT::Image<Vec3<float> > &);
template void Transform<Sophus::SO3d, double, 2>(const Sophus::SO3d &, const NDT::ConstImage<Vec3<double> > &, NDT::Image<Vec3<double> > &);


} // namespace vu