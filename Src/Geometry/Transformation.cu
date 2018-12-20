#include <vu/Geometry/Transformation.h>

namespace vu {

template <typename Scalar, int D>
void Transform(const Sophus::SE3<Scalar> & transform,
               const NDT::ConstDeviceTensor<D, Vec3<Scalar> > & source,
               NDT::DeviceTensor<D, Vec3<Scalar> > & destination) {

    thrust::transform(source.Data(), source.Data() + source.Count(),
                      destination.Data(),
                      [&transform](const Vec3<Scalar> & vertex) {
                          return transform * vertex;
                      });

}

template void Transform<float, 1>(const Sophus::SE3f &, const NDT::ConstDeviceVector<Vec3<float> > &, NDT::DeviceVector<Vec3<float> > &);
template void Transform<double, 1>(const Sophus::SE3d &, const NDT::ConstDeviceVector<Vec3<double> > &, NDT::DeviceVector<Vec3<double> > &);

template void Transform<float, 2>(const Sophus::SE3f &, const NDT::ConstDeviceImage<Vec3<float> > &, NDT::DeviceImage<Vec3<float> > &);
template void Transform<double, 2>(const Sophus::SE3d &, const NDT::ConstDeviceImage<Vec3<double> > &, NDT::DeviceImage<Vec3<double> > &);


} // namespace vu