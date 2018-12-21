#include <vu/Geometry/Transformation.h>

#include <sophus/se3.hpp>

#include <thrust/device_ptr.h>
#include <thrust/transform.h>

namespace vu {

template <typename TransformT, typename Scalar, int D>
void Transform(const TransformT & transform,
               const NDT::ConstDeviceTensor<D, Vec3<Scalar> > & source,
               NDT::DeviceTensor<D, Vec3<Scalar> > & destination) {

    thrust::device_ptr<const Vec3<Scalar> > sourcePointer(source.Data());
    thrust::device_ptr<Vec3<Scalar> > destPointer(destination.Data());

    thrust::transform(sourcePointer, sourcePointer + source.Count(),
                      destPointer,
                      [transform] __device__ (const Vec3<Scalar> & vertex) {
                          return transform * vertex;
                      });

}

// for points
template void Transform<Sophus::SE3f, float, 1>(const Sophus::SE3f &, const NDT::ConstDeviceVector<Vec3<float> > &, NDT::DeviceVector<Vec3<float> > &);
template void Transform<Sophus::SE3d, double, 1>(const Sophus::SE3d &, const NDT::ConstDeviceVector<Vec3<double> > &, NDT::DeviceVector<Vec3<double> > &);

template void Transform<Sophus::SE3f, float, 2>(const Sophus::SE3f &, const NDT::ConstDeviceImage<Vec3<float> > &, NDT::DeviceImage<Vec3<float> > &);
template void Transform<Sophus::SE3d, double, 2>(const Sophus::SE3d &, const NDT::ConstDeviceImage<Vec3<double> > &, NDT::DeviceImage<Vec3<double> > &);


// for vectors
template void Transform<Sophus::SO3f, float, 1>(const Sophus::SO3f &, const NDT::ConstDeviceVector<Vec3<float> > &, NDT::DeviceVector<Vec3<float> > &);
template void Transform<Sophus::SO3d, double, 1>(const Sophus::SO3d &, const NDT::ConstDeviceVector<Vec3<double> > &, NDT::DeviceVector<Vec3<double> > &);

template void Transform<Sophus::SO3f, float, 2>(const Sophus::SO3f &, const NDT::ConstDeviceImage<Vec3<float> > &, NDT::DeviceImage<Vec3<float> > &);
template void Transform<Sophus::SO3d, double, 2>(const Sophus::SO3d &, const NDT::ConstDeviceImage<Vec3<double> > &, NDT::DeviceImage<Vec3<double> > &);


} // namespace vu