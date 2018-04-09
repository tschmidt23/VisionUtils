#pragma once

/***
 * © Tanner Schmidt 2018
 */

#include <pangolin/utils/picojson.h>

#include <Eigen/Core>

#include <cuda_runtime.h>

namespace vu {

namespace internal {

template <template <typename> class ModelT>
struct CameraModelTraits;

} // namespace internal

template <template <typename> class ModelT, typename Scalar>
class CameraModel {
public:

    CameraModel() { }

    CameraModel(const picojson::value & cameraSpec) {

        if (cameraSpec["params"].size() != NumParams()) {
            throw std::runtime_error("wrong number of parameters for specifiec model (" +
                                     std::to_string(cameraSpec.size()) + " vs " + std::to_string(NumParams()));
        }

        std::cout << "params: ";
        for (unsigned int i = 0; i < NumParams(); ++i) {
            std::cout << cameraSpec["params"][i] << "  ";
            params_[i] = atof(cameraSpec["params"][i].to_str().c_str());
        } std::cout << std::endl;

    }

    template <typename T2>
    CameraModel(const CameraModel<ModelT,T2> & other) {
        for (unsigned int i = 0; i < NumParams(); ++i) {
            params_[i] = Scalar(other.Params()[i]);
        }
    }

    CameraModel(const Eigen::Matrix<Scalar,internal::CameraModelTraits<ModelT>::NumParams,1,Eigen::DontAlign> & params)
        : params_(params) { }

    template <typename T2>
    inline ModelT<T2> Cast() const {
        return ModelT<T2>(*this);
    }

    inline operator ModelT<Scalar> () { return *this; }

    inline operator const ModelT<Scalar> () const { return *this; }

    inline static unsigned int NumParams() { return internal::CameraModelTraits<ModelT>::NumParams; }

    inline __host__ __device__ const Scalar * Params() const {
        return params_.data();
    }

    inline __host__ __device__ Eigen::Matrix<Scalar,2,1> Project(const Eigen::Matrix<Scalar,3,1> point3d) const {
        return static_cast<const ModelT<Scalar> *>(this)->Project(point3d);
    }

    inline __host__ __device__ Eigen::Matrix<Scalar,3,1> Unproject(const Eigen::Matrix<Scalar,2,1> point2d, const Scalar depth) const {
        return static_cast<const ModelT<Scalar> *>(this)->Unproject(point2d,depth);
    }

    inline __host__ __device__ Eigen::Matrix<Scalar,2,1> Dehomogenize(const Eigen::Matrix<Scalar,3,1> point3d) const {
//        return point3d(2) > Scalar(0) ? Eigen::Matrix<Scalar,2,1>(point3d(0)/point3d(2),
//                                                                  point3d(1)/point3d(2)) :
//            Eigen::Matrix<Scalar,2,1>::Zero();
        return Eigen::Matrix<Scalar,2,1>(point3d(0)/point3d(2),
                                         point3d(1)/point3d(2));
    }

protected:

    inline __host__ __device__ Eigen::Matrix<Scalar,2,3,Eigen::DontAlign> DehomogenizeDerivative(const Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> & point3d) const {

        const Scalar oneOverZ = Scalar(1) / point3d(2);

        const Scalar oneOverZSquared = oneOverZ * oneOverZ;

        Eigen::Matrix<Scalar,2,3,Eigen::DontAlign> J;
        J << oneOverZ, 0,        -point3d(0) * oneOverZSquared,
             0,        oneOverZ, -point3d(1) * oneOverZSquared;

        return J;

    }

    inline __host__ __device__ Eigen::Matrix<Scalar,2,1> ApplyFocalLengthAndPrincipalPoint(const Eigen::Matrix<Scalar,2,1> dehomogPoint,
                                                                  const Eigen::Matrix<Scalar,2,1> focalLength,
                                                                  const Eigen::Matrix<Scalar,2,1> principalPoint) const {
        return Eigen::Matrix<Scalar,2,1>(dehomogPoint(0)*focalLength(0),
                                    dehomogPoint(1)*focalLength(1)) + principalPoint;
    }

    inline __host__ __device__ Eigen::Matrix<Scalar,2,1> UnapplyFocalLengthAndPrincipalPoint(const Eigen::Matrix<Scalar,2,1> point2d,
                                                                    const Eigen::Matrix<Scalar,2,1> focalLength,
                                                                    const Eigen::Matrix<Scalar,2,1> principalPoint) const {
        const Eigen::Matrix<Scalar,2,1> centered = point2d - principalPoint;
        return Eigen::Matrix<Scalar,2,1>(centered(0)/focalLength(0),
                                         centered(1)/focalLength(1));
    }

    Eigen::Matrix<Scalar,internal::CameraModelTraits<ModelT>::NumParams,1,Eigen::DontAlign> params_;

};

} // namespace vu