#pragma once

/***
 * © Tanner Schmidt 2018
 */

#include <vu/Camera/CameraModel.h>
#include <vu/Camera/CameraFactory.h>

namespace vu {

template <typename Scalar>
class LinearCameraModel;

namespace internal {

template <>
struct CameraModelTraits<LinearCameraModel> {

    static constexpr int NumParams = 4;

};

} // namespace internal

template <typename Scalar>
class LinearCameraModel : public CameraModel<LinearCameraModel, Scalar> {
public:

    LinearCameraModel(const picojson::value & cameraSpec)
        : CameraModel<LinearCameraModel,Scalar>(cameraSpec) { }

    template <typename Derived, typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 4 &&
                                                        Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                                                        std::is_same<typename Eigen::internal::traits<Derived>::Scalar,Scalar>::value, int>::type = 0>
    LinearCameraModel(const Eigen::MatrixBase<Derived> & params)
        : CameraModel<LinearCameraModel,Scalar>(params) { }

    template <typename T2>
    LinearCameraModel(const CameraModel<LinearCameraModel,T2> & other)
        : CameraModel<LinearCameraModel,Scalar>(other) { }

    inline std::string ModelName() const {

        return "Linear";

    }

    inline __host__ __device__ Scalar FocalLengthX() const {
        return this->Params()[0];
    }

    inline __host__ __device__ Scalar FocalLengthY() const {
        return this->Params()[1];
    }

    inline __host__ __device__ Scalar PrincipalPointX() const {
        return this->Params()[2];
    }

    inline __host__ __device__ Scalar PrincipalPointY() const {
        return this->Params()[3];
    }

    inline __host__ __device__ Eigen::Matrix<Scalar,2,1> FocalLength() const {

        return Eigen::Matrix<Scalar,2,1>(FocalLengthX(), FocalLengthY());

    }

    inline __host__ __device__ Eigen::Matrix<Scalar,2,1> PrincipalPoint() const {

        return Eigen::Matrix<Scalar,2,1>(PrincipalPointX(), PrincipalPointY());

    }

    inline __host__ __device__ Eigen::Matrix<Scalar,2,1> Project(const Eigen::Matrix<Scalar,3,1> point3d) const {

        const Eigen::Matrix<Scalar,2,1> dehomog = this->Dehomogenize(point3d);

        return this->ApplyFocalLengthAndPrincipalPoint(dehomog, FocalLength(), PrincipalPoint());

    }

    inline __host__ __device__ Eigen::Matrix<Scalar,2,3,Eigen::DontAlign> ProjectionDerivative(const Eigen::Matrix<Scalar,3,1,Eigen::DontAlign> point3d) const {

        Eigen::Matrix<Scalar,2,3,Eigen::DontAlign> dehomogDerivative = this->DehomogenizeDerivative(point3d);

        dehomogDerivative(0,0) *= this->params_[0];

        dehomogDerivative(0,2) *= this->params_[0];

        dehomogDerivative(1,1) *= this->params_[1];

        dehomogDerivative(1,2) *= this->params_[1];

        return dehomogDerivative;

    }

    inline __host__ __device__ Eigen::Matrix<Scalar,3,1> Unproject(const Eigen::Matrix<Scalar,2,1> point2d, const Scalar depth) const {

        const Eigen::Matrix<Scalar,2,1> dehomog = this->UnapplyFocalLengthAndPrincipalPoint(point2d, FocalLength(), PrincipalPoint());

        const Eigen::Matrix<Scalar,2,1> scaled = dehomog * depth;

        return Eigen::Matrix<Scalar,3,1>(scaled(0), scaled(1), depth);

    }

protected:

};

} // namespace vu
