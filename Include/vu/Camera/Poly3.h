#pragma once

/***
 * © Tanner Schmidt 2018
 */

#include <string>
#include <vu/Camera/CameraModel.h>
#include <vu/Camera/CameraFactory.h>

namespace vu {

template <typename Scalar>
class Poly3CameraModel;

namespace internal {

template <>
struct CameraModelTraits<Poly3CameraModel> {

    static constexpr int NumParams = 7;

};

} // namespace internal

template <typename Scalar>
class Poly3CameraModel : public CameraModel<Poly3CameraModel, Scalar> {
public:

    Poly3CameraModel() {}

    Poly3CameraModel(const picojson::value & cameraSpec)
        : CameraModel<Poly3CameraModel,Scalar>(cameraSpec) { }

    template <typename T2>
    Poly3CameraModel(const CameraModel<Poly3CameraModel,T2> & other)
        : CameraModel<Poly3CameraModel,Scalar>(other) { }

    Poly3CameraModel(const Eigen::Matrix<Scalar,7,1,Eigen::DontAlign> & params)
        : CameraModel<Poly3CameraModel,Scalar>(params) { }

    inline std::string ModelName() const {

        return "Poly3";

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

    inline __host__ __device__ Scalar k1() const {
        return this->Params()[4];
    }

    inline __host__ __device__ Scalar k2() const {
        return this->Params()[5];
    }

    inline __host__ __device__ Scalar k3() const {
        return this->Params()[6];
    }

    inline __host__ __device__ Eigen::Matrix<Scalar,2,1> FocalLength() const {
        return Eigen::Matrix<Scalar,2,1>(FocalLengthX(), FocalLengthY());
    }

    inline __host__ __device__ Eigen::Matrix<Scalar,2,1> PrincipalPoint() const {
        return Eigen::Matrix<Scalar,2,1>(PrincipalPointX(), PrincipalPointY());
    }

    inline Eigen::Matrix<Scalar,2,1> __host__ __device__ Project(const Eigen::Matrix<Scalar,3,1> point3d) const {

        const Eigen::Matrix<Scalar,2,1> dehomog = this->Dehomogenize(point3d);

        const Scalar radius2 = dehomog.squaredNorm();

        const Scalar radius4 = radius2 * radius2;

        const Scalar radius6 = radius4 * radius2;

        const Scalar distortionFactor = Scalar(1) + k1()*radius2 + k2()*radius4 + k3()*radius6;

        const Eigen::Matrix<Scalar,2,1> distorted = distortionFactor * dehomog;

        return this->ApplyFocalLengthAndPrincipalPoint(distorted, FocalLength(), PrincipalPoint());

    }

    template <int Options>
    inline Eigen::Matrix<Scalar,2,3,Options> __host__ __device__ ProjectionDerivative(const Eigen::Matrix<Scalar,3,1,Options> point3d) const {

        const Eigen::Matrix<Scalar,2,1,Options> dehomog = this->Dehomogenize(point3d);

        Eigen::Matrix<Scalar,2,3,Options> dehomogDerivative = this->DehomogenizeDerivative(point3d);

        const Scalar radius = dehomog.norm();

        Scalar factor, factorDerivative;

        this->FactorDerivative(radius, factor, factorDerivative);

        // TODO: this transpose is likely inefficient
        const Eigen::Matrix<Scalar,1,2,Options | Eigen::RowMajor> factorDerivativeByHomog = factorDerivative * dehomog.transpose() / radius;

        Eigen::Matrix<Scalar,2,2,Options> kMult = dehomog * factorDerivativeByHomog;

        kMult(0, 0) += factor;

        kMult(1, 1) += factor;

        kMult.template block<1,2>(0,0) *= this->params_[0];

        kMult.template block<1,2>(1,0) *= this->params_[1];

        return (kMult * dehomogDerivative);

    }

    inline Eigen::Matrix<Scalar,3,1> __host__ __device__ Unproject(const Eigen::Matrix<Scalar,2,1> point2d, const Scalar depth) const {

        const Eigen::Matrix<Scalar,2,1> dehomog = this->UnapplyFocalLengthAndPrincipalPoint(point2d, FocalLength(), PrincipalPoint());

        const Scalar radiusInit = dehomog.norm();

        if (radiusInit > Scalar(0)) {

            Scalar radius = radiusInit;
            for (int i = 0; i < maxUnprojectionIters; ++i) {

                const Scalar radius2 = radius*radius;

                const Scalar radius4 = radius2*radius2;

                const Scalar radius6 = radius4*radius2;

                const Scalar distortionFactor = Scalar(1) + k1()*radius2 + k2()*radius4 + k3()*radius6;

                const Scalar distortionFactor2 = 2*radius2*(k1() + 2*k2()*radius2 + 3*k3()*radius4);

                const Scalar distortionFactor3 = distortionFactor + distortionFactor2;

                const Scalar derivative = (radius * distortionFactor - radiusInit) * 2 * distortionFactor3;

                const Scalar derivative2 = (4 * radius * ( radius * distortionFactor - radiusInit) *
                                            (3 * k1() + 10*k2()*radius2 + 21*k3()*radius4) +
                                            2*distortionFactor3*distortionFactor3);

                const Scalar delta = derivative / derivative2;

                radius -= delta;
            }

            const Scalar undistortionFactor = radius / radiusInit;

            const Eigen::Matrix<Scalar,2,1> undistorted = dehomog*undistortionFactor;

            const Eigen::Matrix<Scalar,2,1> scaled = undistorted * depth;

            return Eigen::Matrix<Scalar,3,1>(scaled(0),scaled(1),depth);

        } else {

            return Eigen::Matrix<Scalar,3,1>(dehomog(0)*depth,dehomog(1)*depth,depth);

        }

    }

    inline Poly3CameraModel<Scalar> DownsampleBy2() const {

        Eigen::Matrix<Scalar,7,1,Eigen::DontAlign> downsampledParams = this->params_;

        downsampledParams.template head<2>() /= Scalar(2);

        downsampledParams.template segment<2>(2) /= Scalar(2);
        downsampledParams.template segment<2>(2) -= Eigen::Matrix<Scalar,2,1>(1 / Scalar(4), 1 / Scalar(4));

        return Poly3CameraModel<Scalar>(downsampledParams);

    }

private:

    static constexpr int maxUnprojectionIters = 5;

    inline __host__ __device__ void FactorDerivative(const Scalar radius, Scalar & factor, Scalar & factorDerivative) const {

        factor = Scalar(1);

        factorDerivative = this->k1() * Scalar(2) * radius;

        Scalar radiusN = radius * radius; // radius^2

        factor += this->k1() * radiusN;

        radiusN *= radius; // radius ^ 3

        factorDerivative += this->k2() * Scalar(4) * radiusN;

        radiusN *= radius; // radius ^ 4

        factor += this->k2() * radiusN;

        radiusN *= radius; // radius ^ 5

        factorDerivative += this->k3() * Scalar(6) * radiusN;

        radiusN *= radius; // radius ^ 6

        factor += this->k3() * radiusN;

    }

};

} // namespace vu
