#pragma once

#include <vu/Optimization/LinearSystems.h>

namespace vu {

namespace internal {

template <typename Derived>
__host__ __device__
inline typename Eigen::internal::traits<Derived>::Scalar GenericNorm(const Eigen::MatrixBase <Derived> & val) {
    return val.norm();
}

__host__ __device__
inline float GenericNorm(const float & val) {
    return fabsf(val);
}

__host__ __device__
inline double GenericNorm(const double & val) {
    return fabs(val);
}

template <typename Derived>
__host__ __device__
inline typename Eigen::internal::traits<Derived>::Scalar GenericSquaredNorm(const Eigen::MatrixBase<Derived> & val) {
    return val.squaredNorm();
}

__host__ __device__
inline float GenericSquaredNorm(const float & val) {
    return val * val;
}

__host__ __device__
inline double GenericeSquaredNorm(const double & val) {
    return val * val;
}

__host__ __device__
inline float GenericSqrt(const float & val) {
    return sqrtf(val);
}

__host__ __device__
inline double GenericSqrt(const double & val) {
    return sqrt(val);
}

} // namespace internal


//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
//                                          Residual Functors
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
template <typename Scalar, int ResidualDim, int ModelDim>
struct ResidualFunctorL2 {

    __attribute__((always_inline)) __host__ __device__
    Scalar operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) const {

        return Scalar(0.5) * jacobianAndResidual.r.squaredNorm();
    }

};

template <typename Scalar, int ModelDim>
struct ResidualFunctorL2<Scalar,1,ModelDim> {

    __attribute__((always_inline)) __host__ __device__
    Scalar operator()(const JacobianAndResidual<Scalar,1,ModelDim> & jacobianAndResidual) const {

        return Scalar(0.5) * jacobianAndResidual.r * jacobianAndResidual.r;

    }

};

template <typename Scalar, int ResidualDim, int ModelDim, typename WeightType>
struct WeightedResidualFunctorL2 {

    WeightedResidualFunctorL2(const WeightType & weight) : weight_(weight) { }

    __attribute__((always_inline)) __host__ __device__
    Scalar operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) const {

        // TODO: I think transpose is slow
        return Scalar(0.5) * jacobianAndResidual.r.transpose() * weight_ * jacobianAndResidual.r;

    }

    WeightType weight_;

};

template <typename Scalar, int ModelDim>
struct WeightedResidualFunctorL2<Scalar,1,ModelDim,Scalar> {

    WeightedResidualFunctorL2(const Scalar weight) : weight_(weight) { }

    __attribute__((always_inline)) __host__ __device__
    Scalar operator()(const JacobianAndResidual<Scalar,1,ModelDim> & jacobianAndResidual) const {

        return Scalar(0.5) * jacobianAndResidual.r * weight_ * jacobianAndResidual.r;

    }

    Scalar weight_;

};

template <typename Scalar, int ResidualDim, int ModelDim>
struct ResidualFunctorHuber {

    ResidualFunctorHuber(const Scalar alpha) : alpha_(alpha) { }

    __attribute__((always_inline)) __host__ __device__
    Scalar operator()(const JacobianAndResidual<Scalar, ResidualDim, ModelDim> & jacobianAndResidual) const {

        const Scalar norm = internal::GenericNorm(jacobianAndResidual.r); //jacobianAndResidual.r.norm();

        if (norm < alpha_) {

            return Scalar(0.5) * norm * norm;

        } else {

            return alpha_ * (norm - Scalar(0.5) * alpha_);

        }

    }

    Scalar alpha_;

};

template <typename Scalar, int ResidualDim, int ModelDim, typename WeightType>
struct WeightedResidualFunctorHuber {

    WeightedResidualFunctorHuber(const Scalar alpha, const WeightType & weight) : alpha_(alpha), weight_(weight) { }

    __attribute__((always_inline)) __host__ __device__
    Scalar operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) const {

        const Scalar normSquared = jacobianAndResidual.r.transpose() * weight_ * jacobianAndResidual.r;

        if (normSquared < alpha_ * alpha_) {

            return Scalar(0.5) * normSquared;

        } else {

            return alpha_ * (internal::GenericSqrt(normSquared) - Scalar(0.5) * alpha_);

        }

    }

    Scalar alpha_;
    WeightType weight_;

};

template <typename Scalar, int ModelDim>
struct WeightedResidualFunctorHuber<Scalar, 1, ModelDim, Scalar> {

    WeightedResidualFunctorHuber(const Scalar alpha, const Scalar weight) : alpha_(alpha), weight_(weight) { }

    __attribute__((always_inline)) __host__ __device__
    Scalar operator()(const JacobianAndResidual<Scalar,1,ModelDim> & jacobianAndResidual) const {

        const Scalar normSquared = jacobianAndResidual.r * weight_ * jacobianAndResidual.r;

        if (normSquared < alpha_ * alpha_) {

            return Scalar(0.5) * normSquared;

        } else {

            return alpha_ * (internal::GenericSqrt(normSquared) - Scalar(0.5) * alpha_);

        }

    }

    Scalar alpha_;
    Scalar weight_;

};

template <typename Scalar, int ResidualDim, int ModelDim>
struct ResidualFunctorTukey {

    ResidualFunctorTukey(const Scalar k) : kSquared_(k * k) { }

    __attribute__((always_inline)) __host__ __device__
    Scalar operator()(const JacobianAndResidual<Scalar, ResidualDim, ModelDim> & jacobianAndResidual) const {

        const Scalar squaredNormOverKSquared = internal::GenericeSquaredNorm(jacobianAndResidual.r) / kSquared_;

        if (squaredNormOverKSquared < Scalar(1)) {

            const Scalar oneMinusSquaredNormOverKSquared = Scalar(1) - squaredNormOverKSquared;

            return kSquared_ / Scalar(6) * (Scalar(1) - oneMinusSquaredNormOverKSquared * oneMinusSquaredNormOverKSquared * oneMinusSquaredNormOverKSquared);

        } else {

            kSquared_ / Scalar(6);

        }

    }

    Scalar kSquared_;

};




//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
//                                          System Creators
//
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
template <typename Scalar, int ResidualDim, int ModelDim>
struct LinearSystemCreationFunctorL2 {

    __attribute__((always_inline)) __host__ __device__
    LinearSystem<Scalar,ModelDim> operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) const {

        // TODO: I thought transpose was slow?
        return { internal::JTJInitializer<Scalar,ResidualDim,ModelDim>::UpperTriangularJTJ(jacobianAndResidual.J),
                 jacobianAndResidual.J.transpose() * jacobianAndResidual.r };

    }

};

template <typename Scalar, int ResidualDim, int ModelDim, typename WeightType>
struct WeightedLinearSystemCreationFunctorL2 {

    WeightedLinearSystemCreationFunctorL2(const WeightType & weight) : weight_(weight) { }

    __attribute__((always_inline)) __host__ __device__
    LinearSystem<Scalar,ModelDim> operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) const {

        // TODO: I thought transpose was slow?
        return { internal::JTJInitializer<Scalar,ResidualDim,ModelDim>::UpperTriangularJTJ(jacobianAndResidual.J),
                 jacobianAndResidual.J.transpose() * weight_ * jacobianAndResidual.r };

    }

    WeightType weight_;

};

template <typename Scalar, int ModelDim>
struct WeightedLinearSystemCreationFunctorL2<Scalar,1,ModelDim,Scalar> {

    WeightedLinearSystemCreationFunctorL2(const Scalar weight) : weight_(weight) { }

    __attribute__((always_inline)) __host__ __device__
    LinearSystem<Scalar,ModelDim> operator()(const JacobianAndResidual<Scalar,1,ModelDim> & jacobianAndResidual) const {

        // TODO: I thought transpose was slow?
        return { internal::JTJInitializer<Scalar,1,ModelDim>::UpperTriangularJTJ(jacobianAndResidual.J),
                 jacobianAndResidual.J.transpose() * weight_ * jacobianAndResidual.r };

    }

    Scalar weight_;

};

template <typename Scalar, int ResidualDim, int ModelDim>
struct LinearSystemCreationFunctorHuber {

    LinearSystemCreationFunctorHuber(const Scalar alpha) : alpha_(alpha) { }

    __attribute__((always_inline)) __host__ __device__
    LinearSystem<Scalar,ModelDim> operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) const {

        const Scalar norm = internal::GenericNorm(jacobianAndResidual.r);

        LinearSystem<Scalar,ModelDim> system{ internal::JTJInitializer<Scalar,ResidualDim,ModelDim>::UpperTriangularJTJ(jacobianAndResidual.J),
                                              jacobianAndResidual.J.transpose() * jacobianAndResidual.r };

        if (norm > alpha_) {

            return (alpha_ / norm) * system;

        } else {

            return system;

        }

    }

    Scalar alpha_;

};

template <typename Scalar, int ResidualDim, int ModelDim, typename WeightType>
struct WeightedLinearSystemCreationFunctorHuber {

    WeightedLinearSystemCreationFunctorHuber(const Scalar alpha, const WeightType & weight) : alpha_(alpha), weight_(weight) { }

    __attribute__((always_inline)) __host__ __device__
    LinearSystem<Scalar, ModelDim> operator()(const JacobianAndResidual<Scalar, ResidualDim, ModelDim> & jacobianAndResidual) const {

        const Scalar normSquared = jacobianAndResidual.r.transpose() * weight_ * jacobianAndResidual.r;

        LinearSystem<Scalar,ModelDim> system{ internal::JTJInitializer<Scalar,ResidualDim,ModelDim>::UpperTriangularJTJ(jacobianAndResidual.J),
                                              jacobianAndResidual.J.transpose() * weight_ * jacobianAndResidual.r };

        if (normSquared > alpha_ * alpha_) {

            return (alpha_ / internal::GenericSqrt(normSquared)) * system;

        } else {

            return system;

        }

    }

    Scalar alpha_;
    WeightType weight_;

};

template <typename Scalar, int ModelDim>
struct WeightedLinearSystemCreationFunctorHuber<Scalar, 1, ModelDim, Scalar> {

    WeightedLinearSystemCreationFunctorHuber(const Scalar alpha, const Scalar weight) : alpha_(alpha), weight_(weight) { }

    __attribute__((always_inline)) __host__ __device__
    LinearSystem<Scalar,ModelDim> operator()(const JacobianAndResidual<Scalar,1,ModelDim> & jacobianAndResidual) const {

        const Scalar normSquared = jacobianAndResidual.r * weight_ * jacobianAndResidual.r;

        LinearSystem<Scalar,ModelDim> system{ internal::JTJInitializer<Scalar,1,ModelDim>::UpperTriangularJTJ(jacobianAndResidual.J),
                                              jacobianAndResidual.J.transpose() * weight_ * jacobianAndResidual.r };

        if (normSquared > alpha_ * alpha_) {

            return (alpha_ / internal::GenericSqrt(normSquared)) * system;

        } else {

            return system;

        }

    }

    Scalar alpha_;
    Scalar weight_;

};

template <typename Scalar, int ResidualDim, int ModelDim>
struct LinearSystemCreationFunctorTukey {

    LinearSystemCreationFunctorTukey(const Scalar k) : oneOverKSquared_(Scalar(1) / (k * k)) { }

    __attribute__((always_inline)) __host__ __device__
    LinearSystem<Scalar, ModelDim> operator()(const JacobianAndResidual<Scalar, ResidualDim, ModelDim> & jacobianAndResidual) const {

        const Scalar normSquaredOverKSquared = internal::GenericSquaredNorm(jacobianAndResidual.r) * oneOverKSquared_;

        if (normSquaredOverKSquared < Scalar(1)) {

            const Scalar oneMinusNormSquaredOverKSquared = Scalar(1) - normSquaredOverKSquared;

            const Scalar reweighting = oneMinusNormSquaredOverKSquared * oneMinusNormSquaredOverKSquared;

            return LinearSystem<Scalar,ModelDim>{ vu::operators::operator*(reweighting, internal::JTJInitializer<Scalar,ResidualDim,ModelDim>::UpperTriangularJTJ(jacobianAndResidual.J)),
                                                  reweighting * jacobianAndResidual.J.transpose() * jacobianAndResidual.r };

        } else {

            return LinearSystem<Scalar, ModelDim>::Zero();

        }

    }

    Scalar oneOverKSquared_;

};


} // namespace vu