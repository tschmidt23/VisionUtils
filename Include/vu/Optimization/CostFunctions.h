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

__host__ __device__

inline float GenericSqrt(const float & val) {
    return sqrtf(val);
}

__host__ __device__

inline double GenericSqrt(const double & val) {
    return sqrt(val);
}

} // namespace internal

template <typename Scalar, int ResidualDim, int ModelDim>
struct ResidualFunctorL2 {

    __attribute__((always_inline)) __host__ __device__
    Scalar operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) {

        return Scalar(0.5) * jacobianAndResidual.r.squaredNorm();
    }

};

template <typename Scalar, int ModelDim>
struct ResidualFunctorL2<Scalar,1,ModelDim> {

    __attribute__((always_inline)) __host__ __device__
    Scalar operator()(const JacobianAndResidual<Scalar,1,ModelDim> & jacobianAndResidual) {

        return Scalar(0.5) * jacobianAndResidual.r * jacobianAndResidual.r;

    }

};

template <typename Scalar, int ResidualDim, int ModelDim, typename WeightType>
struct WeightedResidualFunctorL2 {

    WeightedResidualFunctorL2(const WeightType & weight) : weight_(weight) { }

    __attribute__((always_inline)) __host__ __device__
    Scalar operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) {

        // TODO: I think transpose is slow
        return Scalar(0.5) * jacobianAndResidual.r.transpose() * weight_ * jacobianAndResidual.r;

    }

    WeightType weight_;

};

template <typename Scalar, int ModelDim>
struct WeightedResidualFunctorL2<Scalar,1,ModelDim,Scalar> {

    WeightedResidualFunctorL2(const Scalar weight) : weight_(weight) { }

    __attribute__((always_inline)) __host__ __device__
    Scalar operator()(const JacobianAndResidual<Scalar,1,ModelDim> & jacobianAndResidual) {

        return Scalar(0.5) * jacobianAndResidual.r * weight_ * jacobianAndResidual.r;

    }

    Scalar weight_;

};

template <typename Scalar, int ResidualDim, int ModelDim>
struct ResidualFunctorHuber {

    ResidualFunctorHuber(const Scalar alpha) : alpha_(alpha) { }

    __attribute__((always_inline)) __host__ __device__
    Scalar operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) {

        const Scalar norm = jacobianAndResidual.r.norm();

        if (norm < alpha_) {

            return Scalar(0.5) * norm * norm;

        } else {

            return alpha_ * (norm - Scalar(0.5) * alpha_);

        }

    }

    Scalar alpha_;

};

template <typename Scalar, int ModelDim>
struct ResidualFunctorHuber<Scalar,1,ModelDim> {

    ResidualFunctorHuber(const Scalar alpha) : alpha_(alpha) { }

    __attribute__((always_inline)) __host__ __device__
    Scalar operator()(const JacobianAndResidual<Scalar,1,ModelDim> & jacobianAndResidual) {

        const Scalar norm = fabsf(jacobianAndResidual.r);

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
    Scalar operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) {

        const Scalar normSquared = jacobianAndResidual.r.transpose() * weight_ * jacobianAndResidual.r;

        if (normSquared < alpha_ * alpha_) {

            return Scalar(0.5) * normSquared;

        } else {

            return alpha_ * (GenericSqrt(normSquared) - Scalar(0.5) * alpha_);

        }

    }

    Scalar alpha_;
    WeightType weight_;

};

template <typename Scalar, int ModelDim>
struct WeightedResidualFunctorHuber<Scalar, 1, ModelDim, Scalar> {

    WeightedResidualFunctorHuber(const Scalar alpha, const Scalar weight) : alpha_(alpha), weight_(weight) { }

    __attribute__((always_inline)) __host__ __device__
    Scalar operator()(const JacobianAndResidual<Scalar,1,ModelDim> & jacobianAndResidual) {

        const Scalar normSquared = jacobianAndResidual.r * weight_ * jacobianAndResidual.r;

        if (normSquared < alpha_ * alpha_) {

            return Scalar(0.5) * normSquared;

        } else {

            return alpha_ * (GenericSqrt(normSquared) - Scalar(0.5) * alpha_);

        }

    }

    Scalar alpha_;
    Scalar weight_;

};

template <typename Scalar, int ResidualDim, int ModelDim>
struct LinearSystemCreationFunctorL2 {

    __attribute__((always_inline)) __host__ __device__
    LinearSystem<Scalar,ModelDim> operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) {

        // TODO: I thought transpose was slow?
        return { internal::JTJInitializer<Scalar,ResidualDim,ModelDim>::UpperTriangularJTJ(jacobianAndResidual.J),
                 jacobianAndResidual.J.transpose() * jacobianAndResidual.r };

    }

};

template <typename Scalar, int ResidualDim, int ModelDim, typename WeightType>
struct WeightedLinearSystemCreationFunctorL2 {

    WeightedLinearSystemCreationFunctorL2(const WeightType & weight) : weight_(weight) { }

    __attribute__((always_inline)) __host__ __device__
    LinearSystem<Scalar,ModelDim> operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) {

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
    LinearSystem<Scalar,ModelDim> operator()(const JacobianAndResidual<Scalar,1,ModelDim> & jacobianAndResidual) {

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
    LinearSystem<Scalar,ModelDim> operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) {

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
    LinearSystem<Scalar,ModelDim> operator()(const JacobianAndResidual<Scalar,ResidualDim,ModelDim> & jacobianAndResidual) {

        const Scalar normSquared = jacobianAndResidual.r.transpose() * weight_ * jacobianAndResidual.r;

        LinearSystem<Scalar,ModelDim> system{ internal::JTJInitializer<Scalar,ResidualDim,ModelDim>::UpperTriangularJTJ(jacobianAndResidual.J),
                                              jacobianAndResidual.J.transpose() * weight_ * jacobianAndResidual.r };

        if (normSquared > alpha_ * alpha_) {

            return (alpha_ / GenericSqrt(normSquared)) * system;

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
    LinearSystem<Scalar,ModelDim> operator()(const JacobianAndResidual<Scalar,1,ModelDim> & jacobianAndResidual) {

        const Scalar normSquared = jacobianAndResidual.r * weight_ * jacobianAndResidual.r;

        LinearSystem<Scalar,ModelDim> system{ internal::JTJInitializer<Scalar,1,ModelDim>::UpperTriangularJTJ(jacobianAndResidual.J),
                                              jacobianAndResidual.J.transpose() * weight_ * jacobianAndResidual.r };

        if (normSquared > alpha_ * alpha_) {

            return (alpha_ / GenericSqrt(normSquared)) * system;

        } else {

            return system;

        }

    }

    Scalar alpha_;
    Scalar weight_;

};

} // namespace vu