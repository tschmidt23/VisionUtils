#pragma once

#include <sophus/se3.hpp>

namespace vu {

template <typename Derived>
inline
typename std::enable_if<(Eigen::internal::traits<Derived>::RowsAtCompileTime == 3 || Eigen::internal::traits<Derived>::RowsAtCompileTime == 4) &&
                         Eigen::internal::traits<Derived>::ColsAtCompileTime == 4, Sophus::SE3<typename Eigen::internal::traits<Derived>::Scalar> >::type
EigenToSophus(const Eigen::MatrixBase<Derived> & M) {

    using Scalar = typename Eigen::internal::traits<Derived>::Scalar;

    return Sophus::SE3<Scalar>(Sophus::SO3<Scalar>(M.template block<3, 3>(0, 0)), M.template block<3, 1>(0, 3));

}

template <typename Derived>
inline
typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 3 &&
        Eigen::internal::traits<Derived>::ColsAtCompileTime == 1,
        Sophus::SE3<typename Eigen::internal::traits<Derived>::Scalar> >::type
SophusTranslation(const Eigen::MatrixBase<Derived> & translation) {

    using Scalar = typename Eigen::internal::traits<Derived>::Scalar;

    return Sophus::SE3<Scalar>(Sophus::SO3<Scalar>(), translation);

}

} // namespace vu