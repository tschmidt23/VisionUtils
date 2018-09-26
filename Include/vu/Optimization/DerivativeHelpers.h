#pragma once

namespace vu {

template <typename Derived, typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 3 &&
                                                    Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0>
inline __host__ __device__ Eigen::Matrix<typename Eigen::internal::traits<Derived>::Scalar, 3, 6>
DerivativeOfPointTransformationWrtTransform(const Eigen::MatrixBase<Derived> & point) {

    return (Eigen::Matrix<typename Eigen::internal::traits<Derived>::Scalar, 3, 6>() <<
        1, 0, 0,          0,  point(2), -point(1),
        0, 1, 0,  -point(2),         0,  point(0),
        0, 0, 1,   point(1), -point(0),         0).finished();

}


template <typename Derived, typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 3 &&
                                                    Eigen::internal::traits<Derived>::ColsAtCompileTime == 1, int>::type = 0>
inline __host__ __device__ Eigen::Matrix<typename Eigen::internal::traits<Derived>::Scalar, 3, 6>
DerivativeOfVectorTransformationWrtTransform(const Eigen::MatrixBase<Derived> & vector) {

    return (Eigen::Matrix<typename Eigen::internal::traits<Derived>::Scalar, 3, 6>() <<
            0, 0, 0,           0,  vector(2), -vector(1),
            0, 0, 0,  -vector(2),          0,  vector(0),
            0, 0, 0,   vector(1), -vector(0),          0).finished();

}

} // namespace vu