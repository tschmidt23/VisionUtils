#pragma once

#include <sophus/se3.hpp>

#include <Eigen/SVD>

namespace vu {

template <typename Scalar, typename Derived>
typename std::enable_if<Eigen::internal::traits<Derived>::RowsAtCompileTime == 2 &&
                        Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
                        std::is_same<typename Eigen::internal::traits<Derived>::Scalar, Scalar>::value, Eigen::Matrix<Scalar, 3, 1> >::type
LinearLeastSquaresTriangulation(const Sophus::SE3<Scalar> & poseA, const Eigen::MatrixBase<Derived> & pointA,
                                const Sophus::SE3<Scalar> & poseB, const Eigen::MatrixBase<Derived> & pointB) {

    const Eigen::Matrix<Scalar, 3, 4> dcmA = poseA.matrix3x4();
    const Eigen::Matrix<Scalar, 3, 4> dcmB = poseB.matrix3x4();

    Eigen::Matrix<Scalar, 4, 3> A;
    A.template block<1, 3>(0, 0) = pointA(0) * dcmA.template block<1, 3>(2, 0) - dcmA.template block<1,3>(0, 0);
    A.template block<1, 3>(1, 0) = pointA(1) * dcmA.template block<1, 3>(2, 0) - dcmA.template block<1,3>(1, 0);
    A.template block<1, 3>(2, 0) = pointB(0) * dcmB.template block<1, 3>(2, 0) - dcmB.template block<1,3>(0, 0);
    A.template block<1, 3>(3, 0) = pointB(1) * dcmB.template block<1, 3>(2, 0) - dcmB.template block<1,3>(1, 0);

    Eigen::Matrix<Scalar,4,1> b(-pointA(0) * dcmA(2,3) + dcmA(0,3),
                                -pointA(1) * dcmA(2,3) + dcmA(1,3),
                                -pointB(0) * dcmB(2,3) + dcmB(0,3),
                                -pointB(1) * dcmB(2,3) + dcmB(1,3));

    return A.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV).solve(b);

}

} // namespace vu