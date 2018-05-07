#pragma once

#include <numeric>

#include <vu/EigenHelpers.h>

namespace vu {

template <int D, typename Scalar>
inline Vec<D+1,Scalar> Homogenize(const Vec<D,Scalar> & v) {

    return (Vec<D+1,Scalar>() << v, Scalar(1)).finished();

}

template <int D, typename Scalar>
inline Vec<D-1,Scalar> Dehomogenize(const Vec<D,Scalar> & v) {

    return v.template head<D-1>() / v(D-1);

};


template <int D, typename Scalar>
Vec<D,Scalar> Mean(const NDT::Vector<Vec<D,Scalar> > & points) {

    Vec<D,Scalar> init = Vec<D,Scalar>::Zero();
    return std::accumulate(points.Data(), points.Data() + points.Length(), init) /
           static_cast<Scalar>(points.Length());

}

template <int D, typename Scalar>
Vec<D,Scalar> StandardDeviation(const NDT::Vector<Vec<D,Scalar> > & points, const Vec<D,Scalar> & mean) {

    Vec<D,Scalar> init = Vec<D,Scalar>::Zero();
    return (std::accumulate(points.Data(), points.Data() + points.Length(), init,
                            [&mean](const Vec<D,Scalar> & accumulator, const Vec<D,Scalar> & point) {
                                return accumulator + (point - mean).array().square().matrix();
                            }) / static_cast<Scalar>(points.Length())).cwiseSqrt();
}

template <int D, typename Scalar>
Vec<D,Scalar> InverseStandardDeviation(const NDT::Vector<Vec<D,Scalar> > & points, const Vec<D,Scalar> & mean) {

    Vec<D,Scalar> init = Vec<D,Scalar>::Zero();
    return (std::accumulate(points.Data(), points.Data() + points.Length(), init,
                            [&mean](const Vec<D,Scalar> & accumulator, const Vec<D,Scalar> & point) {
                                return accumulator + (point - mean).array().square().matrix();
                            }) / static_cast<Scalar>(points.Length())).array().rsqrt().matrix();

};

template <int D, typename Scalar>
Eigen::Matrix<Scalar,D+1,D+1> WhitenPoints(const NDT::Vector<Vec<D,Scalar> > & points,
                                           NDT::Vector<Vec<D,Scalar> > & normalizedPoints) {

    const Vec<D, Scalar> mean = Mean(points);

    const Vec<D, Scalar> invStandardDeviation = InverseStandardDeviation(points, mean);

    for (int i = 0; i < points.Length(); ++i) {

        normalizedPoints(i) = (points(i) - mean).cwiseProduct(invStandardDeviation);

    }

    Eigen::Matrix<Scalar, D + 1, D + 1> T;
    T.template block<D, 1>(0, D) = -mean.cwiseProduct(invStandardDeviation);
    T.template block<D, D>(0, 0) = Eigen::DiagonalMatrix<Scalar, D, D>(invStandardDeviation);
    T.template block<1, D>(D, 0) = Eigen::Matrix<Scalar, 1, D>::Zero();
    T(D, D) = 1;

    return T;

}

};