#include <vu/Geometry/FundamentalMatrix.h>

namespace vu {

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 3> ComputeFundamentalMatrix(const NDT::Vector<Vec2<Scalar>> & sourcePoints,
                                                     const NDT::Vector<Vec2<Scalar>> & targetPoints) {

    const int N = sourcePoints.Length();

    Eigen::Matrix<Scalar, Eigen::Dynamic, 9> A(N, 9);

    for (int i = 0; i < N; ++i) {

        const Scalar & x0 = sourcePoints(i)(0);
        const Scalar & y0 = sourcePoints(i)(1);

        const Scalar & x1 = targetPoints(i)(0);
        const Scalar & y1 = targetPoints(i)(1);

        A(i, 0) = x0 * x1;
        A(i, 1) = y0 * x1;
        A(i, 2) = x1;
        A(i, 3) = x0 * y1;
        A(i, 4) = y0 * y1;
        A(i, 5) = y1;
        A(i, 6) = x0;
        A(i, 7) = y0;
        A(i, 8) = Scalar(1);

    }

    Eigen::JacobiSVD<Eigen::Matrix<Scalar, Eigen::Dynamic, 9>> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    std::cout << svd.singularValues() << std::endl << std::endl;

    // TODO: might need to be col
    auto solution = svd.matrixV().col(8);

    // TODO: check if ordinality is correct
    Eigen::Map<const Eigen::Matrix<Scalar, 3, 3>> reshapedF(solution.data());

    Eigen::Matrix<Scalar, 3, 3> F = reshapedF;

    Eigen::JacobiSVD<decltype(F)> svd2(F, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix<Scalar, 3, 1> sigma = svd2.singularValues();

    sigma(2) = 0;

    return svd2.matrixU() * Eigen::DiagonalMatrix<Scalar, 3, 3>(sigma) * svd2.matrixV().transpose();

}

template Eigen::Matrix3f ComputeFundamentalMatrix(const NDT::Vector<Vec2<float> > &,
                                                  const NDT::Vector<Vec2<float> > &);

template Eigen::Matrix3d ComputeFundamentalMatrix(const NDT::Vector<Vec2<double> > &,
                                                  const NDT::Vector<Vec2<double> > &);

} // namespace vu