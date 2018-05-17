/***
 * © Tanner Schmidt 2018
 */
 
#include <vu/Geometry/FundamentalMatrix.h>
#include <vu/Geometry/Triangulation.h>

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
        A(i, 1) = x1 * y0;
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

    std::cout << "solution: " << std::endl << solution.transpose() << std::endl << std::endl;

    // TODO: check if ordinality is correct
    Eigen::Map<const Eigen::Matrix<Scalar, 3, 3, Eigen::RowMajor> > reshapedF(solution.data());

    Eigen::Matrix<Scalar, 3, 3> F = reshapedF;

    std::cout << "F: " << std::endl << F << std::endl << std::endl;

    Eigen::JacobiSVD<decltype(F)> svd2(F, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix<Scalar, 3, 1> sigma = svd2.singularValues();

    std::cout << "sigma: " << std::endl << sigma.transpose() << std::endl << std::endl;

    std::cout << "F: " <<  std::endl << F << std::endl << std::endl;

    std::cout << "or: " << std::endl << (svd2.matrixU() * Eigen::DiagonalMatrix<Scalar, 3, 3>(sigma) * svd2.matrixV().transpose()) << std::endl << std::endl;

    sigma(2) = 0;

    return svd2.matrixU() * Eigen::DiagonalMatrix<Scalar, 3, 3>(sigma) * svd2.matrixV().transpose();

}

template Eigen::Matrix3f ComputeFundamentalMatrix(const NDT::Vector<Vec2<float> > &,
                                                  const NDT::Vector<Vec2<float> > &);

template Eigen::Matrix3d ComputeFundamentalMatrix(const NDT::Vector<Vec2<double> > &,
                                                  const NDT::Vector<Vec2<double> > &);


template <typename Scalar>
inline bool CheckTransform(const Sophus::SE3<Scalar> & transform,
                           const NDT::Vector<Vec2<Scalar> > & raysSource,
                           const NDT::Vector<Vec2<Scalar> > & raysDestination) {

    for (int i = 0; i < raysSource.Length(); ++i) {

        const Eigen::Matrix<Scalar, 3, 1> p = LinearLeastSquaresTriangulation(Sophus::SE3<Scalar>(), raysSource(i), transform, raysDestination(i));

        if (p(2) < 0 || (transform * p)(2) < 0) {
            return false;
        }

    }

    return true;

}

template <typename Scalar>
Sophus::SE3<Scalar> RelativePoseFromFundamentalMatrix(const Eigen::Matrix<Scalar, 3, 3> & F,
                                                      const Eigen::Matrix<Scalar, 3, 3> & K,
                                                      const NDT::Vector<Vec2<Scalar> > & raysSource,
                                                      const NDT::Vector<Vec2<Scalar> > & raysDestination) {

    Sophus::SE3<Scalar> relativePose;

    const Eigen::Matrix<Scalar, 3, 3> E = K.transpose() * F * K;

    Eigen::JacobiSVD<Eigen::Matrix<Scalar, 3, 3> > svdE(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

    const Eigen::Matrix<Scalar, 3, 1> t = svdE.matrixU().col(2).normalized();

    // TODO: this could be more efficient
    const Eigen::Matrix<Scalar, 3, 3> W = (Eigen::Matrix<Scalar, 3, 3>() << 0, -1, 0,  1, 0, 0,  0, 0, 1).finished();

    Eigen::Matrix<Scalar, 3, 3> R1 = svdE.matrixU() * W * svdE.matrixV().transpose();

    if (R1.determinant() < 0) {
        R1 *= -1;
    }


    relativePose = Sophus::SE3<Scalar>(Sophus::SO3<Scalar>(R1), t);

    if (CheckTransform(relativePose, raysSource, raysDestination)) {
        return relativePose;
    }

    relativePose.translation() *= -1;

    if (CheckTransform(relativePose, raysSource, raysDestination)) {
        return relativePose;
    }

    Eigen::Matrix<Scalar, 3, 3> R2 = svdE.matrixU() * W.transpose() * svdE.matrixV().transpose();

    if (R2.determinant() < 0) {
        R2 *= -1;
    }

    relativePose.so3() = Sophus::SO3<Scalar>(R2);

    if (CheckTransform(relativePose, raysSource, raysDestination)) {
        return relativePose;
    }

    relativePose.translation() *= -1;

    if (CheckTransform(relativePose, raysSource, raysDestination)) {
        return relativePose;
    }

    throw std::runtime_error("none of the transforms checked out");

}

template
Sophus::SE3f RelativePoseFromFundamentalMatrix(const Eigen::Matrix3f & F,
                                               const Eigen::Matrix3f & K,
                                               const NDT::Vector<Vec2<float> > & raysSource,
                                               const NDT::Vector<Vec2<float> > & raysDestination);

template
Sophus::SE3d RelativePoseFromFundamentalMatrix(const Eigen::Matrix3d & F,
                                               const Eigen::Matrix3d & K,
                                               const NDT::Vector<Vec2<double> > & raysSource,
                                               const NDT::Vector<Vec2<double> > & raysDestination);

} // namespace vu