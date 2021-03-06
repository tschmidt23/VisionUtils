#pragma once

/***
 * © Tanner Schmidt 2018
 */

#include <Eigen/Core>

#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_HD_PREFIX __host__ __device__
#else
#define CUDA_HD_PREFIX
#endif // __CUDACC__

// -=-=-=- types -=-=-=-
template <int D, typename Scalar>
using Vec = Eigen::Matrix<Scalar, D, 1, Eigen::DontAlign>;

template <typename Scalar>
using Vec2 = Vec<2, Scalar>;

template <typename Scalar>
using Vec3 = Vec<3, Scalar>;

template <typename Scalar>
using Vec4 = Vec<4, Scalar>;

template <int R, int C, typename Scalar>
using Mat = Eigen::Matrix<Scalar, R, C, Eigen::DontAlign>;

template <typename Scalar>
using Mat2 = Mat<2, 2, Scalar>;

template <typename Scalar>
using Mat3 = Mat<3, 3, Scalar>;

template <typename Scalar>
using Mat4 = Mat<4, 4, Scalar>;

namespace vu {


// -=-=-=- stream ops -=-=-=-
namespace operators {

template <typename Derived>
typename std::enable_if<0 < Eigen::internal::traits<Derived>::RowsAtCompileTime &&
                        0 < Eigen::internal::traits<Derived>::ColsAtCompileTime &&
                        !std::is_same<unsigned char, typename Eigen::internal::traits<Derived>::Scalar>::value &&
                        !std::is_same<char, typename Eigen::internal::traits<Derived>::Scalar>::value, std::istream &>::type
operator>>(std::istream & stream, Eigen::MatrixBase<Derived> & m) {
    for (int r = 0; r < Eigen::internal::traits<Derived>::RowsAtCompileTime; ++r) {
        for (int c = 0; c < Eigen::internal::traits<Derived>::ColsAtCompileTime; ++c) {
            stream >> m(r, c);
        }
    }
}

}

// this fix is required because stream >> m(r,c) reads one character at a time for char types.
// e.g. it will read the vector "159 64 12" as (1, 5, 9) instead of (159, 64, 12)
template <typename Derived>
typename std::enable_if<0 < Eigen::internal::traits<Derived>::RowsAtCompileTime &&
                        0 < Eigen::internal::traits<Derived>::ColsAtCompileTime &&
        (std::is_same<unsigned char,typename Eigen::internal::traits<Derived>::Scalar>::value ||
                        std::is_same<char,typename Eigen::internal::traits<Derived>::Scalar>::value), std::istream &>::type
operator>>(std::istream & stream, Eigen::MatrixBase<Derived> & m) {
    int val;
    for (int r = 0; r < Eigen::internal::traits<Derived>::RowsAtCompileTime; ++r) {
        for (int c = 0; c < Eigen::internal::traits<Derived>::ColsAtCompileTime; ++c) {
            stream >> val;
            m(r, c) = static_cast<typename Eigen::internal::traits<Derived>::Scalar>(val);
        }
    }
}


// -=-=-=- round -=-=-=-
template <typename T>
struct StripOptions {
    using Type = T;
};

template <typename Scalar, int M, int N, int Options>
struct StripOptions<Eigen::Matrix<Scalar, M, N, Options> > {
    using Type = Eigen::Matrix<Scalar, M, N>;
};

template <typename Derived>
using InstantiatedType = Eigen::Matrix<typename Eigen::internal::traits<Derived>::Scalar,
                                       Eigen::internal::traits<Derived>::RowsAtCompileTime,
                                       Eigen::internal::traits<Derived>::ColsAtCompileTime,
                                       Eigen::internal::traits<Derived>::Options>;

// -=-=-=- round -=-=-=-
template <typename Derived>
CUDA_HD_PREFIX
inline Eigen::Matrix<int,Eigen::internal::traits<Derived>::RowsAtCompileTime,Eigen::internal::traits<Derived>::ColsAtCompileTime,Eigen::internal::traits<Derived>::Options>
Round(const Eigen::MatrixBase<Derived> & v) {

    return v.array().round().matrix().template cast<int>();

}

template <typename Derived>
CUDA_HD_PREFIX
inline Eigen::Matrix<int,Eigen::internal::traits<Derived>::RowsAtCompileTime,Eigen::internal::traits<Derived>::ColsAtCompileTime,Eigen::internal::traits<Derived>::Options>
Floor(const Eigen::MatrixBase<Derived> & v) {

    return v.array().floor().matrix().template cast<int>();

}

// -=-=-=- element-wise inverse -=-=-=-
template <typename Derived>
CUDA_HD_PREFIX
inline Eigen::Matrix<typename Eigen::internal::traits<Derived>::Scalar,Eigen::internal::traits<Derived>::RowsAtCompileTime,Eigen::internal::traits<Derived>::ColsAtCompileTime,Eigen::internal::traits<Derived>::Options>
ElementwiseInverse(const Eigen::MatrixBase<Derived> & v) {

    typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;

    return v.unaryExpr([](const Scalar val) -> Scalar { return Scalar(1) / val; });

}

// -=-=-=- vector manipulation -=-=-=-
//template <typename Scalar, typename Derived,
//          typename std::enable_if<std::is_same<Eigen::internal::traits<Derived>::Scalar,Scalar>::value &&
//                                  Eigen::internal::traits<Derived>::ColsAtCompileTime == 1,int>::type = 0>
//__host__ __device__
//inline Eigen::Matrix<Eigen::internal::traits<Derived>::Scalar,Eigen::internal::traits<Derived>::RowsAtCompileTime,1> compose(Scalar head, Eigen::Matrix<Eigen::internal)

//template <typename Scalar>
//__host__ __device__
//inline Eigen::Matrix<Scalar,1,1> reverse(const Eigen::Matrix<Scalar,1,1> & vec) {

//    return vec;

//}

//template <typename Scalar, int D>
//__host__ __device__
//inline Eigen::Matrix<Scalar,D,1> reverse(const Eigen::Matrix<Scalar,D,1> & vec) {

//    Eigen::Matrix<Scalar,D,1> rev;
//    rev.template head<1>() = vec.template tail<1>();
//    rev.template tail<D-1>() = reverse(vec.template head<D-1>());
//    return rev;

//}

//template <typename Derived,
//          typename std::enable_if<Eigen::internal::traits<Derived>::ColsAtCompileTime == 1 &&
//                                  Eigen::internal::traits<Derived>::RowsAtCompileTime == 1,int>::type = 0>
//__host__ __device__
//inline Eigen::Matrix<typename Eigen::internal::traits<Derived>::Scalar,Eigen::internal::traits<Derived>::RowsAtCompileTime,1> reverse(const Eigen::MatrixBase<Derived> & vec) {

//    return vec;

//}

//template <typename Derived,
//          typename std::enable_if<Eigen::internal::traits<Derived>::ColsAtCompileTime == 1,int>::type = 0>
//__host__ __device__
//inline Eigen::Matrix<typename Eigen::internal::traits<Derived>::Scalar,Eigen::internal::traits<Derived>::RowsAtCompileTime,1> reverse(const Eigen::MatrixBase<Derived> & vec) {

//    typedef typename Eigen::internal::traits<Derived>::Scalar Scalar;
//    static constexpr int D = Eigen::internal::traits<Derived>::RowsAtCompileTime;

//    Eigen::Matrix<Scalar,D,1> rev;
//    rev.template head<1>() = vec.template tail<1>();
//    if ( D > 0) {
//        rev.template tail<D-1>() = reverse(vec.template head<D-1>());
//    }
//    return rev;

//}

// -=-=-=- generic comparisons -=-=-=-
template <typename Scalar, int D>
struct VecCompare {

    CUDA_HD_PREFIX
    static inline bool Less(const Vec<D, Scalar> & a, const Vec<D, Scalar> & b) {
        if (a(0) < b(0)) return true;
        if (a(0) > b(0)) return false;

        return VecCompare<Scalar,D-1>::Less(a.template tail<D-1>(), b.template tail<D-1>());
    }

    CUDA_HD_PREFIX
    static inline bool Equal(const Vec<D, Scalar> & a, const Vec<D, Scalar> & b) {

        if (a(0) != b(0)) return false;

        return VecCompare<Scalar, D-1>::Equal(a.template tail<D-1>(), b.template tail<D-1>());
    }

};

template <typename Scalar>
struct VecCompare<Scalar, 1> {

    CUDA_HD_PREFIX
    static inline bool Less(const Vec<1, Scalar> & a, const Vec<1, Scalar> & b) {

        return a(0) < b(0);

    }

    CUDA_HD_PREFIX
    static inline bool Equal(const Vec<1, Scalar> & a, const Vec<1, Scalar> & b) {

        return a(0) == b(0);

    }

};

// as functors
template <typename Scalar, int D>
struct VecLess {

    CUDA_HD_PREFIX
    inline bool operator()(const Vec<D, Scalar> & a, const Vec<D, Scalar> & b) const {

        return VecCompare<Scalar, D>::Less(a, b);

    }

};

template <typename Scalar, int D>
struct VecEqual {

    CUDA_HD_PREFIX
    inline bool operator()(const Vec<D, Scalar> & a, const Vec<D, Scalar> & b) const {

        return VecCompare<Scalar, D>::Equal(a, b);

    }


};

} // namespace vu