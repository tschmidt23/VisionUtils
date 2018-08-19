#pragma once

namespace vu {


enum TransformUpdateMethod {
    TransformUpdateLeftMultiply = 0,
    TransformUpdateRightMultiply
};

// -=-=-=-=- helper data structures -=-=-=-=-
template <typename Scalar, int ResidualDim, int ModelDim>
struct JacobianAndResidual {
    Eigen::Matrix<Scalar,ResidualDim,ModelDim,Eigen::DontAlign> J;
    Eigen::Matrix<Scalar,ResidualDim,1,Eigen::DontAlign> r;
};

template <typename Scalar, int ModelDim>
struct JacobianAndResidual<Scalar,1,ModelDim> {
    Eigen::Matrix<Scalar,1,ModelDim,Eigen::DontAlign | Eigen::RowMajor> J;
    Scalar r;
};


template <typename Scalar, int D>
struct UpperTriangularMatrix {
    Eigen::Matrix<Scalar,1,D,Eigen::DontAlign | Eigen::RowMajor> head;
    UpperTriangularMatrix<Scalar,D-1> tail;

    static __attribute__((always_inline)) __host__ __device__
    UpperTriangularMatrix<Scalar,D> Zero() {
        return { Eigen::Matrix<Scalar,1,D,Eigen::DontAlign | Eigen::RowMajor>::Zero(), UpperTriangularMatrix<Scalar,D-1>::Zero() };
    }

    __attribute__((always_inline)) __host__ __device__
    UpperTriangularMatrix<Scalar,D> operator+(const UpperTriangularMatrix<Scalar,D> & other) const {
        return { head + other.head, tail + other.tail };
    }

    __attribute__((always_inline)) __host__ __device__
    UpperTriangularMatrix<Scalar, D> & operator+=(const UpperTriangularMatrix<Scalar, D> & other) {
        head += other.head;
        tail += other.tail;
        return *this;
    }

    __attribute__((always_inline)) __host__ __device__
    Scalar & operator()(const int r, const int c) {
        if (r == 0) {
            return head(c);
        }
        return tail(r-1, c-1);
    }

    __attribute__((always_inline)) __host__ __device__
    const Scalar & operator()(const int r, const int c) const {
        if (r == 0) {
            return head(c);
        }
        return tail(r-1, c-1);
    }


};

template <typename Scalar>
struct  UpperTriangularMatrix<Scalar,1> {
    Eigen::Matrix<Scalar,1,1,Eigen::DontAlign | Eigen::RowMajor> head;

    static __attribute__((always_inline)) __host__ __device__
    UpperTriangularMatrix<Scalar,1> Zero() {
        return { Eigen::Matrix<Scalar,1,1,Eigen::DontAlign | Eigen::RowMajor>::Zero() };
    }

    __attribute__((always_inline)) __host__ __device__
    UpperTriangularMatrix<Scalar,1> operator+(const UpperTriangularMatrix & other) const {
        return { head + other.head };
    }

    __attribute__((always_inline)) __host__ __device__
    UpperTriangularMatrix<Scalar, 1> & operator+=(const UpperTriangularMatrix<Scalar, 1> & other) {
        head += other.head;
        return *this;
    }

    __attribute__((always_inline)) __host__ __device__
    Scalar & operator()(const int /*r*/, const int /*c*/) {
        return head(0);
    }

    __attribute__((always_inline)) __host__ __device__
    const Scalar & operator()(const int /*r*/, const int /*c*/) const {
        return head(0);
    }

};

template <typename Scalar, int ModelDim>
struct LinearSystem {

    static __attribute__((always_inline)) __host__ __device__
    LinearSystem<Scalar,ModelDim> Zero() {
        return { UpperTriangularMatrix<Scalar,ModelDim>::Zero(), Eigen::Matrix<Scalar,ModelDim,1,Eigen::DontAlign | Eigen::ColMajor>::Zero() };
    }

    inline __host__ __device__ LinearSystem<Scalar,ModelDim> operator+(const LinearSystem<Scalar,ModelDim> & other) const {

        return { JTJ + other.JTJ, JTr + other.JTr };

    }

    UpperTriangularMatrix<Scalar,ModelDim> JTJ;
    Eigen::Matrix<Scalar,ModelDim,1,Eigen::DontAlign | Eigen::ColMajor> JTr;
};



// -=-=-=-=- helper functors -=-=-=-=-
template <typename Scalar>
inline __host__ __device__ UpperTriangularMatrix<Scalar,1> operator*(const Scalar a, const UpperTriangularMatrix<Scalar,1> & M) {

    return { a * M.head };

}

template <typename Scalar, int D>
inline __host__ __device__ UpperTriangularMatrix<Scalar,D> operator*(const Scalar a, const UpperTriangularMatrix<Scalar,D> & M) {

    return { a * M.head, a * M.tail };

}

template <typename Scalar, int ModelDim>
inline __host__ __device__ LinearSystem<Scalar,ModelDim> operator*(const Scalar a, const LinearSystem<Scalar,ModelDim> & system) {

    return { a * system.JTJ, a * system.JTr };

}

namespace internal {

template <typename Scalar, int ResidualDim, int ModelDim, int D>
struct JTJRowInitializer {

    static __attribute__((always_inline)) __host__ __device__

    void InitializeRow(Eigen::Matrix<Scalar, 1, ModelDim, Eigen::DontAlign | Eigen::RowMajor> & row,
                       const Eigen::Matrix<Scalar, ResidualDim, ModelDim, Eigen::DontAlign | Eigen::RowMajor> & J) {
        row(D) = J.template block<ResidualDim, 1>(0, 0).dot(J.template block<ResidualDim, 1>(0,
                                                                                             D)); // StaticDotProduct<Scalar,ResidualDim>::dot(J.template block<ResidualDim,1>(0,0),J.template block<ResidualDim,1>(0,D)); //J.template block<ResidualDim,1>(0,0).transpose()*J.template block<ResidualDim,1>(0,D);
        JTJRowInitializer<Scalar, ResidualDim, ModelDim, D - 1>::InitializeRow(row, J);
    }

};

// recursive base case
template <typename Scalar, int ResidualDim, int ModelDim>
struct JTJRowInitializer<Scalar, ResidualDim, ModelDim, -1> {

    static __attribute__((always_inline)) __host__ __device__

    void InitializeRow(Eigen::Matrix<Scalar, 1, ModelDim, Eigen::DontAlign | Eigen::RowMajor> & /*row*/,
                       const Eigen::Matrix<Scalar, ResidualDim, ModelDim,
                               Eigen::DontAlign | Eigen::RowMajor> & /*J*/) {}

};


template <typename Scalar, int ResidualDim, int ModelDim>
struct JTJInitializer {

    static __attribute__((always_inline)) __host__ __device__

    UpperTriangularMatrix<Scalar, ModelDim> UpperTriangularJTJ(
            const Eigen::Matrix<Scalar, ResidualDim, ModelDim, Eigen::DontAlign | Eigen::RowMajor> & jacobian) {
        Eigen::Matrix < Scalar, 1, ModelDim, Eigen::DontAlign | Eigen::RowMajor > row;
        JTJRowInitializer<Scalar, ResidualDim, ModelDim, ModelDim - 1>::InitializeRow(row, jacobian);
        return {row, JTJInitializer<Scalar, ResidualDim, ModelDim - 1>::UpperTriangularJTJ(
                jacobian.template block<ResidualDim, ModelDim - 1>(0, 1))};
    }

};

// recursive base case
template <typename Scalar, int ResidualDim>
struct JTJInitializer<Scalar, ResidualDim, 1> {

    static __attribute__((always_inline)) __host__ __device__

    UpperTriangularMatrix<Scalar, 1>
    UpperTriangularJTJ(const Eigen::Matrix<Scalar, ResidualDim, 1, Eigen::DontAlign> & jacobian) {
        return {jacobian.transpose() * jacobian};
    }

};

template <typename Scalar, int ResidualDim, int ModelDim, int D>
struct JTJRowInitializerHuber {

    static __attribute__((always_inline)) __host__ __device__

    void InitializeRow(Eigen::Matrix<Scalar, 1, ModelDim, Eigen::DontAlign | Eigen::RowMajor> & row,
                       const Eigen::Matrix<Scalar, ResidualDim, ModelDim, Eigen::DontAlign | Eigen::RowMajor> & J,
                       const Eigen::Matrix<Scalar, ResidualDim, ModelDim,
                               Eigen::DontAlign | Eigen::RowMajor> & rhoDoublePrimeJ) {
        row(D) = J.template block<ResidualDim, 1>(0, 0).dot(rhoDoublePrimeJ.template block<ResidualDim, 1>(0,
                                                                                                           D)); // StaticDotProduct<Scalar,ResidualDim>::dot(J.template block<ResidualDim,1>(0,0),J.template block<ResidualDim,1>(0,D)); //J.template block<ResidualDim,1>(0,0).transpose()*J.template block<ResidualDim,1>(0,D);
        JTJRowInitializerHuber<Scalar, ResidualDim, ModelDim, D - 1>::InitializeRow(row, J, rhoDoublePrimeJ);
    }

};

// recursive base case
template <typename Scalar, int ResidualDim, int ModelDim>
struct JTJRowInitializerHuber<Scalar, ResidualDim, ModelDim, -1> {

    static __attribute__((always_inline)) __host__ __device__

    void InitializeRow(Eigen::Matrix<Scalar, 1, ModelDim, Eigen::DontAlign | Eigen::RowMajor> & /*row*/,
                       const Eigen::Matrix<Scalar, ResidualDim, ModelDim, Eigen::DontAlign | Eigen::RowMajor> & /*J*/,
                       const Eigen::Matrix<Scalar, ResidualDim, ModelDim,
                               Eigen::DontAlign | Eigen::RowMajor> & /*rhoDoublePrimeJ*/) {}

};

template <typename Scalar, int ResidualDim, int ModelDim>
struct JTJInitializerHuber {

    static __attribute__((always_inline)) __host__ __device__

    UpperTriangularMatrix<Scalar, ModelDim> UpperTriangularJTJ(
            const Eigen::Matrix<Scalar, ResidualDim, ModelDim, Eigen::DontAlign | Eigen::RowMajor> & jacobian,
            const Eigen::Matrix<Scalar, ResidualDim, ModelDim,
                    Eigen::DontAlign | Eigen::RowMajor> & rhoDoublePrimeJacobian) {
        Eigen::Matrix < Scalar, 1, ModelDim, Eigen::DontAlign | Eigen::RowMajor > row;
        JTJRowInitializerHuber<Scalar, ResidualDim, ModelDim, ModelDim - 1>::InitializeRow(row, jacobian,
                                                                                           rhoDoublePrimeJacobian);
        return {row, JTJInitializerHuber<Scalar, ResidualDim, ModelDim - 1>::UpperTriangularJTJ(
                jacobian.template block<ResidualDim, ModelDim - 1>(0, 1),
                rhoDoublePrimeJacobian.template block<ResidualDim, ModelDim - 1>(0, 1))};
    }

};

template <typename Scalar, int ResidualDim>
struct JTJInitializerHuber<Scalar, ResidualDim, 1> {

    static __attribute__((always_inline)) __host__ __device__

    UpperTriangularMatrix<Scalar, 1>
    UpperTriangularJTJ(const Eigen::Matrix<Scalar, ResidualDim, 1, Eigen::DontAlign> & jacobian,
                       const Eigen::Matrix<Scalar, ResidualDim, 1, Eigen::DontAlign> & rhoDoublePrimeJacobian) {
        return {jacobian.transpose() * rhoDoublePrimeJacobian};
    }

};

} // namespace internal




template <typename Scalar, int ModelDim>
struct LinearSystemSumFunctor {

    __attribute__((always_inline)) __host__ __device__
    LinearSystem<Scalar,ModelDim> operator()(const LinearSystem<Scalar,ModelDim> & lhs,
                                             const LinearSystem<Scalar,ModelDim> & rhs) {

        return lhs + rhs;

    }

};

namespace internal {

template <typename Scalar, int ModelDim, int Index>
struct SquareMatrixRowInitializer {

    static constexpr uint Row = ModelDim - Index;

    __attribute__((always_inline)) __host__ __device__

    static void Initialize(Eigen::Matrix <Scalar, ModelDim, ModelDim> & M,
                           const UpperTriangularMatrix<Scalar, ModelDim - Row> & upperTriangle) {

        M.template block<1, ModelDim - Row>(Row, Row) = upperTriangle.head;
        SquareMatrixRowInitializer<Scalar, ModelDim, Index - 1>::Initialize(M, upperTriangle.tail);

    }

};

template <typename Scalar, int ModelDim>
struct SquareMatrixRowInitializer<Scalar, ModelDim, 1> {

    static constexpr uint Row = ModelDim - 1;

    __attribute__((always_inline)) __host__ __device__

    static void Initialize(Eigen::Matrix <Scalar, ModelDim, ModelDim> & M,
                           const UpperTriangularMatrix<Scalar, 1> & upperTriangle) {

        M.template block<1, ModelDim - Row>(Row, Row) = upperTriangle.head;

    }

};

template <typename Scalar, int ModelDim>
struct SquareMatrixReconstructor {

    static __attribute__((always_inline)) __host__ __device__

    Eigen::Matrix <Scalar, ModelDim, ModelDim>
    Reconstruct(const UpperTriangularMatrix<Scalar, ModelDim> & upperTriangle) {

        Eigen::Matrix <Scalar, ModelDim, ModelDim> M;
        SquareMatrixRowInitializer<Scalar, ModelDim, ModelDim>::Initialize(M, upperTriangle);
        return M;

    }

};

template <typename Scalar, int D>
struct VectorAtomicAdder {

    __host__ __device__

    inline static
    void
    AtomicAdd(Scalar * destination, const Eigen::Matrix<Scalar, 1, D, Eigen::DontAlign | Eigen::RowMajor> & source) {

        if (source(0) != Scalar(0)) {
            Scalar val = source(0);
            AtomicAdd(destination, val);
        }

        VectorAtomicAdder<Scalar, D - 1>::AtomicAdd(destination + 1, source.template block<1, D - 1>(0, 1));

    }

};

template <typename Scalar>
struct VectorAtomicAdder<Scalar, 0> {

    __host__ __device__

    inline static
    void AtomicAdd(Scalar * /*destination*/,
                   const Eigen::Matrix<Scalar, 1, 0, Eigen::DontAlign | Eigen::RowMajor> & /*source*/) {}

};

template <typename Scalar, int D>
struct JTJAtomicAdder {

    __host__ __device__

    inline static
    void atomicAdd(UpperTriangularMatrix<Scalar, D> & destination, const UpperTriangularMatrix<Scalar, D> & source) {

        VectorAtomicAdder<Scalar, D>::AtomicAdd(destination.head.data(), source.head);

        JTJAtomicAdder<Scalar, D - 1>::AtomicAdd(destination.tail, source.tail);

    }

};

template <typename Scalar>
struct JTJAtomicAdder<Scalar, 1> {

    __host__ __device__

    inline static
    void AtomicAdd(UpperTriangularMatrix<Scalar, 1> & destination, const UpperTriangularMatrix<Scalar, 1> & source) {

        VectorAtomicAdder<Scalar, 1>::AtomicAdd(destination.head.data(), source.head);

    }

};

} // namespace internal

template <typename Scalar, int D>
struct LinearSystemAtomicAdder {

    __host__ __device__ inline static
    void AtomicAdd(LinearSystem<Scalar,D> & destination, const LinearSystem<Scalar,D> & source) {

        internal::JTJAtomicAdder<Scalar,D>::AtomicAdd(destination.JTJ, source.JTJ);

        internal::VectorAtomicAdder<Scalar,D>::AtomicAdd(destination.JTr.data(), source.JTr);

    }

};

template <typename Scalar, int D>
struct LinearSystemSolver {

    __host__ __device__ inline static
    Eigen::Matrix<Scalar, D, 1> Solve(const LinearSystem<Scalar, D> & system) {

        Eigen::Matrix<Scalar, D, D, Eigen::DontAlign> JTJ = internal::SquareMatrixReconstructor<Scalar, D>::Reconstruct(system.JTJ);

        return -JTJ.template selfadjointView<Eigen::Upper>().ldlt().solve(system.JTr);

    }

};

} // namespace vu