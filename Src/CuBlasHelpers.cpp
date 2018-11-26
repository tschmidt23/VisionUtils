#include <vu/CuBlasHelpers.h>

namespace vu {

// assumes all matrices are col-major
template <typename Scalar>
void MatrixMatrixMultiply(const NDT::DeviceMatrix<Scalar> & A,
                          const cublasOperation_t transposeA,
                          const NDT::DeviceMatrix<Scalar> & B,
                          const cublasOperation_t transposeB,
                          NDT::DeviceMatrix<Scalar> & result,
                          const Scalar alpha, const Scalar beta) {

    const int numRows = transposeA == CUBLAS_OP_T ? A.DimensionSize(1) : A.DimensionSize(0);

    const int numCols = transposeB == CUBLAS_OP_T ? B.DimensionSize(0) : B.DimensionSize(1);

    const int interiorDimension = transposeA == CUBLAS_OP_T ? A.DimensionSize(0) : A.DimensionSize(1);

    {
        const int interiorDimensionB = transposeB == CUBLAS_OP_T ? B.DimensionSize(1) : B.DimensionSize(0);

        if (interiorDimension != interiorDimensionB) {
            throw std::runtime_error("interior dimension mismatch: " + std::to_string(interiorDimension) + "  vs  " +
                                     std::to_string(interiorDimensionB));
        }

        if (result.DimensionSize(0) != numRows) {
            throw std::runtime_error("num rows mismatch: " + std::to_string(result.DimensionSize(0)) + "  vs  " +
                                     std::to_string(numRows));
        }

        if (result.DimensionSize(1) != numCols) {
            throw std::runtime_error("num cols mismatch: " + std::to_string(result.DimensionSize(1)) + "  vs  " +
                                     std::to_string(numCols));
        }

    }

    const CuBlasHandle & cuBlasHandle = CuBlasHandle::GetInstance();

    //TODO: this will break for doubles
    const cublasStatus_t status = cublasSgemm_v2(cuBlasHandle, transposeA, transposeB,
                                                 numRows, numCols, interiorDimension,
                                                 &alpha,
                                                 A.Data(), A.DimensionSize(0),
                                                 B.Data(), B.DimensionSize(0),
                                                 &beta,
                                                 result.Data(), result.DimensionSize(0));

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasSgemm_v2 error");
    }

}

template void MatrixMatrixMultiply(const NDT::DeviceMatrix<float> &,
                                   const cublasOperation_t,
                                   const NDT::DeviceMatrix<float> &,
                                   const cublasOperation_t,
                                   NDT::DeviceMatrix<float> &,
                                   const float, const float);

// assumes all matrices are col-major
template <typename Scalar>
void MatrixVectorMultiply(const NDT::DeviceMatrix<Scalar> & A,
                          const cublasOperation_t transposeA,
                          const NDT::DeviceVector<Scalar> & B,
                          NDT::DeviceVector<Scalar> & result,
                          const Scalar alpha, const Scalar beta) {

    const int numRows = transposeA == CUBLAS_OP_T ? A.DimensionSize(1) : A.DimensionSize(0);

    const int numCols = transposeA == CUBLAS_OP_T ? A.DimensionSize(0) : A.DimensionSize(1);

    {

        if (B.Length() != numCols) {
            throw std::runtime_error(
                    "num cols mismatch: " + std::to_string(B.Length()) + "  vs  " + std::to_string(numCols));
        }

        if (result.Length() != numRows) {
            throw std::runtime_error(
                    "num rows mismatch: " + std::to_string(result.Length()) + "  vs  " + std::to_string(numRows));
        }

    }

    const CuBlasHandle & cuBlasHandle = CuBlasHandle::GetInstance();

    //TODO: this will break for doubles
    const cublasStatus_t status = cublasSgemv_v2(cuBlasHandle, transposeA,
                                                 numRows, numCols,
                                                 &alpha,
                                                 A.Data(), A.DimensionSize(0),
                                                 B.Data(), 1,
                                                 &beta,
                                                 result.Data(), 1);

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasSgemv_v2 error");
    }

}

template void MatrixVectorMultiply(const NDT::DeviceMatrix<float> &,
                                   const cublasOperation_t,
                                   const NDT::DeviceVector<float> &,
                                   NDT::DeviceVector<float> &,
                                   const float, const float);

} // namespace vu