#pragma once

#include <NDT/Tensor.h>
#include <cublas_v2.h>

namespace vu {

class CuBlasHandle {
public:

    static const CuBlasHandle & GetInstance() {

        static const CuBlasHandle handle;
        return handle;

    }

    CuBlasHandle() {

        cublasCreate_v2(&handle_);

    }

    operator cublasHandle_t() const { return handle_; }

    ~CuBlasHandle() {

        cublasDestroy_v2(handle_);

    }

private:

    cublasHandle_t handle_;

};

// assumes all matrices are col-major
template <typename Scalar>
void MatrixMatrixMultiply(const NDT::DeviceMatrix <Scalar> & A,
                          const cublasOperation_t transposeA,
                          const NDT::DeviceMatrix <Scalar> & B,
                          const cublasOperation_t transposeB,
                          NDT::DeviceMatrix <Scalar> & result,
                          const Scalar alpha = Scalar(1), const Scalar beta = Scalar(0));


// assumes all matrices are col-major
template <typename Scalar>
void MatrixVectorMultiply(const NDT::DeviceMatrix <Scalar> & A,
                          const cublasOperation_t transposeA,
                          const NDT::DeviceVector <Scalar> & B,
                          NDT::DeviceVector <Scalar> & result,
                          const Scalar alpha = Scalar(1), const Scalar beta = Scalar(0));

} // namespace vu