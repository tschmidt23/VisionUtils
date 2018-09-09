#pragma once

#include <cuda_runtime.h>

namespace vu {

template <typename Scalar>
inline __device__ Scalar AtomicAdd(Scalar * address, Scalar val) {

return atomicAdd(address,val);

}

#ifdef __CUDACC__
template <>
inline __device__ double AtomicAdd(double * address, double val) {
    unsigned long long int * address_as_ull =
            (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif // __CUDACC__

template <typename I>
__host__ __device__
inline I IntDivideAndCeil(I numerator, I denominator) {
    return (numerator + denominator - 1) / denominator;
}


} // namespace vu