#pragma once

#include <cuda_fp16.h>
#include "noa/gpu/cuda/Types.h"

namespace noa::cuda::util::traits {
    template<typename T>
    struct private_uninitialized_type { using type = T; };
    template<>
    struct private_uninitialized_type<half_t> { using type = ::half; };
    template<>
    struct private_uninitialized_type<chalf_t> { using type = ::half2; };
    template<>
    struct private_uninitialized_type<cfloat_t> { using type = ::float2; };
    template<>
    struct private_uninitialized_type<cdouble_t> { using type = ::double2; };

    // Static initialization of shared variables is illegal in CUDA. Some types (e.g. half_t) cannot be used with
    // the __shared__ attribute. This trait returns an equivalent type of T, i.e. same size and alignment,
    // meant to be used to declare static shared arrays/pointers. The returned type can be the same as T.
    // Once declared, this region of shared memory can be reinterpreted to T "safely". While these types are
    // very similar (again, same size, same alignment, standard layouts), we may be close to C++ undefined
    // behavior. However, this is CUDA and I doubt this would cause any issue (they reinterpret pointers to very
    // different types quite often in their examples, so give me a break).
    template<typename T>
    struct uninitialized_type { using type = typename private_uninitialized_type<T>::type; };
    template<typename T>
    using uninitialized_type_t = typename uninitialized_type<T>::type;
}

namespace noa::cuda::util::traits {
    // Aligned vector that generates vectorized load/store in CUDA.
    template<typename T, size_t VEC_SIZE>
    struct alignas(sizeof(T) * VEC_SIZE) aligned_vector_t {
        T val[VEC_SIZE];
    };
}
