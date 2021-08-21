/// \file noa/gpu/cuda/util/Atomic.h
/// \brief Device atomic adds.
/// \author Thomas - ffyr2w
/// \date 20 Aug 2021

#pragma once

// These are device only functions and should only be
// compiled if the compilation is steered by nvcc.
#ifdef __CUDACC__

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"

namespace noa::cuda::atomic {
    NOA_FD int add(int* address, int val) {
        return ::atomicAdd(address, val);
    }
    NOA_FD uint add(uint* address, uint val) {
        return ::atomicAdd(address, val);
    }
    NOA_FD unsigned long long add(unsigned long long* address, unsigned long long val) {
        return ::atomicAdd(address, val);
    }
    NOA_FD float add(float* address, float val) {
        return ::atomicAdd(address, val);
    }

#if __CUDA_ARCH__ < 600
    NOA_DEVICE double add(double* address, double val) {
        using ull = unsigned long long int;
        auto* address_as_ull = static_cast<ull*>(address);
        ull old = *address_as_ull;
        ull assumed;

        do {
            assumed = old;
            old = ::atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old); // uses integer comparison to avoid hang in case of NaN (since NaN != NaN)

        return __longlong_as_double(old); // like every other atomicAdd, return old
    }
#else
    NOA_FD double add(double* address, double val) { return ::atomicAdd(address, val); }
#endif
}

#endif // __CUDACC__
