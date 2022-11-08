#pragma once

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"

namespace noa::cuda::utils::atomic {
    NOA_FD int add(int* address, int val) {
        return ::atomicAdd(address, val);
    }

    NOA_FD uint add(uint* address, uint val) {
        return ::atomicAdd(address, val);
    }

    NOA_FD unsigned long long add(unsigned long long* address, unsigned long long val) {
        return ::atomicAdd(address, val);
    }

    #if __CUDA_ARCH__ >= 700
    NOA_FD half_t add(half_t* address, half_t val) {
        return half_t(::atomicAdd(reinterpret_cast<__half*>(address), val.native()));
        // atomicCAS for ushort requires 700 as well, so I don't think there's an easy way to do atomics
        // on 16-bits values on 5.3 and 6.X devices...
    }
    #endif

    NOA_FD float add(float* address, float val) {
        return ::atomicAdd(address, val);
    }

    NOA_DEVICE double add(double* address, double val) {
        #if __CUDA_ARCH__ < 600
        using ull = unsigned long long int;
        auto* address_as_ull = (ull*) address;
        ull old = *address_as_ull;
        ull assumed;

        do {
            assumed = old;
            old = ::atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old); // uses integer comparison to avoid hang in case of NaN (since NaN != NaN)

        return __longlong_as_double(old); // like every other atomicAdd, return old
        #else
        return ::atomicAdd(address, val);
        #endif
    }

    template<typename T>
    NOA_FD Complex<T> add(Complex<T>* address, Complex<T> val) {
        return {add(&(address->real), val.real), add(&(address->imag), val.imag)};
    }
}
