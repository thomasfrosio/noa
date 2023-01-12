#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/traits/BaseTypes.h"
#include "noa/common/types/Accessor.h"
#include "noa/common/types/Half.h"

#if defined(__CUDA_ARCH__)
namespace noa::cuda::details {
    NOA_FD int atomicAdd(int* address, int val) {
        return ::atomicAdd(address, val);
    }

    NOA_FD uint atomicAdd(uint* address, uint val) {
        return ::atomicAdd(address, val);
    }

    NOA_FD unsigned long long atomicAdd(unsigned long long* address, unsigned long long val) {
        return ::atomicAdd(address, val);
    }

    #if __CUDA_ARCH__ >= 700
    NOA_FD half_t atomicAdd(half_t* address, half_t val) {
        return half_t(::atomicAdd(reinterpret_cast<__half*>(address), val.native()));
        // atomicCAS for ushort requires 700 as well, so I don't think there's an easy way to do atomics
        // on 16-bits values on 5.3 and 6.X devices...
    }
    #endif

    NOA_FD float atomicAdd(float* address, float val) {
        return ::atomicAdd(address, val);
    }

    NOA_ID double atomicAdd(double* address, double val) {
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
    NOA_FD Complex<T> atomicAdd(Complex<T>* address, Complex<T> val) {
        return {atomicAdd(&(address->real), val.real), atomicAdd(&(address->imag), val.imag)};
    }
}
#else
namespace noa::cpu::details {
    template<typename Pointer, typename Value>
    NOA_FD void atomicAdd(Pointer pointer, Value value) {
        if constexpr (noa::traits::is_complex_v<Value>) {
            #pragma omp atomic
            (*pointer)[0] += value[0];
            #pragma omp atomic
            (*pointer)[1] += value[1];
        } else {
            #pragma omp atomic
            *pointer += value;
        }
    }
}
#endif

namespace noa::details {
    // Atomic add for CUDA and OpenMP.
    template<typename Pointer, typename Value,
             typename = std::enable_if_t<std::is_pointer_v<Pointer> && noa::traits::is_data_v<Value>>>
    NOA_FHD void atomicAdd(Pointer pointer, Value value) {
        #if defined(__CUDA_ARCH__)
        ::noa::cuda::details::atomicAdd(pointer, value);
        #else
        ::noa::cpu::details::atomicAdd(pointer, value);
        #endif
    }

    template<typename Value, AccessorTraits TRAITS, typename Offset, typename Index,
             typename = std::enable_if_t<noa::traits::is_int_v<Index> && noa::traits::is_data_v<Value>>>
    NOA_FHD void atomicAdd(const Accessor<Value, 1, Offset, TRAITS>& accessor,
                           Index i,
                           Value value) {
        auto* pointer = accessor.offsetPointer(accessor.get(), i);
        atomicAdd(pointer, value);
    }

    template<typename Value, AccessorTraits TRAITS, typename Offset, typename Index,
             typename = std::enable_if_t<noa::traits::is_int_v<Index> && noa::traits::is_data_v<Value>>>
    NOA_FHD void atomicAdd(const Accessor<Value, 2, Offset, TRAITS>& accessor,
                           Index i, Index j,
                           Value value) {
        auto* pointer = accessor.offsetPointer(accessor.get(), i, j);
        atomicAdd(pointer, value);
    }

    template<typename Value, AccessorTraits TRAITS, typename Offset, typename Index,
             typename = std::enable_if_t<noa::traits::is_int_v<Index> && noa::traits::is_data_v<Value>>>
    NOA_FHD void atomicAdd(const Accessor<Value, 3, Offset, TRAITS>& accessor,
                           Index i, Index j, Index k,
                           Value value) {
        auto* pointer = accessor.offsetPointer(accessor.get(), i, j, k);
        atomicAdd(pointer, value);
    }

    template<typename Value, AccessorTraits TRAITS, typename Offset, typename Index,
             typename = std::enable_if_t<noa::traits::is_int_v<Index> && noa::traits::is_data_v<Value>>>
    NOA_FHD void atomicAdd(const Accessor<Value, 4, Offset, TRAITS>& accessor,
                           Index i, Index j, Index k, Index l,
                           Value value) {
        auto* pointer = accessor.offsetPointer(accessor.get(), i, j, k, l);
        atomicAdd(pointer, value);
    }
}
