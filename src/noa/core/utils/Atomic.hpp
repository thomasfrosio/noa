#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/traits/Numerics.hpp"
#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Half.hpp"

#if defined(__CUDA_ARCH__)
namespace noa::cuda::details {
    NOA_FD int atomic_add(int* address, int val) {
        return ::atomicAdd(address, val);
    }

    NOA_FD uint atomic_add(uint* address, uint val) {
        return ::atomicAdd(address, val);
    }

    NOA_FD unsigned long long atomic_add(unsigned long long* address, unsigned long long val) {
        return ::atomicAdd(address, val);
    }

    #if __CUDA_ARCH__ >= 700
    NOA_FD Half atomic_add(Half* address, Half val) {
        return Half(::atomicAdd(reinterpret_cast<__half*>(address), val.native()));
        // atomicCAS for ushort requires 700 as well, so I don't think there's an easy way to do atomics
        // on 16-bits values on 5.3 and 6.X devices...
    }
    #endif

    NOA_FD float atomic_add(float* address, float val) {
        return ::atomicAdd(address, val);
    }

    NOA_ID double atomic_add(double* address, double val) {
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
    NOA_FD Complex<T> atomic_add(Complex<T>* address, Complex<T> val) {
        return {atomicAdd(&(address->real), val.real), atomicAdd(&(address->imag), val.imag)};
    }
}
#else
namespace noa::cpu::details {
    // FIXME C++20 atomic_ref
    template<typename Pointer, typename Value>
    NOA_FD void atomic_add(Pointer pointer, Value value) {
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
             typename = std::enable_if_t<std::is_pointer_v<Pointer> && noa::traits::is_numeric_v<Value>>>
    NOA_FHD void atomic_add(Pointer pointer, Value value) {
        #if defined(__CUDA_ARCH__)
        ::noa::cuda::details::atomic_add(pointer, value);
        #else
        ::noa::cpu::details::atomic_add(pointer, value);
        #endif
    }

    template<typename Value,
             typename Offset,
             typename Index,
             PointerTraits PointerTrait,
             StridesTraits StridesTrait,
             typename = std::enable_if_t<noa::traits::is_int_v<Index> && noa::traits::is_numeric_v<Value>>>
    NOA_FHD void atomic_add(const Accessor<Value, 1, Offset, PointerTrait, StridesTrait>& accessor,
                            Index i,
                            Value value) {
        auto* pointer = accessor.offset_pointer(accessor.get(), i);
        atomic_add(pointer, value);
    }

    template<typename Value,
             typename Offset,
             typename Index,
             PointerTraits PointerTrait,
             StridesTraits StridesTrait,
             typename = std::enable_if_t<noa::traits::is_int_v<Index> && noa::traits::is_numeric_v<Value>>>
    NOA_FHD void atomic_add(const Accessor<Value, 2, Offset, PointerTrait, StridesTrait>& accessor,
                            Index i, Index j,
                            Value value) {
        auto* pointer = accessor.offset_pointer(accessor.get(), i, j);
        atomic_add(pointer, value);
    }

    template<typename Value,
             typename Offset,
             typename Index,
             PointerTraits PointerTrait,
             StridesTraits StridesTrait,
             typename = std::enable_if_t<noa::traits::is_int_v<Index> && noa::traits::is_numeric_v<Value>>>
    NOA_FHD void atomic_add(const Accessor<Value, 3, Offset, PointerTrait, StridesTrait>& accessor,
                            Index i, Index j, Index k,
                            Value value) {
        auto* pointer = accessor.offset_pointer(accessor.get(), i, j, k);
        atomic_add(pointer, value);
    }

    template<typename Value,
             typename Offset,
             typename Index,
             PointerTraits PointerTrait,
             StridesTraits StridesTrait,
             typename = std::enable_if_t<noa::traits::is_int_v<Index> && noa::traits::is_numeric_v<Value>>>
    NOA_FHD void atomic_add(const Accessor<Value, 4, Offset, PointerTrait, StridesTrait>& accessor,
                            Index i, Index j, Index k, Index l,
                            Value value) {
        auto* pointer = accessor.offset_pointer(accessor.get(), i, j, k, l);
        atomic_add(pointer, value);
    }
}
