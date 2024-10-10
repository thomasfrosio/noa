#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Half.hpp"

// TODO Eventually, std::atomic_ref will replace everything...

#if defined(NOA_IS_GPU_CODE)
namespace noa::cuda::guts {
    NOA_FD i32 atomic_add(i32* address, i32 val) {
        return ::atomicAdd(address, val);
    }

    NOA_FD u32 atomic_add(u32* address, u32 val) {
        return ::atomicAdd(address, val);
    }

    NOA_FD u64 atomic_add(u64* address, u64 val) {
        return ::atomicAdd(reinterpret_cast<unsigned long long*>(address), val);
    }

    #if __CUDA_ARCH__ >= 700
    NOA_FD f16 atomic_add(f16* address, f16 val) {
        return f16(::atomicAdd(reinterpret_cast<__half*>(address), val.native()));
        // atomicCAS for ushort requires 700 as well, so I don't think there's an easy way to do atomics
        // on 16-bits values on 5.3 and 6.X devices...
    }
    #endif

    NOA_FD f32 atomic_add(f32* address, f32 val) {
        return ::atomicAdd(address, val);
    }

    NOA_ID f64 atomic_add(f64* address, f64 val) {
        #if __CUDA_ARCH__ < 600
        using ull = unsigned long long;
        auto* address_as_ull = (ull*) address;
        ull old = *address_as_ull;
        ull assumed;

        do {
            assumed = old;
            old = ::atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
        } while (assumed != old); // uses integer comparison to avoid hanging in case of NaN (since NaN != NaN)

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
namespace noa::cpu::guts {
    // FIXME C++20 atomic_ref, but lib clang doesn't support it yet
    template<typename Pointer, typename Value>
    NOA_FD void atomic_add(Pointer pointer, Value value) {
        if constexpr (nt::complex<Value>) {
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

namespace noa::guts {
    // Atomic add for CUDA and OpenMP.
    template<nt::pointer_numeric P>
    NOA_FHD void atomic_add(P pointer, nt::mutable_value_type_t<P> value) {
        #if defined(NOA_IS_GPU_CODE)
        ::noa::cuda::guts::atomic_add(pointer, value);
        #else
        ::noa::cpu::guts::atomic_add(pointer, value);
        #endif
    }

    template<typename... I, nt::atomic_addable_nd<sizeof...(I)> T>
    NOA_FHD void atomic_add(const T& input, nt::mutable_value_type_t<T> value, I... indices) {
        auto pointer = input.offset_pointer(input.get(), indices...);
        atomic_add(pointer, value);
    }
}
