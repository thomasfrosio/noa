#pragma once

#include <cuda_fp16.h>

#include "noa/common/Definitions.h"
#include "noa/common/Functors.h"
#include "noa/gpu/cuda/Types.h"

namespace noa::cuda::util::warp::details {
    template<typename T>
    constexpr bool is_valid_suffle_v = noa::traits::is_data_v<T> || noa::traits::is_any_v<T, half, half2>;
}

namespace noa::cuda::util::warp {
    using namespace noa;

    template <typename T, typename = std::enable_if_t<details::is_valid_suffle_v<T>>>
    NOA_FD T shuffle(T value, int32_t source, int32_t width = 32, uint32_t mask = 0xffffffff) {
        if constexpr (noa::traits::is_almost_same_v<chalf_t, T>) {
            __half2 tmp = __shfl_sync(mask, *reinterpret_cast<__half2*>(&value), source, width);
            return *reinterpret_cast<chalf_t*>(&tmp);
        } else if constexpr (std::is_same_v<half_t, T>) {
            return T(__shfl_sync(mask, value.native(), source, width));
        } else if constexpr (noa::traits::is_complex_v<T>) {
            return T(__shfl_sync(mask, value.real, source, width),
                     __shfl_sync(mask, value.imag, source, width));
        } else if (noa::traits::is_int_v<T> && sizeof(T) < 4) {
            return static_cast<T>(__shfl_sync(mask, static_cast<int32_t>(value), source, width));
        } else {
            return __shfl_sync(mask, value, source, width);
        }
        return T{}; // unreachable
    }

    template <typename T, typename = std::enable_if_t<details::is_valid_suffle_v<T>>>
    NOA_FD T shuffleDown(T value, uint32_t delta, int32_t width = 32, uint32_t mask = 0xffffffff) {
        if constexpr (noa::traits::is_almost_same_v<chalf_t, T>) {
            __half2 tmp = __shfl_down_sync(mask, *reinterpret_cast<__half2*>(&value), delta, width);
            return *reinterpret_cast<chalf_t*>(&tmp);
        } else if constexpr (std::is_same_v<half_t, T>) {
            return T(__shfl_down_sync(mask, value.native(), delta, width));
        } else if constexpr (noa::traits::is_complex_v<T>) {
            return T(__shfl_down_sync(mask, value.real, delta, width),
                     __shfl_down_sync(mask, value.imag, delta, width));
        } else if (noa::traits::is_int_v<T> && sizeof(T) < 4) {
            return static_cast<T>(__shfl_down_sync(mask, static_cast<int32_t>(value), delta, width));
        } else {
            return __shfl_down_sync(mask, value, delta, width);
        }
        return T{}; // unreachable
    }

    // Reduces one warp to one element. Returns the reduced value in tid 0 (undefined in other threads).
    // T:           Any data type.
    // value:       Per-thread value.
    // reduce_op:   Reduction operator.
    template<typename ReduceOp, typename T>
    NOA_ID T reduce(T value, ReduceOp reduce_op) {
        T other;
        for (int delta = 1; delta < 32; delta *= 2) {
            other = shuffleDown(value, delta);
            value = reduce_op(value, other);
        }
        return value;
    }

    template<typename value_t, typename offset_t, typename find_op_t>
    NOA_ID Pair<value_t, offset_t> find(value_t value, offset_t offset, find_op_t find_op) {
        using pair_t = Pair<value_t, offset_t>;
        pair_t current{value, offset};
        pair_t candidate;
        for (int delta = 1; delta < 32; delta *= 2) {
            candidate.first = shuffleDown(current.first, delta);
            candidate.second = shuffleDown(current.second, delta);
            current = find_op(current, candidate);
        }
        return current;
    }
}
