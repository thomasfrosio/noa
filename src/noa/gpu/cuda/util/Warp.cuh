/// \file noa/gpu/cuda/util/Warp.cuh
/// \brief Warp utilities.
/// \author Thomas - ffyr2w
/// \date 13 Feb 2022
#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Functors.h"
#include "noa/gpu/cuda/Types.h"

namespace noa::cuda::util::warp {
    using namespace noa;

    template <typename T>
    NOA_FD T shuffleDown(T value, unsigned int delta, int width = 32, unsigned int mask = 0xffffffff) {
        return __shfl_down_sync(mask, value, delta, width);
    }

    /// Reduces one warp to one element.
    /// \tparam T           Any data type.
    /// \param[in] input    Memory to reduce. If \p tid is between 0 and 31,
    ///                     the first 32 elements from \p data are reduced.
    /// \param tid          Thread index. Usually from 0 to 31, included.
    /// \param reduce_op    Reduction operator.
    /// \return Reduced value in tid 0 (undefined in other threads).
    template<typename ReduceOp, typename T>
    NOA_ID T reduce(const T* input, uint tid, ReduceOp reduce_op) {
        T value = input[tid];
        T other;

        for (int delta = 1; delta < 32; delta *= 2) {
            if constexpr (noa::traits::is_almost_same_v<chalf_t, T>) {
                __half2 tmp = shuffleDown(*reinterpret_cast<__half2*>(&value), delta);
                other = *reinterpret_cast<chalf_t*>(&tmp);
            } else if constexpr (std::is_same_v<half_t, T>) {
                other = T(shuffleDown(value.native(), delta));
            } else if constexpr (noa::traits::is_complex_v<T>) {
                other = T(shuffleDown(value.real, delta), shuffleDown(value.imag, delta));
            } else if (noa::traits::is_int_v<T> && sizeof(T) < 4) {
                other = static_cast<T>(shuffleDown(static_cast<int>(value), delta));
            } else {
                other = shuffleDown(value, delta);
            }
            value = reduce_op(value, other);
        }
        return value;
    }
}
