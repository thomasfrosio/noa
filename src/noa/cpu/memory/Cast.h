#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"
#include "noa/cpu/utils/EwiseUnary.h"

namespace noa::cpu::memory {
    // Casts one array to another type.
    template<typename T, typename U>
    void cast(const shared_t<T[]>& input, const shared_t<U[]>& output,
              dim_t elements, bool clamp, Stream& stream) {
        NOA_ASSERT(input && output);
        stream.enqueue([=]() {
            const T* input_ = input.get();
            U* output_ = output.get();
            for (dim_t i = 0; i < elements; ++i, ++input_, ++output_)
                *output_ = clamp ? clamp_cast<U>(*input_) : static_cast<U>(*input_);
        });
    }

    // Casts one array to another type.
    template<typename T, typename U>
    void cast(const shared_t<T[]>& input, const dim4_t& input_strides,
              const shared_t<U[]>& output, const dim4_t& output_strides,
              const dim4_t& shape, bool clamp, Stream& stream) {
        NOA_ASSERT(all(shape > 0) && input && output);
        const auto threads_omp = stream.threads();
        stream.enqueue([=]() {
            const T* input_ptr = input.get();
            U* output_ptr = output.get();
            const auto cast_op = [=] (T value) { return clamp ? clamp_cast<U>(value) : static_cast<U>(value); };
            cpu::utils::ewiseUnary(input_ptr, input_strides,
                                   output_ptr, output_strides,
                                   shape, cast_op, threads_omp);
        });
    }
}
