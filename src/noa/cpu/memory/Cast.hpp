#pragma once

#include "noa/core/Types.hpp"
#include "noa/cpu/utils/EwiseUnary.hpp"

namespace noa::cpu::memory {
    // Casts one array to another type.
    template<typename Input, typename Output>
    void cast(const Input* input,
              Output* output,
              i64 elements, bool clamp) {
        NOA_ASSERT(input && output);
        for (i64 i = 0; i < elements; ++i, ++input, ++output)
            *output = clamp ? clamp_cast<Output>(*input) : static_cast<Output>(*input);
    }

    // Casts one array to another type.
    template<typename Input, typename Output>
    void cast(const Input* input, const Strides4<i64>& input_strides,
              Output* output, const Strides4<i64>& output_strides,
              const Shape4<i64>& shape, bool clamp, i64 threads) {
        NOA_ASSERT(noa::all(shape > 0) && input && output);
        cpu::utils::ewise_unary(
                input, input_strides,
                output, output_strides, shape,
                [=](Input value) {
                    return clamp ?
                           clamp_cast<Output>(value) :
                           static_cast<Output>(value);
                },
                threads);
    }
}
