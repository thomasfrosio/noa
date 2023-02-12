#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Types.hpp"
#include "noa/cpu/Stream.hpp"
#include "noa/cpu/utils/EwiseUnary.hpp"

namespace noa::cpu::memory {
    // Casts one array to another type.
    template<typename Input, typename Output>
    void cast(const Shared<Input[]>& input,
              const Shared<Output[]>& output,
              i64 elements, bool clamp, Stream& stream) {
        NOA_ASSERT(input && output);
        stream.enqueue([=]() {
            const Input* input_ = input.get();
            Output* output_ = output.get();
            for (i64 i = 0; i < elements; ++i, ++input_, ++output_)
                *output_ = clamp ? clamp_cast<Output>(*input_) : static_cast<Output>(*input_);
        });
    }

    // Casts one array to another type.
    template<typename Input, typename Output>
    void cast(const Shared<Input[]>& input, const Strides4<i64>& input_strides,
              const Shared<Output[]>& output, const Strides4<i64>& output_strides,
              const Shape4<i64>& shape, bool clamp, Stream& stream) {
        NOA_ASSERT(noa::all(shape > 0) && input && output);
        const auto threads_omp = stream.threads();
        stream.enqueue([=]() {
            const Input* input_ptr = input.get();
            Output* output_ptr = output.get();
            cpu::utils::ewise_unary(
                    input_ptr, input_strides,
                    output_ptr, output_strides, shape,
                    [=](Input value) {
                        return clamp ?
                               clamp_cast<Output>(value) :
                               static_cast<Output>(value);
                    },
                    threads_omp);
        });
    }
}
