#pragma once

#ifndef NOA_REDUCTIONS_INL_
#error "This is an internal header; it should not be included."
#endif

#include "noa/common/Exception.h"
#include "noa/cpu/math/Ewise.h"
#include "noa/cpu/memory/Set.h"

namespace noa::cpu::math {
    template<typename T, typename BinaryOp>
    void reduce(const T* input, size4_t stride, size4_t shape, T* output,
                BinaryOp binary_op, T init, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
            T reduce = init;
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            reduce = binary_op(reduce, input[at(i, j, k, l, stride)]);
            *output = reduce;
        });
    }

    template<typename T>
    void mean(const T* input, size4_t stride, size4_t shape, T* output, Stream& stream) {
        stream.enqueue([=, &stream]() {
            sum(input, stride, shape, output, stream);
            using value_t = noa::traits::value_type_t<T>;
            const auto count = static_cast<value_t>(shape.elements());
            *output /= count;
        });
    }

    template<int DDOF, typename T, typename U>
    void std(const T* input, size4_t stride, size4_t shape, U* output, Stream& stream) {
        stream.enqueue([=, &stream]() {
            var<DDOF>(input, stride, shape, output, stream);
            *output = noa::math::sqrt(*output);
        });
    }

    template<int DDOF, typename T, typename U>
    void statistics(const T* input, size4_t stride, size4_t shape,
                    T* output_sum, T* output_mean,
                    U* output_var, U* output_std,
                    Stream& stream) {
        // It is faster to call these one after the other than to merge everything into one loop.
        stream.enqueue([=, &stream]() {
            sum(input, stride, shape, output_sum, stream);
            *output_mean = *output_sum / static_cast<T>(shape.elements());
            var<DDOF>(input, stride, shape, output_var, stream);
            *output_std = noa::math::sqrt(*output_var);
        });
    }
}

namespace noa::cpu::math {
    template<int DDOF, typename T, typename U>
    void std(const T* input, size4_t input_stride, size4_t input_shape,
             U* output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        stream.enqueue([=, &stream]() {
            var<DDOF>(input, input_stride, input_shape, output, output_stride, output_shape, stream);
            if (any(input_shape != output_shape))
                math::ewise(output, output_stride, output, output_stride, output_shape, noa::math::sqrt_t{}, stream);
        });
    }
}
