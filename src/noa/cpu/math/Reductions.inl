#pragma once

#ifndef NOA_REDUCTIONS_INL_
#error "This is an internal header; it should not be included."
#endif

#include "noa/cpu/math/Ewise.h"
#include "noa/cpu/memory/Set.h"

namespace noa::cpu::math {
    template<typename T, typename BinaryOp>
    void reduce(const T* input, size4_t input_stride, size4_t shape, T* outputs,
                BinaryOp binary_op, T init, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
            for (size_t i = 0; i < shape[0]; ++i) {
                T reduce = init;
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            reduce = binary_op(reduce, input[at(i, j, k, l, input_stride)]);
                outputs[i] = reduce;
            }
        });
    }

    template<typename T>
    void min(const T* input, size4_t stride, size4_t shape, T* outputs, Stream& stream) {
        return reduce(input, stride, shape, outputs, noa::math::min_t{}, noa::math::Limits<T>::max(), stream);
    }

    template<typename T>
    void max(const T* input, size4_t stride, size4_t shape, T* outputs, Stream& stream) {
        return reduce(input, stride, shape, outputs, noa::math::max_t{}, noa::math::Limits<T>::min(), stream);
    }

    template<typename T>
    void mean(const T* input, size4_t stride, size4_t shape, T* outputs, Stream& stream) {
        stream.enqueue([=, &stream]() {
            sum(input, stride, shape, outputs, stream);
            using value_t = noa::traits::value_type_t<T>;
            const auto count = static_cast<value_t>(shape[1] * shape[2] * shape[3]);
            for (size_t batch = 0; batch < shape[0]; ++batch)
                outputs[batch] /= count;
        });
    }

    template<typename T>
    void std(const T* input, size4_t stride, size4_t shape, T* outputs, Stream& stream) {
        stream.enqueue([=, &stream]() {
            var(input, stride, shape, outputs, shape[0], stream);
            for (size_t batch = 0; batch < shape[0]; ++batch)
                outputs[batch] = noa::math::sqrt(outputs[batch]);
        });
    }

    template<typename T>
    void statistics(const T* input, size4_t stride, size4_t shape,
                    T* output_mins, T* output_maxs,
                    T* output_sums, T* output_means,
                    T* output_vars, T* output_stds,
                    Stream& stream) {
        // It is faster to call these one after the other than to merge everything into one loop.
        stream.enqueue([=, &stream]() {
            const auto count = static_cast<T>(shape[1] * shape[2] * shape[3]);
            if (output_mins)
                min(input, stride, shape, output_mins, stream);
            if (output_maxs)
                max(input, stride, shape, output_maxs, stream);
            if (output_sums) {
                sum(input, stride, shape, output_sums, stream);
                if (output_means)
                    for (size_t batch = 0; batch < shape[0]; ++batch)
                        output_means[batch] = output_sums[batch] / count;
            } else if (output_means) {
                sum(input, stride, shape, output_means, stream);
                for (size_t batch = 0; batch < shape[0]; ++batch)
                    output_means[batch] = output_means[batch] / count;
            }
            if (output_vars) {
                var(input, stride, shape, output_vars, stream);
                if (output_stds)
                    for (size_t batch = 0; batch < shape[0]; ++batch)
                        output_stds[batch] = noa::math::sqrt(output_vars[batch]);
            } else if (output_stds) {
                var(input, stride, shape, output_stds, stream);
                for (size_t batch = 0; batch < shape[0]; ++batch)
                    output_stds[batch] = noa::math::sqrt(output_stds[batch]);
            }
        });
    }

    template<typename T, typename BinaryOp>
    void reduce(const T* input, size4_t input_stride, size4_t input_shape,
                T* output, size4_t output_stride, size4_t output_shape,
                BinaryOp binary_op, T init, Stream& stream) {
        NOA_ASSERT(all(input_shape >= output_shape));
        const bool4_t mask{input_shape == output_shape};
        const size4_t stride{size4_t{mask} * output_stride}; // don't increment the reduced dimensions
        cpu::memory::set(output, stride, output_shape, init, stream);
        cpu::math::ewise(input, input_stride, output, stride, output, stride, input_shape, binary_op, stream);
    }
}
