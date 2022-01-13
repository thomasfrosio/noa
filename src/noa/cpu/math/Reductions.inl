#pragma once

#ifndef NOA_REDUCTIONS_INL_
#error "This shoud not be directly included"
#endif

namespace noa::cpu::math {
    template<typename T, typename BinaryOp>
    void reduce(const T* inputs, size3_t input_pitch, size3_t shape, T* outputs, size_t batches,
                BinaryOp binary_op, T init, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        const size_t offset = elements(input_pitch);
        stream.enqueue([=]() {
            for (size_t batch = 0; batch < batches; ++batch) {
                T reduce = init;
                for (size_t z = 0; z < shape.z; ++z)
                    for (size_t y = 0; y < shape.y; ++y)
                        for (size_t x = 0; x < shape.x; ++x)
                            reduce = binary_op(reduce, inputs[offset * batch + index(x, y, z, input_pitch)]);
                outputs[batch] = reduce;
            }
        });
    }

    template<typename T>
    void min(const T* inputs, size3_t input_pitch, size3_t shape, T* outputs, size_t batches, Stream& stream) {
        return reduce(inputs, input_pitch, shape, outputs, batches, noa::math::min_t{},
                      noa::math::Limits<T>::max(), stream);
    }

    template<typename T>
    void max(const T* inputs, size3_t input_pitch, size3_t shape, T* outputs, size_t batches, Stream& stream) {
        return reduce(inputs, input_pitch, shape, outputs, batches, noa::math::max_t{},
                      noa::math::Limits<T>::min(), stream);
    }

    template<typename T>
    void mean(const T* inputs, size3_t input_pitch, size3_t shape, T* outputs, size_t batches, Stream& stream) {
        stream.enqueue([=, &stream]() {
            sum(inputs, input_pitch, shape, outputs, batches, stream);
            using value_t = noa::traits::value_type_t<T>;
            const auto count = static_cast<value_t>(elements(shape));
            for (size_t batch = 0; batch < batches; ++batch)
                outputs[batch] /= count;
        });
    }

    template<typename T>
    void std(const T* inputs, size3_t input_pitch, size3_t shape, T* outputs, size_t batches, Stream& stream) {
        stream.enqueue([=, &stream]() {
            var(inputs, input_pitch, shape, outputs, batches, stream);
            for (size_t batch = 0; batch < batches; ++batch)
                outputs[batch] = noa::math::sqrt(outputs[batch]);
        });
    }

    template<typename T>
    void statistics(const T* inputs, size3_t input_pitch, size3_t shape,
                    T* output_mins, T* output_maxs,
                    T* output_sums, T* output_means,
                    T* output_vars, T* output_stds,
                    size_t batches, Stream& stream) {
        // It is faster to call these one after the other than to merge everything into one loop.
        stream.enqueue([=, &stream]() {
            const auto count = static_cast<T>(elements(shape));
            if (output_mins)
                min(inputs, input_pitch, shape, output_mins, batches, stream);
            if (output_maxs)
                max(inputs, input_pitch, shape, output_maxs, batches, stream);
            if (output_sums) {
                sum(inputs, input_pitch, shape, output_sums, batches, stream);
                if (output_means)
                    for (size_t batch = 0; batch < batches; ++batch)
                        output_means[batch] = output_sums[batch] / count;
            } else if (output_means) {
                sum(inputs, input_pitch, shape, output_means, batches, stream);
                for (size_t batch = 0; batch < batches; ++batch)
                    output_means[batch] = output_means[batch] / count;
            }
            if (output_vars) {
                var(inputs, input_pitch, shape, output_vars, batches, stream);
                if (output_stds)
                    for (size_t batch = 0; batch < batches; ++batch)
                        output_stds[batch] = noa::math::sqrt(output_vars[batch]);
            } else if (output_stds) {
                var(inputs, input_pitch, shape, output_stds, batches, stream);
                for (size_t batch = 0; batch < batches; ++batch)
                    output_stds[batch] = noa::math::sqrt(output_stds[batch]);
            }
        });
    }
}
