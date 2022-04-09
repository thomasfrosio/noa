#pragma once

#ifndef NOA_REDUCTIONS_INL_
#error "This is an internal header; it should not be included."
#endif

#include "noa/common/Exception.h"
#include "noa/cpu/math/Ewise.h"
#include "noa/cpu/memory/Set.h"

namespace noa::cpu::math {
    template<typename T, typename BinaryOp>
    T reduce(const shared_t<T[]>& input, size4_t stride, size4_t shape,
             BinaryOp binary_op, T init, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        T output;
        stream.enqueue([=, &output]() {
            T reduce = init;
            const T* iptr = input.get();
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            reduce = binary_op(reduce, iptr[indexing::at(i, j, k, l, stride)]);
            output = reduce;
        });
        stream.synchronize();
        return output;
    }

    template<typename T>
    T mean(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        T output = sum(input, stride, shape, stream);
        using value_t = noa::traits::value_type_t<T>;
        const auto count = static_cast<value_t>(shape.elements());
        return output / count;
    }

    template<int DDOF, typename T, typename U>
    U std(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        const U output = var<DDOF>(input, stride, shape, stream);
        return noa::math::sqrt(output);
    }

    template<int DDOF, typename T, typename U>
    std::tuple<T, T, U, U> statistics(const shared_t<T[]>& input, size4_t stride, size4_t shape, Stream& stream) {
        // It is faster to call these one after the other than to merge everything into one loop.
        const T output_sum = sum(input, stride, shape, stream);
        const T output_mean = output_sum / static_cast<T>(shape.elements());
        const U output_var = var<DDOF>(input, stride, shape, stream);
        const U output_std = noa::math::sqrt(output_var);
        return {output_sum, output_mean, output_var, output_std};
    }
}

namespace noa::cpu::math {
    template<int DDOF, typename T, typename U>
    void std(const shared_t<T[]>& input, size4_t input_stride, size4_t input_shape,
             const shared_t<U[]>& output, size4_t output_stride, size4_t output_shape, Stream& stream) {
        var<DDOF>(input, input_stride, input_shape, output, output_stride, output_shape, stream);
        if (any(input_shape != output_shape))
            math::ewise<U, U>(output, output_stride, output, output_stride, output_shape, noa::math::sqrt_t{}, stream);
    }
}
