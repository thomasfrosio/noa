#pragma once

#include "noa/core/Definitions.hpp"
#include "noa/core/Exception.hpp"
#include "noa/core/Types.hpp"
#include "noa/cpu/Stream.hpp"

namespace noa::cpu::memory::details {
    template<typename T>
    void permute(const T*, const Strides4<i64>&, const Shape4<i64>&, T*, const Strides4<i64>&, const Vec4<i64>&);
    template<typename T>
    void permute_inplace_0213(T*, const Strides4<i64>&, const Shape4<i64>&);
    template<typename T>
    void permute_inplace_0132(T*, const Strides4<i64>&, const Shape4<i64>&);
    template<typename T>
    void permute_inplace_0321(T*, const Strides4<i64>&, const Shape4<i64>&);
}

namespace noa::cpu::memory {
    // Permutes, in memory, the axes of an array.
    template<typename Value, typename = std::enable_if_t<traits::is_restricted_numeric_v<Value>>>
    void permute(const Shared<Value[]>& input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
                 const Shared<Value[]>& output, const Strides4<i64>& output_strides,
                 const Vec4<i64>& permutation,
                 Stream& stream) {
        if (noa::any(permutation > 3) || noa::math::sum(permutation) != 6)
            NOA_THROW("Permutation {} is not valid", permutation);

        NOA_ASSERT(input && output && noa::all(input_shape > 0));
        if (input == output) {
            const auto idx = permutation[0] * 1000 + permutation[1] * 100 + permutation[2] * 10 + permutation[3];
            switch (idx) {
                case 123:
                    return;
                case 213:
                    return stream.enqueue([=]() {
                        details::permute_inplace_0213<Value>(output.get(), output_strides, input_shape);
                    });
                case 132:
                    return stream.enqueue([=]() {
                        details::permute_inplace_0132<Value>(output.get(), output_strides, input_shape);
                    });
                case 321:
                    return stream.enqueue([=]() {
                        details::permute_inplace_0321<Value>(output.get(), output_strides, input_shape);
                    });
                default:
                    NOA_THROW("The in-place permutation {} is not supported", permutation);
            }
        } else {
            return stream.enqueue([=]() {
                details::permute<Value>(
                        input.get(), input_strides, input_shape,
                        output.get(), output_strides, permutation);
            });
        }
    }
}
