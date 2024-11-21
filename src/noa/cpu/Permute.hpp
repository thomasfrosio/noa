#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Error.hpp"
#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/cpu/Copy.hpp"

// TODO Use BLAS-like copy for out-of-place permute
// TODO Try https://stackoverflow.com/a/16743203

namespace noa::cpu::guts {
    template<typename T>
    void permute_copy(
        const T* input, const Strides4<i64>& input_strides,
        const Shape4<i64>& input_shape,
        T* output, const Strides4<i64>& output_strides,
        const Vec4<i64>& permutation, i64 threads
    ) {
        NOA_ASSERT(input != output);
        const auto output_shape = ni::reorder(input_shape, permutation);
        const auto input_strides_permuted = ni::reorder(input_strides, permutation);
        copy(input, input_strides_permuted, output, output_strides, output_shape, threads);
    }

    template<typename T>
    void permute_inplace_0213(AccessorI64<T, 4> output, const Shape4<i64>& shape) {
        check(shape[2] == shape[1],
              "For a \"0213\" in-place permutation, shape[2] should be equal to shape[1], but got shape={}", shape);

        for (i64 i = 0; i < shape[0]; ++i) {
            for (i64 l = 0; l < shape[3]; ++l) {
                // Transpose YZ: swap bottom triangle with upper triangle.
                for (i64 j = 0; j < shape[1]; ++j)
                    for (i64 k = j + 1; k < shape[2]; ++k)
                        std::swap(output(i, j, k, l), output(i, k, j, l));
            }
        }
    }

    template<typename T>
    void permute_inplace_0132(AccessorI64<T, 4> output, const Shape4<i64>& shape) {
        check(shape[3] == shape[2],
              "For a \"0132\" in-place permutation, shape[3] should be equal to shape[2], but got shape={}", shape);

        for (i64 i = 0; i < shape[0]; ++i) {
            for (i64 j = 0; j < shape[1]; ++j) {
                // Transpose XY: swap upper triangle with lower triangle.
                for (i64 k = 0; k < shape[2]; ++k)
                    for (i64 l = k + 1; l < shape[3]; ++l)
                        std::swap(output(i, j, k, l), output(i, j, l, k));
            }
        }
    }

    template<typename T>
    void permute_inplace_0321(AccessorI64<T, 4> output, const Shape4<i64>& shape) {
        check(shape[3] == shape[1],
              "For a \"0321\" in-place permutation, shape[3] should be equal to shape[1], but got shape={}", shape);

        for (i64 i = 0; i < shape[0]; ++i) {
            for (i64 k = 0; k < shape[2]; ++k) {
                // Transpose XZ: swap upper triangle with lower triangle.
                for (i64 j = 0; j < shape[1]; ++j)
                    for (i64 l = j + 1; l < shape[3]; ++l)
                        std::swap(output(i, j, k, l), output(i, l, k, j));
            }
        }
    }
}

namespace noa::cpu {
    template<typename T>
    void permute_copy(
        const T* input,
        const Strides4<i64>& input_strides,
        const Shape4<i64>& input_shape,
        T* output,
        const Strides4<i64>& output_strides,
        const Vec4<i64>& permutation,
        i64 threads
    ) {
        if (input == output) {
            using accessor_t = AccessorI64<T, 4>;
            const auto idx = permutation[0] * 1000 + permutation[1] * 100 + permutation[2] * 10 + permutation[3];
            switch (idx) {
                case 123:
                    return;
                case 213:
                    return guts::permute_inplace_0213(accessor_t(output, output_strides), input_shape);
                case 132:
                    return guts::permute_inplace_0132(accessor_t(output, output_strides), input_shape);
                case 321:
                    return guts::permute_inplace_0321(accessor_t(output, output_strides), input_shape);
                default:
                    panic("The in-place permutation {} is not supported", permutation);
            }
        } else {
            guts::permute_copy(input, input_strides, input_shape, output, output_strides, permutation, threads);
        }
    }
}
