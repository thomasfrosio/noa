#pragma once

#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/cpu/Copy.hpp"

// TODO Use BLAS-like copy for out-of-place permute
// TODO Try https://stackoverflow.com/a/16743203

namespace noa::cpu::details {
    template<typename T>
    void permute_copy(
        const T* input, const Strides4& input_strides,
        const Shape4& input_shape,
        T* output, const Strides4& output_strides,
        const Vec<i32, 4>& permutation, i32 threads
    ) {
        NOA_ASSERT(input != output);
        const auto output_shape = input_shape.permute(permutation);
        const auto input_strides_permuted = input_strides.permute(permutation);
        copy(input, input_strides_permuted, output, output_strides, output_shape, threads);
    }

    template<typename T>
    void permute_inplace_0213(Accessor<T, 4> output, const Shape4& shape) {
        check(shape[2] == shape[1],
              "For a \"0213\" in-place permutation, shape[2] should be equal to shape[1], but got shape={}", shape);

        for (isize i = 0; i < shape[0]; ++i) {
            for (isize l = 0; l < shape[3]; ++l) {
                // Transpose YZ: swap bottom triangle with upper triangle.
                for (isize j = 0; j < shape[1]; ++j)
                    for (isize k = j + 1; k < shape[2]; ++k)
                        std::swap(output(i, j, k, l), output(i, k, j, l));
            }
        }
    }

    template<typename T>
    void permute_inplace_0132(Accessor<T, 4> output, const Shape4& shape) {
        check(shape[3] == shape[2],
              "For a \"0132\" in-place permutation, shape[3] should be equal to shape[2], but got shape={}", shape);

        for (isize i = 0; i < shape[0]; ++i) {
            for (isize j = 0; j < shape[1]; ++j) {
                // Transpose XY: swap upper triangle with lower triangle.
                for (isize k = 0; k < shape[2]; ++k)
                    for (isize l = k + 1; l < shape[3]; ++l)
                        std::swap(output(i, j, k, l), output(i, j, l, k));
            }
        }
    }

    template<typename T>
    void permute_inplace_0321(Accessor<T, 4> output, const Shape4& shape) {
        check(shape[3] == shape[1],
              "For a \"0321\" in-place permutation, shape[3] should be equal to shape[1], but got shape={}", shape);

        for (isize i = 0; i < shape[0]; ++i) {
            for (isize k = 0; k < shape[2]; ++k) {
                // Transpose XZ: swap upper triangle with lower triangle.
                for (isize j = 0; j < shape[1]; ++j)
                    for (isize l = j + 1; l < shape[3]; ++l)
                        std::swap(output(i, j, k, l), output(i, l, k, j));
            }
        }
    }
}

namespace noa::cpu {
    template<typename T>
    void permute_copy(
        const T* input,
        const Strides4& input_strides,
        const Shape4& input_shape,
        T* output,
        const Strides4& output_strides,
        const Vec<i32, 4>& permutation,
        i32 threads
    ) {
        if (input == output) {
            const auto idx = permutation[0] * 1000 + permutation[1] * 100 + permutation[2] * 10 + permutation[3];
            switch (idx) {
                case 123:
                    return;
                case 213:
                    return details::permute_inplace_0213(Accessor<T, 4>(output, output_strides), input_shape);
                case 132:
                    return details::permute_inplace_0132(Accessor<T, 4>(output, output_strides), input_shape);
                case 321:
                    return details::permute_inplace_0321(Accessor<T, 4>(output, output_strides), input_shape);
                default:
                    panic("The in-place permutation {} is not supported", permutation);
            }
        } else {
            details::permute_copy(input, input_strides, input_shape, output, output_strides, permutation, threads);
        }
    }
}
