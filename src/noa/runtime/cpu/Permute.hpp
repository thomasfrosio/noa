#pragma once

#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/cpu/Copy.hpp"

// TODO Use BLAS-like copy for out-of-place permute
// TODO Try https://stackoverflow.com/a/16743203

namespace noa::cpu::details {
    template<typename T, usize N>
    void permute_copy(
        const T* input, const Strides<isize, N>& input_strides,
        const Shape<isize, N>& input_shape,
        T* output, const Strides<isize, N>& output_strides,
        const Vec<i32, N>& permutation, i32 threads
    ) {
        NOA_ASSERT(input != output);
        const auto output_shape = input_shape.permute(permutation);
        const auto input_strides_permuted = input_strides.permute(permutation);
        copy(input, input_strides_permuted, output, output_strides, output_shape, threads);
    }

    template<typename T, usize N>
    void permute_inplace_two_innermost(Accessor<T, N> output, const Shape<isize, N>& shape) {
        // Swap upper triangle with lower triangle.
        if constexpr (N == 4) {
            for (isize i = 0; i < shape[0]; ++i) {
                for (isize j = 0; j < shape[1]; ++j) {
                    auto output_ij = output[i][j];
                    for (isize k = 0; k < shape[2]; ++k)
                        for (isize l = k + 1; l < shape[3]; ++l)
                            std::swap(output_ij(k, l), output_ij(l, k));
                }
            }
        } else if constexpr (N == 3) {
            for (isize i = 0; i < shape[0]; ++i) {
                auto output_i = output[i];
                for (isize j = 0; j < shape[1]; ++j)
                    for (isize k = j + 1; k < shape[2]; ++k)
                        std::swap(output_i(j, k), output_i(k, j));
            }
        } else if constexpr (N == 2) {
            for (isize i = 0; i < shape[0]; ++i)
                for (isize j = i + 1; j < shape[1]; ++j)
                    std::swap(output(i, j), output(j, i));
        }
    }
}

namespace noa::cpu {
    template<typename T, usize N>
        requires (N >= 2)
    void permute_copy(
        const T* input,
        const Strides<isize, N>& input_strides,
        const Shape<isize, N>& input_shape,
        T* output,
        const Strides<isize, N>& output_strides,
        const Vec<i32, N>& permutation,
        i32 threads
    ) {
        if (input == output) {
            const auto permuted_axes = permutation.cmp_ne(Vec<i32, N>::arange());
            check(sum(permuted_axes.template as<i32>()) == 2,
                  "In-place permutations require that only 2 axes are swapped, but got permutation={}",
                  permutation);
            // TODO if more than 2, do multiple inplace swaps

            // Move dimensions that are not permuted on the left side.
            auto shape = input_shape;
            for (usize i{}; i < N; ++i)
                if (not permuted_axes[i])
                    shape[i] = 1;
            const auto order = noa::squeeze_empty_dimensions_left(shape);
            const auto accessor = Accessor<T, 4>(output, output_strides.permute(order));

            details::permute_inplace_two_innermost(accessor, input_shape.permute(order));
        } else {
            details::permute_copy(input, input_strides, input_shape, output, output_strides, permutation, threads);
        }
    }
}
