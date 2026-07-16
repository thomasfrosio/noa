#pragma once

#include "noa/runtime/core/Accessor.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/cpu/Copy.hpp"

// TODO Use BLAS-like copy for out-of-place permute
// TODO Try https://stackoverflow.com/a/16743203

namespace noa::cpu {
    template<typename T, usize N>
        requires (N >= 2)
    void permute_copy_hw(
        const T* input,
        const Strides<isize, N>& input_strides,
        const Shape<isize, N>& input_shape,
        T* output,
        const Strides<isize, N>& output_strides,
        i32 threads
    ) {
        if (input == output) {
            auto accessor = Accessor<T, N>(output, output_strides);
            if constexpr (N == 6) {
                for (isize i = 0; i < input_shape[0]; ++i) {
                    for (isize j = 0; j < input_shape[1]; ++j) {
                        for (isize k = 0; k < input_shape[2]; ++k) {
                            for (isize l = 0; l < input_shape[3]; ++l) {
                                auto accessor_ijkl = accessor[Vec{i, j, k, l}];
                                for (isize m = 0; m < input_shape[4]; ++m)
                                    for (isize n = m + 1; n < input_shape[5]; ++n)
                                        std::swap(accessor_ijkl(m, n), accessor_ijkl(n, m));
                            }
                        }
                    }
                }
            } else if constexpr (N == 5) {
                for (isize i = 0; i < input_shape[0]; ++i) {
                    for (isize j = 0; j < input_shape[1]; ++j) {
                        for (isize k = 0; k < input_shape[2]; ++k) {
                            auto accessor_ijk = accessor[Vec{i, j, k}];
                            for (isize l = 0; l < input_shape[3]; ++l)
                                for (isize m = l + 1; m < input_shape[4]; ++m)
                                    std::swap(accessor_ijk(l, m), accessor_ijk(m, l));
                        }
                    }
                }
            } else if constexpr (N == 4) {
                for (isize i = 0; i < input_shape[0]; ++i) {
                    for (isize j = 0; j < input_shape[1]; ++j) {
                        auto accessor_ij = accessor[i][j];
                        for (isize k = 0; k < input_shape[2]; ++k)
                            for (isize l = k + 1; l < input_shape[3]; ++l)
                                std::swap(accessor_ij(k, l), accessor_ij(l, k));
                    }
                }
            } else if constexpr (N == 3) {
                for (isize i = 0; i < input_shape[0]; ++i) {
                    auto accessor_i = accessor[i];
                    for (isize j = 0; j < input_shape[1]; ++j)
                        for (isize k = j + 1; k < input_shape[2]; ++k)
                            std::swap(accessor_i(j, k), accessor_i(k, j));
                }
            } else if constexpr (N == 2) {
                for (isize i = 0; i < input_shape[0]; ++i)
                    for (isize j = i + 1; j < input_shape[1]; ++j)
                        std::swap(accessor(i, j), accessor(j, i));
            }
        } else {
            const auto output_shape = input_shape.permute(Vec<i32, N>::arange().swap(N - 2, N - 1));
            const auto input_strides_permuted = input_strides.permute(Vec<i32, N>::arange().swap(N - 2, N - 1));
            copy(input, input_strides_permuted, output, output_strides, output_shape, threads);
        }
    }
}
