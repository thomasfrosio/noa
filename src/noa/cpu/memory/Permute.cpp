#include "noa/core/Assert.hpp"
#include "noa/cpu/memory/Copy.hpp"
#include "noa/cpu/memory/Permute.hpp"

// TODO Use BLAS copy for out-of-place permute
// TODO Try https://stackoverflow.com/a/16743203

namespace noa::cpu::memory::details {
    template<typename T>
    void permute(const T* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
                 T* output, const Strides4<i64>& output_strides,
                 const Vec4<i64>& permutation, i64 threads) {
        NOA_ASSERT(input != output);
        const auto output_shape = noa::indexing::reorder(input_shape, permutation);
        const auto input_strides_permuted = noa::indexing::reorder(input_strides, permutation);
        noa::cpu::memory::copy(input, input_strides_permuted, output, output_strides, output_shape, threads);
    }

    template<typename T>
    void permute_inplace_0213(T* output, const Strides4<i64>& strides, const Shape4<i64>& shape) {
        if (shape[2] != shape[1])
            NOA_THROW("For a \"0213\" in-place permutation, shape[2] should be equal to shape[1]. Got {}", shape);

        for (i64 i = 0; i < shape[0]; ++i) {
            for (i64 l = 0; l < shape[3]; ++l) {
                // Transpose YZ: swap bottom triangle with upper triangle.
                for (i64 j = 0; j < shape[1]; ++j)
                    for (i64 k = j + 1; k < shape[2]; ++k)
                        std::swap(output[noa::indexing::at(i, j, k, l, strides)],
                                  output[noa::indexing::at(i, k, j, l, strides)]);
            }
        }
    }

    template<typename T>
    void permute_inplace_0132(T* output, const Strides4<i64>& strides, const Shape4<i64>& shape) {
        if (shape[3] != shape[2])
            NOA_THROW("For a \"0132\" in-place permutation, shape[3] should be equal to shape[2]. Got {}", shape);

        for (i64 i = 0; i < shape[0]; ++i) {
            for (i64 j = 0; j < shape[1]; ++j) {
                // Transpose XY: swap upper triangle with lower triangle.
                for (i64 k = 0; k < shape[2]; ++k)
                    for (i64 l = k + 1; l < shape[3]; ++l)
                        std::swap(output[noa::indexing::at(i, j, k, l, strides)],
                                  output[noa::indexing::at(i, j, l, k, strides)]);
            }
        }
    }

    template<typename T>
    void permute_inplace_0321(T* output, const Strides4<i64>& strides, const Shape4<i64>& shape) {
        if (shape[3] != shape[1])
            NOA_THROW("For a \"0321\" in-place permutation, shape[3] should be equal to shape[1]. Got {}", shape);

        for (i64 i = 0; i < shape[0]; ++i) {
            for (i64 k = 0; k < shape[2]; ++k) {
                // Transpose XZ: swap upper triangle with lower triangle.
                for (i64 j = 0; j < shape[1]; ++j)
                    for (i64 l = j + 1; l < shape[3]; ++l)
                        std::swap(output[noa::indexing::at(i, j, k, l, strides)],
                                  output[noa::indexing::at(i, l, k, j, strides)]);
            }
        }
    }
}

namespace noa::cpu::memory::details {
    #define NOA_INSTANTIATE_TRANSPOSE_(T)                   \
    template void permute<T>(                               \
        const T*, const Strides4<i64>&, const Shape4<i64>&, \
        T*, const Strides4<i64>&, const Vec4<i64>&, i64);   \
    template void permute_inplace_0213<T>(                  \
        T*, const Strides4<i64>&, const Shape4<i64>&);      \
    template void permute_inplace_0132<T>(                  \
        T*, const Strides4<i64>&, const Shape4<i64>&);      \
    template void permute_inplace_0321<T>(                  \
        T*, const Strides4<i64>&, const Shape4<i64>&)

    NOA_INSTANTIATE_TRANSPOSE_(i8);
    NOA_INSTANTIATE_TRANSPOSE_(i16);
    NOA_INSTANTIATE_TRANSPOSE_(i32);
    NOA_INSTANTIATE_TRANSPOSE_(i64);
    NOA_INSTANTIATE_TRANSPOSE_(u8);
    NOA_INSTANTIATE_TRANSPOSE_(u16);
    NOA_INSTANTIATE_TRANSPOSE_(u32);
    NOA_INSTANTIATE_TRANSPOSE_(u64);
    NOA_INSTANTIATE_TRANSPOSE_(f16);
    NOA_INSTANTIATE_TRANSPOSE_(f32);
    NOA_INSTANTIATE_TRANSPOSE_(f64);
    NOA_INSTANTIATE_TRANSPOSE_(c16);
    NOA_INSTANTIATE_TRANSPOSE_(c32);
    NOA_INSTANTIATE_TRANSPOSE_(c64);
}
