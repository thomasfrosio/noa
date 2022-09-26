#include "noa/common/Assert.h"
#include "noa/cpu/memory/Copy.h"
#include "noa/cpu/memory/Permute.h"

// TODO Use BLAS copy for out-of-place permute
// TODO Try https://stackoverflow.com/a/16743203

namespace noa::cpu::memory::details {
    template<typename T>
    void permute(const T* input, dim4_t input_strides, dim4_t input_shape,
                 T* output, dim4_t output_strides, dim4_t permutation) {
        NOA_ASSERT(input != output);
        const dim4_t output_shape = indexing::reorder(input_shape, permutation);
        const dim4_t input_strides_permuted = indexing::reorder(input_strides, permutation);
        copy(input, input_strides_permuted, output, output_strides, output_shape);
    }
}

namespace noa::cpu::memory::details::inplace {
    template<typename T>
    void permute0213(T* output, dim4_t strides, dim4_t shape) {
        if (shape[2] != shape[1])
            NOA_THROW("For a \"0213\" in-place permutation, shape[2] should be equal to shape[1]. Got {}", shape);

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t l = 0; l < shape[3]; ++l) {

                // Transpose YZ: swap bottom triangle with upper triangle.
                for (dim_t j = 0; j < shape[1]; ++j)
                    for (dim_t k = j + 1; k < shape[2]; ++k)
                        std::swap(output[indexing::at(i, j, k, l, strides)],
                                  output[indexing::at(i, k, j, l, strides)]);
            }
        }
    }

    template<typename T>
    void permute0132(T* output, dim4_t strides, dim4_t shape) {
        if (shape[3] != shape[2])
            NOA_THROW("For a \"0132\" in-place permutation, shape[3] should be equal to shape[2]. Got {}", shape);

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t j = 0; j < shape[1]; ++j) {

                // Transpose XY: swap upper triangle with lower triangle.
                for (dim_t k = 0; k < shape[2]; ++k)
                    for (dim_t l = k + 1; l < shape[3]; ++l)
                        std::swap(output[indexing::at(i, j, k, l, strides)],
                                  output[indexing::at(i, j, l, k, strides)]);
            }
        }
    }

    template<typename T>
    void permute0321(T* output, dim4_t strides, dim4_t shape) {
        if (shape[3] != shape[1])
            NOA_THROW("For a \"0321\" in-place permutation, shape[3] should be equal to shape[1]. Got {}", shape);

        for (dim_t i = 0; i < shape[0]; ++i) {
            for (dim_t k = 0; k < shape[2]; ++k) {

                // Transpose XZ: swap upper triangle with lower triangle.
                for (dim_t j = 0; j < shape[1]; ++j)
                    for (dim_t l = j + 1; l < shape[3]; ++l)
                        std::swap(output[indexing::at(i, j, k, l, strides)],
                                  output[indexing::at(i, l, k, j, strides)]);
            }
        }
    }
}

namespace noa::cpu::memory::details {
    #define NOA_INSTANTIATE_TRANSPOSE_(T)                                   \
    template void permute<T>(const T*, dim4_t, dim4_t, T*, dim4_t, dim4_t); \
    template void inplace::permute0213<T>(T*, dim4_t, dim4_t);              \
    template void inplace::permute0132<T>(T*, dim4_t, dim4_t);              \
    template void inplace::permute0321<T>(T*, dim4_t, dim4_t)

    NOA_INSTANTIATE_TRANSPOSE_(bool);
    NOA_INSTANTIATE_TRANSPOSE_(int8_t);
    NOA_INSTANTIATE_TRANSPOSE_(int16_t);
    NOA_INSTANTIATE_TRANSPOSE_(int32_t);
    NOA_INSTANTIATE_TRANSPOSE_(int64_t);
    NOA_INSTANTIATE_TRANSPOSE_(uint8_t);
    NOA_INSTANTIATE_TRANSPOSE_(uint16_t);
    NOA_INSTANTIATE_TRANSPOSE_(uint32_t);
    NOA_INSTANTIATE_TRANSPOSE_(uint64_t);
    NOA_INSTANTIATE_TRANSPOSE_(half_t);
    NOA_INSTANTIATE_TRANSPOSE_(float);
    NOA_INSTANTIATE_TRANSPOSE_(double);
    NOA_INSTANTIATE_TRANSPOSE_(chalf_t);
    NOA_INSTANTIATE_TRANSPOSE_(cfloat_t);
    NOA_INSTANTIATE_TRANSPOSE_(cdouble_t);
}
