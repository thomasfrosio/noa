#include "noa/common/Assert.h"
#include "noa/cpu/memory/Transpose.h"

// TODO Use BLAS copy for out-of-place transpose
// TODO Try https://stackoverflow.com/a/16743203

namespace noa::cpu::memory::details {
    template<typename T>
    void transpose(const T* input, size4_t input_stride, size4_t input_shape,
                   T* output, size4_t output_stride, uint4_t permutation) {
        NOA_ASSERT(input != output);
        size4_t offset;
        const size4_t output_shape = indexing::reorder(input_shape, permutation);
        const size4_t input_stride_permuted = indexing::reorder(input_stride, permutation);
        for (size_t i = 0; i < output_shape[0]; ++i) {
            offset[0] = i * input_stride_permuted[0];
            for (size_t j = 0; j < output_shape[1]; ++j) {
                offset[1] = j * input_stride_permuted[1];
                for (size_t k = 0; k < output_shape[2]; ++k) {
                    offset[2] = k * input_stride_permuted[2];
                    for (size_t l = 0; l < output_shape[3]; ++l) {
                        offset[3] = l * input_stride_permuted[3];
                        output[indexing::at(i, j, k, l, output_stride)] = input[math::sum(offset)];
                    }
                }
            }
        }
    }
}

namespace noa::cpu::memory::details::inplace {
    template<typename T>
    void transpose0213(T* output, size4_t stride, size4_t shape) {
        if (shape[2] != shape[1])
            NOA_THROW("For a \"0213\" in-place permutation, shape[2] should be equal to shape[1]. Got {}", shape);

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t l = 0; l < shape[3]; ++l) {

                // Transpose YZ: swap bottom triangle with upper triangle.
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = j + 1; k < shape[2]; ++k)
                        std::swap(output[indexing::at(i, j, k, l, stride)],
                                  output[indexing::at(i, k, j, l, stride)]);
            }
        }
    }

    template<typename T>
    void transpose0132(T* output, size4_t stride, size4_t shape) {
        if (shape[3] != shape[2])
            NOA_THROW("For a \"0132\" in-place transposition, shape[3] should be equal to shape[2]. Got {}", shape);

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {

                // Transpose XY: swap upper triangle with lower triangle.
                for (size_t k = 0; k < shape[2]; ++k)
                    for (size_t l = k + 1; l < shape[3]; ++l)
                        std::swap(output[indexing::at(i, j, k, l, stride)],
                                  output[indexing::at(i, j, l, k, stride)]);
            }
        }
    }

    template<typename T>
    void transpose0321(T* output, size4_t stride, size4_t shape) {
        if (shape[3] != shape[1])
            NOA_THROW("For a \"0321\" in-place permutation, shape[3] should be equal to shape[1]. Got {}", shape);

        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t k = 0; k < shape[2]; ++k) {

                // Transpose XZ: swap upper triangle with lower triangle.
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t l = j + 1; l < shape[3]; ++l)
                        std::swap(output[indexing::at(i, j, k, l, stride)],
                                  output[indexing::at(i, l, k, j, stride)]);
            }
        }
    }
}

namespace noa::cpu::memory::details {
    #define NOA_INSTANTIATE_TRANSPOSE_(T)                                           \
    template void transpose<T>(const T*, size4_t, size4_t, T*, size4_t, uint4_t);   \
    template void inplace::transpose0213<T>(T*, size4_t, size4_t);                  \
    template void inplace::transpose0132<T>(T*, size4_t, size4_t);                  \
    template void inplace::transpose0321<T>(T*, size4_t, size4_t)

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
