#pragma once

#ifndef NOA_EWISE_INL_
#error "This is an internal header; it should not be included."
#endif

// TODO OpenMP support? I don't want to add #pragma omp parallel for everywhere since it prevents constant-evaluated
//      expression to be move to outer-loops. Benchmarking will be necessary.

namespace noa::cpu::math {
    template<typename T, typename U, typename UnaryOp>
    void ewise(const T* input, size4_t input_stride,
               U* output, size4_t output_stride,
               size4_t shape, UnaryOp unary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            output[at(i, j, k, l, output_stride)] =
                                    unary_op(input[at(i, j, k, l, input_stride)]);
        });
    }

    template<typename T, typename U, typename V, typename BinaryOp>
    void ewise(const T* inputs, size4_t input_stride, U values,
               V* outputs, size4_t output_stride,
               size4_t shape, BinaryOp binary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
            if constexpr (std::is_pointer_v<U>) {
                for (size_t i = 0; i < shape[0]; ++i)
                    for (size_t j = 0; j < shape[1]; ++j)
                        for (size_t k = 0; k < shape[2]; ++k)
                            for (size_t l = 0; l < shape[3]; ++l)
                                outputs[at(i, j, k, l, output_stride)] =
                                        binary_op(inputs[at(i, j, k, l, input_stride)], values[i]);
            } else {
                for (size_t i = 0; i < shape[0]; ++i)
                    for (size_t j = 0; j < shape[1]; ++j)
                        for (size_t k = 0; k < shape[2]; ++k)
                            for (size_t l = 0; l < shape[3]; ++l)
                                outputs[at(i, j, k, l, output_stride)] =
                                        binary_op(inputs[at(i, j, k, l, input_stride)], values);
            }
        });
    }

    template<typename T, typename U, typename V, typename BinaryOp>
    void ewise(const T* input, size4_t input_stride,
               const U* array, size4_t array_stride,
               V* output, size4_t output_stride,
               size4_t shape, BinaryOp binary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            output[at(i, j, k, l, output_stride)] =
                                    binary_op(input[at(i, j, k, l, input_stride)],
                                              array[at(i, j, k, l, array_stride)]);
        });
    }

    template<typename T, typename U, typename V, typename TrinaryOp>
    void ewise(const T* input, size4_t input_stride, U value1, U value2,
               V* output, size4_t output_stride,
               size4_t shape, TrinaryOp trinary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
            if constexpr (std::is_pointer_v<U>) {
                for (size_t i = 0; i < shape[0]; ++i)
                    for (size_t j = 0; j < shape[1]; ++j)
                        for (size_t k = 0; k < shape[2]; ++k)
                            for (size_t l = 0; l < shape[3]; ++l)
                                output[at(i, j, k, l, output_stride)] =
                                        trinary_op(input[at(i, j, k, l, input_stride)], value1[i], value2[i]);
            } else {
                for (size_t i = 0; i < shape[0]; ++i)
                    for (size_t j = 0; j < shape[1]; ++j)
                        for (size_t k = 0; k < shape[2]; ++k)
                            for (size_t l = 0; l < shape[3]; ++l)
                                output[at(i, j, k, l, output_stride)] =
                                        trinary_op(input[at(i, j, k, l, input_stride)], value1, value2);
            }
        });
    }

    template<typename T, typename U, typename V, typename W, typename TrinaryOp>
    void ewise(const T* input, size4_t input_stride,
               const U* array1, size4_t array1_stride,
               const V* array2, size4_t array2_stride,
               W* output, size4_t output_stride,
               size4_t shape, TrinaryOp trinary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            output[at(i, j, k, l, output_stride)] =
                                    trinary_op(input[at(i, j, k, l, input_stride)],
                                               array1[at(i, j, k, l, array1_stride)],
                                               array2[at(i, j, k, l, array2_stride)]);
        });
    }
}
