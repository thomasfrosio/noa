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
                                    static_cast<U>(unary_op(input[at(i, j, k, l, input_stride)]));
        });
    }

    template<typename T, typename U, typename V, typename BinaryOp, typename>
    void ewise(const T* lhs, size4_t lhs_stride, U rhs,
               V* output, size4_t output_stride,
               size4_t shape, BinaryOp binary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            output[at(i, j, k, l, output_stride)] =
                                    static_cast<V>(binary_op(lhs[at(i, j, k, l, lhs_stride)], rhs));
        });
    }

    template<typename T, typename U, typename V, typename BinaryOp>
    void ewise(const T* lhs, size4_t lhs_stride, const U* rhs,
               V* output, size4_t output_stride,
               size4_t shape, BinaryOp binary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            output[at(i, j, k, l, output_stride)] =
                                    static_cast<V>(binary_op(lhs[at(i, j, k, l, lhs_stride)], rhs[i]));
        });
    }

    template<typename T, typename U, typename V, typename BinaryOp>
    void ewise(const T* lhs, size4_t lhs_stride,
               const U* rhs, size4_t rhs_stride,
               V* output, size4_t output_stride,
               size4_t shape, BinaryOp binary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            output[at(i, j, k, l, output_stride)] =
                                    static_cast<V>(binary_op(lhs[at(i, j, k, l, lhs_stride)],
                                                             rhs[at(i, j, k, l, rhs_stride)]));
        });
    }

    template<typename T, typename U, typename V, typename TrinaryOp, typename>
    void ewise(const T* lhs, size4_t lhs_stride, U mhs, U rhs,
               V* output, size4_t output_stride,
               size4_t shape, TrinaryOp trinary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
                for (size_t i = 0; i < shape[0]; ++i)
                    for (size_t j = 0; j < shape[1]; ++j)
                        for (size_t k = 0; k < shape[2]; ++k)
                            for (size_t l = 0; l < shape[3]; ++l)
                                output[at(i, j, k, l, output_stride)] =
                                        static_cast<V>(trinary_op(lhs[at(i, j, k, l, lhs_stride)],
                                                                  mhs, rhs));
        });
    }

    template<typename T, typename U, typename V, typename TrinaryOp>
    void ewise(const T* lhs, size4_t lhs_stride, const U* mhs, const U* rhs,
               V* output, size4_t output_stride,
               size4_t shape, TrinaryOp trinary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            output[at(i, j, k, l, output_stride)] =
                                    static_cast<V>(trinary_op(lhs[at(i, j, k, l, lhs_stride)],
                                                              mhs[i], rhs[i]));
        });
    }

    template<typename T, typename U, typename V, typename W, typename TrinaryOp>
    void ewise(const T* lhs, size4_t lhs_stride,
               const U* mhs, size4_t mhs_stride,
               const V* rhs, size4_t rhs_stride,
               W* output, size4_t output_stride,
               size4_t shape, TrinaryOp trinary_op, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        stream.enqueue([=]() {
            for (size_t i = 0; i < shape[0]; ++i)
                for (size_t j = 0; j < shape[1]; ++j)
                    for (size_t k = 0; k < shape[2]; ++k)
                        for (size_t l = 0; l < shape[3]; ++l)
                            output[at(i, j, k, l, output_stride)] =
                                    static_cast<W>(trinary_op(lhs[at(i, j, k, l, lhs_stride)],
                                                              mhs[at(i, j, k, l, mhs_stride)],
                                                              rhs[at(i, j, k, l, rhs_stride)]));
        });
    }
}
