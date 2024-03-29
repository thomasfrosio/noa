#pragma once

#include "noa/core/Types.hpp"
#include "noa/cpu/utils/EwiseUnary.hpp"
#include "noa/cpu/utils/EwiseBinary.hpp"
#include "noa/cpu/utils/Iwise.hpp"

namespace noa::cpu::math {
    // Extracts the real and imaginary part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_real_v<T>>>
    void decompose(const Complex<T>* input, Strides4<i64> input_strides,
                   T* real, Strides4<i64> real_strides,
                   T* imag, Strides4<i64> imag_strides,
                   Shape4<i64> shape, i64 threads) {
        NOA_ASSERT(real != imag && all(shape > 0));

        if (noa::all(input_strides > 0)) {
            const auto order = noa::indexing::order(input_strides, shape);
            input_strides = noa::indexing::reorder(input_strides, order);
            real_strides = noa::indexing::reorder(real_strides, order);
            imag_strides = noa::indexing::reorder(imag_strides, order);
            shape = noa::indexing::reorder(shape, order);
        }

        // TODO Having one input and two outputs is unusual.
        //      Maybe add a generic ewise_unary with two outputs.
        const auto input_accessor = AccessorRestrict<const Complex<T>, 4, i64>(input, input_strides);
        const auto real_accessor = AccessorRestrict<T, 4, i64>(real, real_strides);
        const auto imag_accessor = AccessorRestrict<T, 4, i64>(imag, imag_strides);
        auto decompose_op = [=](i64 i, i64 j, i64 k, i64 l) {
            const Complex<T> value = input_accessor(i, j, k, l);
            real_accessor(i, j, k, l) = value.real;
            imag_accessor(i, j, k, l) = value.imag;
        };
        noa::cpu::utils::iwise_4d(shape, decompose_op, threads);
    }

    // Extracts the real part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_real_v<T>>>
    void real(const Complex<T>* input, const Strides4<i64>& input_strides,
              T* real, const Strides4<i64>& real_strides,
              const Shape4<i64>& shape, i64 threads) {
        noa::cpu::utils::ewise_unary(input, input_strides, real, real_strides, shape, noa::real_t{}, threads);
    }

    // Extracts the imaginary part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_real_v<T>>>
    void imag(const Complex<T>* input, const Strides4<i64>& input_strides,
              T* imag, const Strides4<i64>& imag_strides,
              const Shape4<i64>& shape, i64 threads) {
        noa::cpu::utils::ewise_unary(input, input_strides, imag, imag_strides, shape, noa::imag_t{}, threads);
    }

    // Fuses the real and imaginary components.
    template<typename T, typename = std::enable_if_t<traits::is_real_v<T>>>
    void complex(const T* real, const Strides4<i64>& real_strides,
                 const T* imag, const Strides4<i64>& imag_strides,
                 Complex<T>* output, const Strides4<i64>& output_strides,
                 const Shape4<i64>& shape, i64 threads) {
        return noa::cpu::utils::ewise_binary(real, real_strides, imag, imag_strides, output, output_strides, shape,
                                             [](const T& r, const T& i) { return noa::Complex<T>(r, i); }, threads);
    }
}
