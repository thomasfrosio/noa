#pragma once

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"

#include "noa/cpu/Stream.h"
#include "noa/cpu/math/Ewise.h"

namespace noa::cpu::math {
    // Extracts the real and imaginary part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    void decompose(const shared_t<Complex<T>[]>& input, size4_t input_strides,
                   const shared_t<T[]>& real, size4_t real_strides,
                   const shared_t<T[]>& imag, size4_t imag_strides,
                   size4_t shape, Stream& stream) {
        stream.enqueue([=]() mutable {
            if (all(input_strides > 0)) {
                const size4_t order = indexing::order(input_strides, shape);
                input_strides = indexing::reorder(input_strides, order);
                real_strides = indexing::reorder(real_strides, order);
                imag_strides = indexing::reorder(imag_strides, order);
                shape = indexing::reorder(shape, order);
            }

            const Complex<T> ptr = input.get();
            T* rptr = real.get();
            T* iptr = imag.get();
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    for (size_t k = 0; k < shape[2]; ++k) {
                        for (size_t l = 0; l < shape[3]; ++l) {
                            const size_t offset = indexing::at(i, j, k, l, input_strides);
                            rptr[indexing::at(i, j, k, l, real_strides)] = ptr[offset].real;
                            iptr[indexing::at(i, j, k, l, imag_strides)] = ptr[offset].imag;
                        }
                    }
                }
            }
        });
    }

    // Extracts the real part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    inline void real(const shared_t<Complex<T>[]>& input, size4_t input_strides,
                     const shared_t<T[]>& real, size4_t real_strides,
                     size4_t shape, Stream& stream) {
        cpu::math::ewise(input, input_strides, real, real_strides, shape, noa::math::real_t{}, stream);
    }

    // Extracts the imaginary part of complex numbers.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    inline void imag(const shared_t<Complex<T>[]>& input, size4_t input_strides,
                     const shared_t<T[]>& imag, size4_t imag_strides,
                     size4_t shape, Stream& stream) {
        cpu::math::ewise(input, input_strides, imag, imag_strides, shape, noa::math::imag_t{}, stream);
    }

    // Fuses the real and imaginary components.
    template<typename T, typename = std::enable_if_t<traits::is_float_v<T>>>
    inline void complex(const shared_t<T[]>& real, size4_t real_strides,
                        const shared_t<T[]>& imag, size4_t imag_strides,
                        const shared_t<Complex<T>[]>& output, size4_t output_strides,
                        size4_t shape, Stream& stream) {
        return math::ewise(real, real_strides, imag, imag_strides, output, output_strides, shape,
                           [](const T& r, const T& i) { return noa::Complex<T>(r, i); }, stream);
    }
}
