#pragma once

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/Types.h"
#include "noa/util/Math.h"

/*
 * f = redundant, non-centered
 * fc = redundant, centered
 * h = non-redundant, non-centered
 * hc = non-redundant, centered
 *
 * Supported reorders
 * ==================
 *
 * fc2f <-> f2fc    (ifftshift <-> fftshift)
 * hc2h <-> h2hc    (file to fft <-> fft to file)
 */

namespace Noa::Fourier {
    /**
     * Remaps "half centered to half", i.e. file format to FFT format.
     * @tparam T        Real or complex.
     * @param[in] in    The first `getElementsFFT(shape) * sizeof(T)` bytes will be read.
     * @param[out] out  The first `getElementsFFT(shape) * sizeof(T)` bytes will be written.
     * @param shape     Logical {fast, medium, slow} shape of @a in and @a out.
     */
    template<typename T>
    NOA_HOST void hc2h(const T* in, T* out, size3_t shape) {
        size_t half_x = (shape.x / 2 + 1);
        size_t base_z, base_y;
        for (size_t z = 0; z < shape.z; ++z) {
            base_z = Math::iFFTShift(z, shape.z);
            for (size_t y = 0; y < shape.y; ++y) {
                base_y = Math::iFFTShift(y, shape.y);
                std::memcpy(out + (base_z * shape.y + base_y) * half_x,
                            in + (z * shape.y + y) * half_x,
                            half_x * sizeof(T));
            }
        }
    }

    /**
     * Remaps "half to half centered", i.e. FFT format to file format.
     * @tparam T        Real or complex.
     * @param[in] in    The first `getElementsFFT(shape) * sizeof(T)` bytes will be read.
     * @param[out] out  The first `getElementsFFT(shape) * sizeof(T)` bytes will be written.
     * @param shape     Logical {fast, medium, slow} shape of @a in and @a out.
     */
    template<typename T>
    NOA_IH void h2hc(const T* in, T* out, size3_t shape) {
        size_t half_x = (shape.x / 2 + 1);
        size_t base_z, base_y;
        for (size_t z = 0; z < shape.z; ++z) {
            base_z = Math::FFTShift(z, shape.z);
            for (size_t y = 0; y < shape.y; ++y) {
                base_y = Math::FFTShift(y, shape.y);
                std::memcpy(out + (base_z * shape.y + base_y) * half_x,
                            in + (z * shape.y + y) * half_x,
                            half_x * sizeof(T));
            }
        }
    }

    /**
     * Remaps "full centered to full", i.e. iFFTShift.
     * @tparam T        Real or complex.
     * @param shape     Logical {fast, medium, slow} shape of @a in and @a out.
     * @TODO: Replace the x loop with 2 memcpy to remove iFFTShift.
     */
    template<typename T>
    NOA_HOST void fc2f(const T* in, T* out, size3_t shape) {
        size3_t base;
        for (size_t z = 0; z < shape.z; ++z) {
            base.z = Math::iFFTShift(z, shape.z);
            for (size_t y = 0; y < shape.y; ++y) {
                base.y = Math::iFFTShift(y, shape.y);
                for (size_t x = 0; x < shape.x; ++x) {
                    base.x = Math::iFFTShift(x, shape.x);
                    out[(base.z * shape.y + base.y) * shape.x + base.x] = in[(z * shape.y + y) * shape.x + x];
                }
            }
        }
    }

    /**
     * Remaps "full to full centered", i.e. FFTShift.
     * @tparam T        Real or complex.
     * @param shape     Logical {fast, medium, slow} shape of @a in and @a out.
     * @TODO: Replace the x loop with 2 memcpy to get remove iFFTShift.
     */
    template<typename T>
    NOA_HOST void f2fc(const T* in, T* out, size3_t shape) {
        size3_t base;
        for (size_t z = 0; z < shape.z; ++z) {
            base.z = Math::FFTShift(z, shape.z);
            for (size_t y = 0; y < shape.y; ++y) {
                base.y = Math::FFTShift(y, shape.y);
                for (size_t x = 0; x < shape.x; ++x) {
                    base.x = Math::FFTShift(x, shape.x);
                    out[(base.z * shape.y + base.y) * shape.x + base.x] = in[(z * shape.y + y) * shape.x + x];
                }
            }
        }
    }

    /**
     * Remaps "full centered to half".
     * @param[in] in    Full transform. The first `getElements(shape) * sizeof(T)` bytes will be read.
     * @param[out] out  Half transform. The first `getElementsFFT(shape) * sizeof(T)` bytes will be written.
     * @param shape     Logical {fast, medium, slow} shape of @a in and @a out.
     * @todo Replace the last x loop with one memcpy + if even, copy single Nyquist at the end of the row.
     * @todo This function is not tested. Add tests.
     */
    template<typename T>
    NOA_HOST void fc2h(const T* in, T* out, size3_t shape) {
        size_t half = shape.x / 2 + 1;
        size_t base_z, base_y;
        for (size_t z = 0; z < shape.z; ++z) {
            base_z = Math::iFFTShift(z, shape.z);
            for (size_t y = 0; y < shape.y; ++y) {
                base_y = Math::iFFTShift(y, shape.y);
                for (size_t x = 0; x < half; ++x) {
                    out[(base_z * shape.y + base_y) * half + x] =
                            in[(z * shape.y + y) * shape.x + Math::FFTShift(x, shape.x)];
                }
            }
        }
    }

    /**
     * Remaps "full to half".
     * @tparam T        Real or complex.
     * @param[in] in    Full transform. The first `getElements(shape) * sizeof(T)` bytes will be read.
     * @param[out] out  Half transform. The first `getElementsFFT(shape) * sizeof(T)` bytes will be written.
     * @param shape     Logical {fast, medium, slow} shape of @a in and @a out.
     * @todo This function is not tested. Add tests.
     */
    template<typename T>
    NOA_HOST void f2h(const T* in, T* out, size3_t shape) {
        size_t half_x = (shape.x / 2 + 1);
        for (size_t z = 0; z < shape.z; ++z) {
            for (size_t y = 0; y < shape.y; ++y) {
                for (size_t x = 0; x < half_x; ++x)
                    out[(z * shape.y + y) * half_x + x] = in[(z * shape.y + y) * shape.x + x];
            }
        }
    }
}
