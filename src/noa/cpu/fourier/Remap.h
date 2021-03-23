#pragma once

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/Types.h"
#include "noa/Math.h"
#include "noa/util/Profiler.h"

/*
 * f = redundant, non-centered
 * fc = redundant, centered
 * h = non-redundant, non-centered
 * hc = non-redundant, centered
 *
 * Supported reordering
 * ====================
 *
 * F2FC <-> FC2F    (fftshift <-> ifftshift)
 * HC2H <-> H2HC    (file to fft <-> fft to file)
 * H2F  <-> F2H
 * FC2H
 */

namespace Noa::Fourier {
    /**
     * Remaps "half centered to half", i.e. file format to FFT format.
     * @tparam T        Real or complex.
     * @param[in] in    The first `getElementsFFT(shape) * sizeof(T)` bytes will be read.
     * @param[out] out  The first `getElementsFFT(shape) * sizeof(T)` bytes will be written.
     * @param shape     Logical {fast, medium, slow} shape of @a in and @a out.
     * @warning @a in and @a out should not overlap.
     */
    template<typename T>
    NOA_HOST void HC2H(const T* in, T* out, size3_t shape) {
        NOA_PROFILE_FUNCTION("fft");
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
     * @warning @a in and @a out should not overlap.
     */
    template<typename T>
    NOA_HOST void H2HC(const T* in, T* out, size3_t shape) {
        NOA_PROFILE_FUNCTION("fft");
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
     * @warning @a in and @a out should not overlap.
     *
     * @TODO Replace the x loop with 2 memcpy to remove iFFTShift.
     */
    template<typename T>
    NOA_HOST void FC2F(const T* in, T* out, size3_t shape) {
        NOA_PROFILE_FUNCTION("fft");
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
     * @warning @a in and @a out should not overlap.
     *
     * @TODO Replace the x loop with 2 memcpy to get remove iFFTShift.
     */
    template<typename T>
    NOA_HOST void F2FC(const T* in, T* out, size3_t shape) {
        NOA_PROFILE_FUNCTION("fft");
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
     * Remaps "half to full", i.e. applies the hermitian symmetry to generate the non-redundant transform.
     * @tparam T        Real or complex.
     * @param[in] in    Half transform. The first `getElementsFFT(shape) * sizeof(T)` bytes will be read.
     * @param[out] out  Full transform. The first `getElements(shape) * sizeof(T)` bytes will be written.
     * @param shape     Logical {fast, medium, slow} shape of @a in and @a out.
     *
     * @warning @a in and @a out should not overlap.
     * @note If @a T is complex, the complex conjugate is computed to generate the redundant elements.
     */
    template<typename T>
    NOA_HOST void H2F(const T* in, T* out, size3_t shape) {
        NOA_PROFILE_FUNCTION("fft");
        size_t half_x = shape.x / 2 + 1;

        // Copy first non-redundant half.
        for (size_t z = 0; z < shape.z; ++z) {
            for (size_t y = 0; y < shape.y; ++y) {
                std::memcpy(out + (z * shape.y + y) * shape.x,
                            in + (z * shape.y + y) * half_x,
                            half_x * sizeof(T));
            }
        }

        // Compute the redundant elements.
        for (size_t z = 0; z < shape.z; ++z) {
            size_t in_z = z ? shape.z - z : 0;
            for (size_t y = 0; y < shape.y; ++y) {
                size_t in_y = y ? shape.y - y : 0;
                for (size_t x = half_x; x < shape.x; ++x) {
                    T value = in[(in_z * shape.y + in_y) * half_x + shape.x - x];
                    if constexpr (Traits::is_complex_v<T>)
                        out[(z * shape.y + y) * shape.x + x] = Math::conj(value);
                    else
                        out[(z * shape.y + y) * shape.x + x] = value;
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
     * @warning @a in and @a out should not overlap.
     */
    template<typename T>
    NOA_HOST void F2H(const T* in, T* out, size3_t shape) {
        NOA_PROFILE_FUNCTION("fft");
        size_t half_x = shape.x / 2 + 1;
        for (size_t z = 0; z < shape.z; ++z) {
            for (size_t y = 0; y < shape.y; ++y) {
                std::memcpy(out + (z * shape.y + y) * half_x,
                            in + (z * shape.y + y) * shape.x,
                            half_x * sizeof(T));
            }
        }
    }

    /**
     * Remaps "full centered to half".
     * @param[in] in    Full transform. The first `getElements(shape) * sizeof(T)` bytes will be read.
     * @param[out] out  Half transform. The first `getElementsFFT(shape) * sizeof(T)` bytes will be written.
     * @param shape     Logical {fast, medium, slow} shape of @a in and @a out.
     * @warning @a in and @a out should not overlap.
     *
     * @TODO H2FC is missing, since it seems a bit more complicated and it would be surprising if we ever use it.
     *       Nevertheless, this function is tested and could be used, for instance to import full transforms
     *       from Python or MATLAB.
     */
    template<typename T>
    NOA_HOST void FC2H(const T* in, T* out, size3_t shape) {
        NOA_PROFILE_FUNCTION("fft");
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
}
