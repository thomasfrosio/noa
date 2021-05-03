#pragma once

#include "noa/Definitions.h"
#include "noa/Types.h"

/*
 * F = redundant, non-centered
 * FC = redundant, centered
 * H = non-redundant, non-centered
 * HC = non-redundant, centered
 *
 * TODO     H2FC is missing, since it seems a bit more complicated and it would be surprising if we ever use it.
 *          Nevertheless, the same can be achieved with H2F and then F2FC.
 */

namespace Noa::Fourier {
    /**
     * Remaps "half centered to half", i.e. file format to FFT format.
     * @tparam T        Real or complex.
     * @param[in] in    The first `getElementsFFT(shape) * sizeof(T)` bytes will be read.
     * @param[out] out  The first `getElementsFFT(shape) * sizeof(T)` bytes will be written.
     * @param shape     Logical {fast, medium, slow} shape of @a in and @a out.
     * @warning         @a in and @a out should not overlap.
     */
    template<typename T>
    NOA_HOST void HC2H(const T* in, T* out, size3_t shape);

    /**
     * Remaps "half to half centered", i.e. FFT format to file format.
     * @tparam T        Real or complex.
     * @param[in] in    The first `getElementsFFT(shape) * sizeof(T)` bytes will be read.
     * @param[out] out  The first `getElementsFFT(shape) * sizeof(T)` bytes will be written.
     * @param shape     Logical {fast, medium, slow} shape of @a in and @a out.
     * @warning         @a in and @a out should not overlap.
     */
    template<typename T>
    NOA_HOST void H2HC(const T* in, T* out, size3_t shape);

    /**
     * Remaps "full centered to full", i.e. iFFTShift.
     * @tparam T        Real or complex.
     * @param shape     Logical {fast, medium, slow} shape of @a in and @a out.
     * @warning         @a in and @a out should not overlap.
     */
    template<typename T>
    NOA_HOST void FC2F(const T* in, T* out, size3_t shape);

    /**
     * Remaps "full to full centered", i.e. FFTShift.
     * @tparam T        Real or complex.
     * @param shape     Logical {fast, medium, slow} shape of @a in and @a out.
     * @warning         @a in and @a out should not overlap.
     */
    template<typename T>
    NOA_HOST void F2FC(const T* in, T* out, size3_t shape);

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
    NOA_HOST void H2F(const T* in, T* out, size3_t shape);

    /**
     * Remaps "full to half".
     * @tparam T        Real or complex.
     * @param[in] in    Full transform. The first `getElements(shape) * sizeof(T)` bytes will be read.
     * @param[out] out  Half transform. The first `getElementsFFT(shape) * sizeof(T)` bytes will be written.
     * @param shape     Logical {fast, medium, slow} shape of @a in and @a out.
     * @warning         @a in and @a out should not overlap.
     */
    template<typename T>
    NOA_HOST void F2H(const T* in, T* out, size3_t shape);

    /**
     * Remaps "full centered to half".
     * @param[in] in    Full transform. The first `getElements(shape) * sizeof(T)` bytes will be read.
     * @param[out] out  Half transform. The first `getElementsFFT(shape) * sizeof(T)` bytes will be written.
     * @param shape     Logical {fast, medium, slow} shape of @a in and @a out.
     * @warning         @a in and @a out should not overlap.
     */
    template<typename T>
    NOA_HOST void FC2H(const T* in, T* out, size3_t shape);
}
