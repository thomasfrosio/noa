#pragma once

#include "noa/Definitions.h"
#include "noa/Types.h"

/*
 * Centering
 * =========
 *
 * The not-centered format is the layout used by FFT routines, with the origin (the DC) at index 0.
 * The centered format is the layout often used in files, with the origin in the middle left (N/2).
 *
 * Redundancy
 * ==========
 *
 * Refers to non-redundant Fourier transforms of real inputs, resulting into transforms with a LOGICAL shape of
 * {fast,medium,slow} real elements having a PHYSICAL shape of {fast/2+1,medium,slow} complex elements.
 * Note that with even dimensions, the Nyquist frequency is real and the C2R routines will assume the imaginary
 * part is zero.
 */

namespace Noa::Fourier {
    /**
     * Crops a Fourier transform.
     * @tparam T            float, double, cfloat_t, cdouble_t.
     * @param[in] in        Input array. Should be not-centered, not-redundant and contiguous.
     * @param shape_in      Logical {fast, medium, slow} shape of @a in, in complex elements.
     * @param[out] out      Output array. Will be not-centered, not-redundant and contiguous.
     * @param shape_out     Logical {fast, medium, slow} shape of @a out.
     *                      All dimensions should be less or equal than the dimensions of @a shape_in.
     * @note If @a shape_in and @a shape_out are equal, @a in is copied into @a out.
     * @note The physical size for the fast dimension is expected to be x / 2 + 1 elements.
     * @warning @a in and @a out should not overlap.
     */
    template<typename T>
    NOA_HOST void crop(const T* in, size3_t shape_in, T* out, size3_t shape_out);

    /**
     * Crops a Fourier transform.
     * @tparam T            float, double, cfloat_t, cdouble_t.
     * @param[in] in        Input array. Should be not-centered, redundant and contiguous.
     * @param shape_in      Logical and physical {fast, medium, slow} shape of @a in.
     * @param[out] out      Output array. Will be not-centered, redundant and contiguous.
     * @param shape_out     Logical and physical {fast, medium, slow} shape of @a out.
     *                      All dimensions should be less or equal than the dimensions of @a shape_in.
     * @note If @a shape_in and @a shape_out are equal, @a in is copied into @a out.
     * @warning @a in and @a out should not overlap.
     */
    template<typename T>
    NOA_HOST void cropFull(const T* in, size3_t shape_in, T* out, size3_t shape_out);

    /**
     * Pads a Fourier transform with zeros.
     * @tparam T            float, double, cfloat_t, cdouble_t.
     * @param[in] in        Input array. Should be not-centered, not-redundant and contiguous.
     * @param shape_in      Logical {fast, medium, slow} shape of @a in.
     * @param[out] out      Output array. Will be not-centered, not-redundant and contiguous.
     * @param shape_out     Logical {fast, medium, slow} shape of @a out.
     *                      All dimensions should be greater or equal than the dimensions of @a shape_in.
     * @note If @a shape_in and @a shape_out are equal, @a in is copied into @a out.
     * @note The physical size for the fast dimension is expected to be x / 2 + 1 elements.
     * @warning @a in and @a out should not overlap.
     */
    template<typename T>
    NOA_HOST void pad(const T* in, size3_t shape_in, T* out, size3_t shape_out);

    /**
     * Pads a Fourier transform.
     * @tparam T            float, double, cfloat_t, cdouble_t.
     * @param[in] in        Input array. Should be not-centered, redundant and contiguous.
     * @param shape_in      Logical and physical {fast, medium, slow} shape of @a in.
     * @param[out] out      Output array. Will be not-centered, redundant and contiguous.
     * @param shape_out     Logical and physical {fast, medium, slow} shape of @a out.
     *                      All dimensions should be greater or equal than the dimensions of @a shape_in.
     * @note If @a shape_in and @a shape_out are equal, @a in is copied into @a out.
     * @warning @a in and @a out should not overlap.
     */
    template<typename T>
    NOA_HOST void padFull(const T* in, size3_t shape_in, T* out, size3_t shape_out);
}
