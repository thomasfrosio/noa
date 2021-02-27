#pragma once

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/Types.h"

/*
 * Centering
 * =========
 *
 * The not-centered format is the layout used by FFT routines, with the origin (the DC) at the first index.
 * The centered format is the layout often used in files, with the origin in the middle left (N/2).
 *
 * Redundancy
 * ==========
 *
 * Refers to non-redundant Fourier transforms of real inputs, resulting into transforms with a logical shape of
 * {fast,medium,slow} real elements having a physical shape of {fast/2+1,medium,slow} complex elements.
 */

namespace Noa::Fourier {
    /**
     * Crops an array in Fourier space.
     * @param[in] in        Input array. Should be not-centered, not-redundant and contiguous.
     * @param[out] out      Output array. Will be not-centered, not-redundant and contiguous.
     * @param shape_in      Logical {fast, medium, slow} shape of @a in.
     * @param shape_out     Logical {fast, medium, slow} shape of @a out.
     *                      All dimensions should be less or equal than the dimensions of @a shape_in.
     */
    void crop(const cfloat_t* in, cfloat_t* out, size3_t shape_in, size3_t shape_out);

    // pad
    void pad(const cfloat_t* in, cfloat_t* out, size3_t shape_in, size3_t shape_out, cfloat_t value);

    /**
     * Crops an array in Fourier space.
     * @param[in] in        Input array. Should be not-centered, redundant and contiguous.
     * @param[out] out      Output array. Will be not-centered, redundant and contiguous.
     * @param shape_in      Logical {fast, medium, slow} shape of @a in.
     * @param shape_out     Logical {fast, medium, slow} shape of @a out.
     *                      All dimensions should be less or equal than the dimensions of @a shape_in.
     */
    void cropFull(const float* in, float* out, size3_t shape_in, size3_t shape_out);

    // padFull
    void padFull(const float* in, float* out, size3_t shape_in, size3_t shape_out);
}
