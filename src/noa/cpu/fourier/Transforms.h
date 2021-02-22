#pragma once

#include <fftw3.h>

#include "noa/Definitions.h"
#include "noa/Exception.h"
#include "noa/Types.h"
#include "noa/cpu/fourier/Plan.h"

namespace Noa::Fourier {
    /* ---------------- */
    /* --- Execute  --- */
    /* ---------------- */

    /**
     * Computes the r2c transform (i.e forward transform) of @a input and stores the result into @a output.
     * @param output
     * @param input
     * @param normalize
     * @param[in] plan      If nullptr,
     *                      If valid plan, execute the existing plan. Note
     *
     * @note Since the plan is not modified by @c fftw_execute, it is safe to execute the same plan in parallel by
     *       multiple threads. However, since a given plan operates by default on a fixed array, one needs to use
     *       one of the new-array functions so that different threads compute the transform of different data.
     */
    void transform(const Fourier::Plan<float>& plan) {
        fftwf_execute(plan.get());
    }

    void transform(const Fourier::Plan<double>& plan) {
        fftw_execute(plan.get());
    }

    /* ---------------------------- */
    /* --- New-array transforms --- */
    /* ---------------------------- */

    /**
     * Computes the r2c transform (i.e forward transform) of @a input and stores the result into @a output.
     * @param output
     * @param input
     * @param normalize
     * @param[in] plan      If nullptr,
     *                      If valid plan, execute the existing plan. Note
     *
     * @note This function is thread-safe as long as the @a input and @a output arrays are only accessed by
     *       one single thread (i.e. the @a plan can be access by multiple threads concurrently).
     */
    void r2c(cfloat_t* output, float* input, const Fourier::Plan<float>& plan) {
        fftwf_execute_dft_r2c(plan.get(), input, reinterpret_cast<fftwf_complex*>(output));
    }

    void r2c(cdouble_t* output, double* input, const Fourier::Plan<double>& plan) {
        fftw_execute_dft_r2c(plan.get(), input, reinterpret_cast<fftw_complex*>(output));
    }

    void c2r(float* output, cfloat_t* input, const Fourier::Plan<float>& plan) {
        fftwf_execute_dft_c2r(plan.get(), reinterpret_cast<fftwf_complex*>(input), output);
    }

    void c2r(double* output, cdouble_t* input, const Fourier::Plan<double>& plan) {
        fftw_execute_dft_c2r(plan.get(), reinterpret_cast<fftw_complex*>(input), output);
    }

    /* ----------------------------- */
    /* --- "One time" transforms --- */
    /* ----------------------------- */

    void r2c(cfloat_t* output, float* input, shape_t shape) {
        Plan<float> fast_plan(output, input, shape, Flag::estimate);
        fftwf_execute(fast_plan.get());
    }

    void r2c(cdouble_t* output, double* input, shape_t shape) {
        Plan<double> fast_plan(output, input, shape, Flag::estimate);
        fftw_execute(fast_plan.get());
    }

    void c2r(float* output, cfloat_t* input, shape_t shape) {
        Plan<float> fast_plan(output, input, shape, Flag::estimate);
        fftwf_execute(fast_plan.get());
    }

    void c2r(double* output, cdouble_t* input, shape_t shape) {
        Plan<double> fast_plan(output, input, shape, Flag::estimate);
        fftw_execute(fast_plan.get());
    }
}
