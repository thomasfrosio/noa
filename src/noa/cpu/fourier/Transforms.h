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
     * Executes the @a plan.
     * @note It is safe to execute the same plan in parallel by multiple threads. However, since a given plan operates
     *       by default on a fixed array, one needs to use one of the new-array functions so that different threads
     *       compute the transform of different data.
     */
    NOA_IH void execute(const Fourier::Plan<float>& plan) { fftwf_execute(plan.get()); }
    NOA_IH void execute(const Fourier::Plan<double>& plan) { fftw_execute(plan.get()); }

    /* ---------------------------- */
    /* --- New-array transforms --- */
    /* ---------------------------- */

    /**
     * Computes the r2c transform (i.e forward transform) using the @a plan.
     * @param[in] input     Input. Should match the output used to create @a plan.
     * @param[out] output   Output. Should match the output used to create @a plan.
     * @param plan          Existing plan.
     *
     * @note This function is thread-safe as long as the @a input and @a output arrays are only accessed by
     *       one single thread (i.e. the @a plan can be access by multiple threads concurrently).
     *
     * @note The arrays used to create the @a plan should be similar to @a input and @a output.
     *       The shape should be the same. The input and output arrays are the same (in-place) or different
     *       (out-of-place) if the plan was originally created to be in-place or out-of-place, respectively.
     *       The alignment should be the same as well.
     */
    NOA_IH void r2c(float* input, cfloat_t* output, const Fourier::Plan<float>& plan) {
        fftwf_execute_dft_r2c(plan.get(), input, reinterpret_cast<fftwf_complex*>(output));
    }

    NOA_IH void r2c(double* input, cdouble_t* output, const Fourier::Plan<double>& plan) {
        fftw_execute_dft_r2c(plan.get(), input, reinterpret_cast<fftw_complex*>(output));
    }

    /**
     * Computes the c2r transform (i.e backward transform) using the @a plan.
     * @param[in] input     Input. Should match the output used to create @a plan.
     * @param[out] output   Output. Should match the output used to create @a plan.
     * @param plan          Existing plan.
     *
     * @note This function is thread-safe as long as the @a input and @a output arrays are only accessed by
     *       one single thread (i.e. the @a plan can be access by multiple threads concurrently).
     *
     * @note The arrays used to create the @a plan should be similar to @a input and @a output.
     *       The shape should be the same. The input and output arrays are the same (in-place) or different
     *       (out-of-place) if the plan was originally created to be in-place or out-of-place, respectively.
     *       The alignment should be the same as well.
     */
    NOA_IH void c2r(cfloat_t* input, float* output, const Fourier::Plan<float>& plan) {
        fftwf_execute_dft_c2r(plan.get(), reinterpret_cast<fftwf_complex*>(input), output);
    }

    NOA_IH void c2r(cdouble_t* input, double* output, const Fourier::Plan<double>& plan) {
        fftw_execute_dft_c2r(plan.get(), reinterpret_cast<fftw_complex*>(input), output);
    }

    /**
     * Computes the c2c transform using the @a plan.
     * @param[in] input     Input. Should match the output used to create @a plan.
     * @param[out] output   Output. Should match the output used to create @a plan.
     * @param plan          Existing plan.
     *
     * @note This function is thread-safe as long as the @a input and @a output arrays are only accessed by
     *       one single thread (i.e. the @a plan can be access by multiple threads concurrently).
     *
     * @note The arrays used to create the @a plan should be similar to @a input and @a output.
     *       The shape should be the same. The input and output arrays are the same (in-place) or different
     *       (out-of-place) if the plan was originally created to be in-place or out-of-place, respectively.
     *       The alignment should be the same as well.
     */
    NOA_IH void c2c(cfloat_t* input, cfloat_t* output, const Fourier::Plan<float>& plan) {
        fftwf_execute_dft(plan.get(),
                          reinterpret_cast<fftwf_complex*>(input),
                          reinterpret_cast<fftwf_complex*>(output));
    }

    NOA_IH void c2c(cdouble_t* input, cdouble_t* output, const Fourier::Plan<double>& plan) {
        fftw_execute_dft(plan.get(),
                         reinterpret_cast<fftw_complex*>(input),
                         reinterpret_cast<fftw_complex*>(output));
    }

    /* ----------------------------- */
    /* --- "One time" transforms --- */
    /* ----------------------------- */

    /**
     * Computes the r2c transform (i.e forward transform).
     * @see Fourier::Plan<float> for more details.
     * @note @a input and @a output can be the same, which will trigger an in-place transform.
     */
    NOA_IH void r2c(float* input, cfloat_t* output, size3_t shape) {
        Plan<float> fast_plan(input, output, shape, Fourier::ESTIMATE);
        fftwf_execute(fast_plan.get());
    }

    NOA_IH void r2c(double* input, cdouble_t* output, size3_t shape) {
        Plan<double> fast_plan(input, output, shape, Fourier::ESTIMATE);
        fftw_execute(fast_plan.get());
    }

    /// Computes the in-place r2c transform.
    NOA_IH void r2c(float* data, size3_t shape) { r2c(data, reinterpret_cast<cfloat_t*>(data), shape); }
    NOA_IH void r2c(double* data, size3_t shape) { r2c(data, reinterpret_cast<cdouble_t*>(data), shape); }

    /**
     * Computes the c2r transform (i.e backward transform).
     * @see Fourier::Plan<float> for more details.
     * @note @a input and @a output can be the same, which will trigger an in-place transform.
     */
    NOA_IH void c2r(cfloat_t* input, float* output, size3_t shape) {
        Plan<float> fast_plan(input, output, shape, Fourier::ESTIMATE);
        fftwf_execute(fast_plan.get());
    }

    NOA_IH void c2r(cdouble_t* input, double* output, size3_t shape) {
        Plan<double> fast_plan(input, output, shape, Fourier::ESTIMATE);
        fftw_execute(fast_plan.get());
    }

    /// Computes the in-place c2r transform.
    NOA_IH void c2r(cfloat_t* data, size3_t shape) { c2r(data, reinterpret_cast<float*>(data), shape); }
    NOA_IH void c2r(cdouble_t* data, size3_t shape) { c2r(data, reinterpret_cast<double*>(data), shape); }

    /**
     * Computes the c2c transform (i.e forward or backward transform depending on the @a sign).
     * @see Fourier::Plan<float> for more details.
     * @note @a input and @a output can be the same, which will trigger an in-place transform.
     */
    NOA_IH void c2c(cfloat_t* input, cfloat_t* output, size3_t shape, int sign) {
        Plan<float> fast_plan(input, output, shape, sign, Fourier::ESTIMATE);
        fftwf_execute(fast_plan.get());
    }

    NOA_IH void c2c(cdouble_t* input, cdouble_t* output, size3_t shape, int sign) {
        Plan<double> fast_plan(input, output, shape, sign, Fourier::ESTIMATE);
        fftw_execute(fast_plan.get());
    }

    /// Computes the in-place c2c transform.
    NOA_IH void c2c(cfloat_t* data, int sign, size3_t shape) { c2c(data, data, shape, sign); }
    NOA_IH void c2c(cdouble_t* data, int sign, size3_t shape) { c2c(data, data, shape, sign); }
}
