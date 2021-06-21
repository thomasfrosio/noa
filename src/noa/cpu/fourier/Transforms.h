/// \file noa/cpu/fourier/Transforms.h
/// \brief Fast Fourier Transforms.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <fftw3.h>

#include "noa/Definitions.h"
#include "noa/Types.h"
#include "noa/Profiler.h"
#include "noa/cpu/fourier/Plan.h"

namespace noa::fourier {
    // -- Execute -- //

    /// Executes the \a plan.
    /// \note It is safe to execute the same plan in parallel by multiple threads. However, since a given plan operates
    ///       by default on a fixed array, one needs to use one of the new-array functions so that different threads
    ///       compute the transform of different data.
    NOA_IH void execute(const fourier::Plan<float>& plan) {
        NOA_PROFILE_FUNCTION();
        fftwf_execute(plan.get());
    }
    NOA_IH void execute(const fourier::Plan<double>& plan) {
        NOA_PROFILE_FUNCTION();
        fftw_execute(plan.get());
    }

    // -- New-array transforms -- //

    /// Computes the R2C transform (i.e forward transform) using the \a plan.
    /// \param[in] input     Input. Should match the output used to create \a plan.
    /// \param[out] output   Output. Should match the output used to create \a plan.
    /// \param plan          Existing plan.
    ///
    /// \note This function is thread-safe as long as the \a input and \a output arrays are only accessed by
    ///       one single thread (i.e. the \a plan can be access by multiple threads concurrently).
    /// \note The arrays used to create the \a plan should be similar to \a input and \a output.
    ///       The shape should be the same. The input and output arrays are the same (in-place) or different
    ///       (out-of-place) if the plan was originally created to be in-place or out-of-place, respectively.
    ///       The alignment should be the same as well.
    NOA_IH void r2c(float* input, cfloat_t* output, const fourier::Plan<float>& plan) {
        NOA_PROFILE_FUNCTION();
        fftwf_execute_dft_r2c(plan.get(), input, reinterpret_cast<fftwf_complex*>(output));
    }

    NOA_IH void r2c(double* input, cdouble_t* output, const fourier::Plan<double>& plan) {
        NOA_PROFILE_FUNCTION();
        fftw_execute_dft_r2c(plan.get(), input, reinterpret_cast<fftw_complex*>(output));
    }

    /// Computes the C2R transform (i.e backward transform) using the \a plan.
    /// \param[in] input     Input. Should match the output used to create \a plan.
    /// \param[out] output   Output. Should match the output used to create \a plan.
    /// \param plan          Existing plan.
    ///
    /// \note This function is thread-safe as long as the \a input and \a output arrays are only accessed by
    ///       one single thread (i.e. the \a plan can be access by multiple threads concurrently).
    /// \note The arrays used to create the \a plan should be similar to \a input and \a output.
    ///       The shape should be the same. The input and output arrays are the same (in-place) or different
    ///       (out-of-place) if the plan was originally created to be in-place or out-of-place, respectively.
    ///       The alignment should be the same as well.
    NOA_IH void c2r(cfloat_t* input, float* output, const fourier::Plan<float>& plan) {
        NOA_PROFILE_FUNCTION();
        fftwf_execute_dft_c2r(plan.get(), reinterpret_cast<fftwf_complex*>(input), output);
    }

    NOA_IH void c2r(cdouble_t* input, double* output, const fourier::Plan<double>& plan) {
        NOA_PROFILE_FUNCTION();
        fftw_execute_dft_c2r(plan.get(), reinterpret_cast<fftw_complex*>(input), output);
    }

    /// Computes the C2C transform using the \a plan.
    /// \param[in] input     Input. Should match the output used to create \a plan.
    /// \param[out] output   Output. Should match the output used to create \a plan.
    /// \param plan          Existing plan.
    ///
    /// \note This function is thread-safe as long as the \a input and \a output arrays are only accessed by
    ///       one single thread (i.e. the \a plan can be access by multiple threads concurrently).
    /// \note The arrays used to create the \a plan should be similar to \a input and \a output.
    ///       The shape should be the same. The input and output arrays are the same (in-place) or different
    ///       (out-of-place) if the plan was originally created to be in-place or out-of-place, respectively.
    ///       The alignment should be the same as well.
    NOA_IH void c2c(cfloat_t* input, cfloat_t* output, const fourier::Plan<float>& plan) {
        NOA_PROFILE_FUNCTION();
        fftwf_execute_dft(plan.get(),
                          reinterpret_cast<fftwf_complex*>(input),
                          reinterpret_cast<fftwf_complex*>(output));
    }

    NOA_IH void c2c(cdouble_t* input, cdouble_t* output, const fourier::Plan<double>& plan) {
        NOA_PROFILE_FUNCTION();
        fftw_execute_dft(plan.get(),
                         reinterpret_cast<fftw_complex*>(input),
                         reinterpret_cast<fftw_complex*>(output));
    }

    // -- "One time" transforms -- //

    /// Computes the R2C transform (i.e forward transform).
    /// \see fourier::Plan<float> for more details.
    /// \note \a input and \a output can be the same, which will trigger an in-place transform.
    NOA_IH void r2c(float* input, cfloat_t* output, size3_t shape, uint batches) {
        NOA_PROFILE_FUNCTION();
        Plan<float> fast_plan(input, output, shape, batches, fourier::ESTIMATE);
        fftwf_execute(fast_plan.get());
    }

    NOA_IH void r2c(double* input, cdouble_t* output, size3_t shape, uint batches) {
        NOA_PROFILE_FUNCTION();
        Plan<double> fast_plan(input, output, shape, batches, fourier::ESTIMATE);
        fftw_execute(fast_plan.get());
    }

    /// Computes the in-place R2C transform.
    NOA_IH void r2c(float* data, size3_t shape, uint batches) {
        r2c(data, reinterpret_cast<cfloat_t*>(data), shape, batches);
    }
    NOA_IH void r2c(double* data, size3_t shape, uint batches) {
        r2c(data, reinterpret_cast<cdouble_t*>(data), shape, batches);
    }

    /// Computes the C2R transform (i.e backward transform).
    /// \see fourier::Plan<float> for more details.
    /// \note \a input and \a output can be the same, which will trigger an in-place transform.
    NOA_IH void c2r(cfloat_t* input, float* output, size3_t shape, uint batches) {
        NOA_PROFILE_FUNCTION();
        Plan<float> fast_plan(input, output, shape, batches, fourier::ESTIMATE);
        fftwf_execute(fast_plan.get());
    }

    NOA_IH void c2r(cdouble_t* input, double* output, size3_t shape, uint batches) {
        NOA_PROFILE_FUNCTION();
        Plan<double> fast_plan(input, output, shape, batches, fourier::ESTIMATE);
        fftw_execute(fast_plan.get());
    }

    /// Computes the in-place C2R transform.
    NOA_IH void c2r(cfloat_t* data, size3_t shape, uint batches) {
        c2r(data, reinterpret_cast<float*>(data), shape, batches);
    }

    NOA_IH void c2r(cdouble_t* data, size3_t shape, uint batches) {
        c2r(data, reinterpret_cast<double*>(data), shape, batches);
    }

    /// Computes the C2C transform (i.e forward or backward transform depending on the \a sign).
    /// \see fourier::Plan<float> for more details.
    /// \note \a input and \a output can be the same, which will trigger an in-place transform.
    NOA_IH void c2c(cfloat_t* input, cfloat_t* output, size3_t shape, uint batches, Sign sign) {
        NOA_PROFILE_FUNCTION();
        Plan<float> fast_plan(input, output, shape, batches, sign, fourier::ESTIMATE);
        fftwf_execute(fast_plan.get());
    }

    NOA_IH void c2c(cdouble_t* input, cdouble_t* output, size3_t shape, uint batches, Sign sign) {
        NOA_PROFILE_FUNCTION();
        Plan<double> fast_plan(input, output, shape, batches, sign, fourier::ESTIMATE);
        fftw_execute(fast_plan.get());
    }

    /// Computes the in-place C2C transform.
    NOA_IH void c2c(cfloat_t* data, size3_t shape, uint batches, Sign sign) { c2c(data, data, shape, batches, sign); }
    NOA_IH void c2c(cdouble_t* data, size3_t shape, uint batches, Sign sign) { c2c(data, data, shape, batches, sign); }
}
