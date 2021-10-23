/// \file noa/cpu/fft/Transforms.h
/// \brief Fast Fourier Transforms.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <fftw3.h>

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"
#include "noa/cpu/fft/Plan.h"

// -- Execute -- //
namespace noa::cpu::fft {
    /// Executes the \p plan.
    /// \note It is safe to execute the same plan in parallel by multiple threads. However, since a given plan operates
    ///       by default on a fixed array, one needs to use one of the new-array functions so that different threads
    ///       compute the transform of different data.
    NOA_IH void execute(const fft::Plan<float>& plan) {
        NOA_PROFILE_FUNCTION();
        fftwf_execute(plan.get());
    }
    NOA_IH void execute(const fft::Plan<double>& plan) {
        NOA_PROFILE_FUNCTION();
        fftw_execute(plan.get());
    }
}

// -- New-array transforms -- //
namespace noa::cpu::fft {
    /// Computes the R2C transform (i.e forward transform) using \p plan.
    /// \param[in] input     On the \b host. Should match the input used to create \p plan.
    /// \param[out] output   On the \b host. Should match the output used to create \p plan.
    /// \param plan          Existing plan.
    ///
    /// \note This function is thread-safe as long as the \p input and \p output arrays are only accessed by
    ///       one single thread (i.e. \p plan can be access by multiple threads concurrently).
    /// \note The arrays used to create \p plan should be similar to \p input and \p output.
    ///       The shape should be the same. The input and output arrays are the same (in-place) or different
    ///       (out-of-place) if the plan was originally created to be in-place or out-of-place, respectively.
    ///       The alignment should be the same as well.
    NOA_IH void r2c(float* input, cfloat_t* output, const fft::Plan<float>& plan) {
        NOA_PROFILE_FUNCTION();
        fftwf_execute_dft_r2c(plan.get(), input, reinterpret_cast<fftwf_complex*>(output));
    }
    NOA_IH void r2c(double* input, cdouble_t* output, const fft::Plan<double>& plan) {
        NOA_PROFILE_FUNCTION();
        fftw_execute_dft_r2c(plan.get(), input, reinterpret_cast<fftw_complex*>(output));
    }

    /// Computes the C2R transform (i.e backward transform) using \p plan.
    /// \param[in] input     On the \b host. Should match the input used to create \p plan.
    /// \param[out] output   On the \b host. Should match the output used to create \p plan.
    /// \param plan          Existing plan.
    ///
    /// \note This function is thread-safe as long as the \p input and \p output arrays are only accessed by
    ///       one single thread (i.e. \p plan can be access by multiple threads concurrently).
    /// \note The arrays used to create \p plan should be similar to \p input and \p output.
    ///       The shape should be the same. The input and output arrays are the same (in-place) or different
    ///       (out-of-place) if the plan was originally created to be in-place or out-of-place, respectively.
    ///       The alignment should be the same as well.
    NOA_IH void c2r(cfloat_t* input, float* output, const fft::Plan<float>& plan) {
        NOA_PROFILE_FUNCTION();
        fftwf_execute_dft_c2r(plan.get(), reinterpret_cast<fftwf_complex*>(input), output);
    }
    NOA_IH void c2r(cdouble_t* input, double* output, const fft::Plan<double>& plan) {
        NOA_PROFILE_FUNCTION();
        fftw_execute_dft_c2r(plan.get(), reinterpret_cast<fftw_complex*>(input), output);
    }

    /// Computes the C2C transform using \p plan.
    /// \param[in] input     On the \b host. Should match the input used to create \p plan.
    /// \param[out] output   On the \b host. Should match the output used to create \p plan.
    /// \param plan          Existing plan.
    ///
    /// \note This function is thread-safe as long as the \p input and \p output arrays are only accessed by
    ///       one single thread (i.e. \p plan can be access by multiple threads concurrently).
    /// \note The arrays used to create \p plan should be similar to \p input and \p output.
    ///       The shape should be the same. The input and output arrays are the same (in-place) or different
    ///       (out-of-place) if the plan was originally created to be in-place or out-of-place, respectively.
    ///       The alignment should be the same as well.
    NOA_IH void c2c(cfloat_t* input, cfloat_t* output, const fft::Plan<float>& plan) {
        NOA_PROFILE_FUNCTION();
        fftwf_execute_dft(plan.get(),
                          reinterpret_cast<fftwf_complex*>(input),
                          reinterpret_cast<fftwf_complex*>(output));
    }
    NOA_IH void c2c(cdouble_t* input, cdouble_t* output, const fft::Plan<double>& plan) {
        NOA_PROFILE_FUNCTION();
        fftw_execute_dft(plan.get(),
                         reinterpret_cast<fftw_complex*>(input),
                         reinterpret_cast<fftw_complex*>(output));
    }
}

// -- "One time" transforms -- //
namespace noa::cpu::fft {
    /// Computes the R2C transform (i.e forward transform).
    /// \param[in] inputs   On the \b host. Real space array.
    /// \param[out] outputs On the \b host. Non-redundant, non-centered transform. Can be equal to \p inputs.
    /// \param shape        Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches      Batch size, in number of contiguous batches.
    /// \see fft::Plan<float> for more details.
    NOA_IH void r2c(float* inputs, cfloat_t* outputs, size3_t shape, uint batches) {
        NOA_PROFILE_FUNCTION();
        Plan<float> fast_plan(inputs, outputs, shape, batches, fft::ESTIMATE);
        fftwf_execute(fast_plan.get());
    }
    NOA_IH void r2c(double* inputs, cdouble_t* outputs, size3_t shape, uint batches) {
        NOA_PROFILE_FUNCTION();
        Plan<double> fast_plan(inputs, outputs, shape, batches, fft::ESTIMATE);
        fftw_execute(fast_plan.get());
    }

    /// Computes the in-place R2C transform.
    /// \param[in] data     On the \b host. Real space array with appropriate padding.
    /// \param shape        Logical {fast, medium, slow} shape of \p data.
    /// \param batches      Batch size, in number of contiguous batches.
    /// \see fft::Plan<float> for more details.
    NOA_IH void r2c(float* data, size3_t shape, uint batches) {
        r2c(data, reinterpret_cast<cfloat_t*>(data), shape, batches);
    }
    NOA_IH void r2c(double* data, size3_t shape, uint batches) {
        r2c(data, reinterpret_cast<cdouble_t*>(data), shape, batches);
    }

    /// Computes the C2R transform (i.e backward transform).
    /// \param[in] inputs   On the \b host. Non-redundant, non-centered transform.
    /// \param[out] outputs On the \b host. Real space array. Can be equal to \p inputs.
    /// \param shape        Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches      Batch size, in number of contiguous batches.
    /// \see fft::Plan<float> for more details.
    NOA_IH void c2r(cfloat_t* inputs, float* outputs, size3_t shape, uint batches) {
        NOA_PROFILE_FUNCTION();
        Plan<float> fast_plan(inputs, outputs, shape, batches, fft::ESTIMATE);
        fftwf_execute(fast_plan.get());
    }
    NOA_IH void c2r(cdouble_t* inputs, double* outputs, size3_t shape, uint batches) {
        NOA_PROFILE_FUNCTION();
        Plan<double> fast_plan(inputs, outputs, shape, batches, fft::ESTIMATE);
        fftw_execute(fast_plan.get());
    }

    /// Computes the in-place C2R transform.
    /// \param[in] data     On the \b host. Non-redundant, non-centered transform.
    /// \param shape        Logical {fast, medium, slow} shape of \p data.
    /// \param batches      Batch size, in number of contiguous batches.
    /// \see fft::Plan<float> for more details.
    NOA_IH void c2r(cfloat_t* data, size3_t shape, uint batches) {
        c2r(data, reinterpret_cast<float*>(data), shape, batches);
    }
    NOA_IH void c2r(cdouble_t* data, size3_t shape, uint batches) {
        c2r(data, reinterpret_cast<double*>(data), shape, batches);
    }

    /// Computes the C2C transform (i.e forward or backward transform depending on the \p sign).
    /// \param[in] inputs   On the \b host.
    /// \param[out] outputs On the \b host. Can be equal to \p inputs.
    /// \param shape        Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches      Batch size, in number of contiguous batches.
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c FORWARD) or +1 (\c BACKWARD).
    /// \see fft::Plan<float> for more details.
    NOA_IH void c2c(cfloat_t* inputs, cfloat_t* outputs, size3_t shape, uint batches, Sign sign) {
        NOA_PROFILE_FUNCTION();
        Plan<float> fast_plan(inputs, outputs, shape, batches, sign, fft::ESTIMATE);
        fftwf_execute(fast_plan.get());
    }
    NOA_IH void c2c(cdouble_t* inputs, cdouble_t* outputs, size3_t shape, uint batches, Sign sign) {
        NOA_PROFILE_FUNCTION();
        Plan<double> fast_plan(inputs, outputs, shape, batches, sign, fft::ESTIMATE);
        fftw_execute(fast_plan.get());
    }

    /// Computes the in-place C2C transform.
    /// \param[in] data     On the \b host.
    /// \param shape        Logical {fast, medium, slow} shape of \p data.
    /// \param batches      Batch size, in number of contiguous batches.
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c FORWARD) or +1 (\c BACKWARD).
    /// \see fft::Plan<float> for more details.
    NOA_IH void c2c(cfloat_t* data, size3_t shape, uint batches, Sign sign) {
        c2c(data, data, shape, batches, sign);
    }
    NOA_IH void c2c(cdouble_t* data, size3_t shape, uint batches, Sign sign) {
        c2c(data, data, shape, batches, sign);
    }
}
