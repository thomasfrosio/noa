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
    ///       compute the transform on different data.
    template<typename T>
    NOA_IH void execute(const Plan<T>& plan) {
        NOA_PROFILE_FUNCTION();
        if constexpr (std::is_same_v<T, float>)
            fftwf_execute(plan.get());
        else
            fftw_execute(plan.get());
    }
}

// -- New-array transforms -- //
namespace noa::cpu::fft {
    /// Computes the forward R2C transform encoded in \p plan.
    /// \tparam T           float, double.
    /// \param[in] input    On the \b host. Should match the input used to create \p plan.
    /// \param[out] output  On the \b host. Should match the output used to create \p plan.
    /// \param[in] plan     Existing plan.
    ///
    /// \note This function is thread-safe as long as \p input and \p output are only accessed by
    ///       one single thread. However \p plan can be access by multiple threads concurrently.
    /// \note The arrays used to create \p plan should be similar to \p input and \p output.
    ///       The shape should be the same. The input and output arrays are the same (in-place) or different
    ///       (out-of-place) if the plan was originally created to be in-place or out-of-place, respectively.
    ///       The alignment should be the same as well.
    template<typename T>
    NOA_IH void r2c(T* input, Complex<T>* output, const Plan<T>& plan) {
        NOA_PROFILE_FUNCTION();
        if constexpr (std::is_same_v<T, float>)
            fftwf_execute_dft_r2c(plan.get(), input, reinterpret_cast<fftwf_complex*>(output));
        else
            fftw_execute_dft_r2c(plan.get(), input, reinterpret_cast<fftw_complex*>(output));
    }

    /// Computes the inverse C2R transform encoded in \p plan.
    /// \tparam T           float, double.
    /// \param[in] input    On the \b host. Should match the input used to create \p plan.
    /// \param[out] output  On the \b host. Should match the output used to create \p plan.
    /// \param[in] plan     Existing plan.
    ///
    /// \note This function is thread-safe as long as \p input and \p output are only accessed by
    ///       one single thread. However \p plan can be access by multiple threads concurrently.
    /// \note The arrays used to create \p plan should be similar to \p input and \p output.
    ///       The shape should be the same. The input and output arrays are the same (in-place) or different
    ///       (out-of-place) if the plan was originally created to be in-place or out-of-place, respectively.
    ///       The alignment should be the same as well.
    template<typename T>
    NOA_IH void c2r(Complex<T>* input, T* output, const Plan<T>& plan) {
        NOA_PROFILE_FUNCTION();
        if constexpr (std::is_same_v<T, float>)
            fftwf_execute_dft_c2r(plan.get(), reinterpret_cast<fftwf_complex*>(input), output);
        else
            fftw_execute_dft_c2r(plan.get(), reinterpret_cast<fftw_complex*>(input), output);
    }

    /// Computes the C2C transform encoded in \p plan.
    /// \tparam T           float, double.
    /// \param[in] input    On the \b host. Should match the input used to create \p plan.
    /// \param[out] output  On the \b host. Should match the output used to create \p plan.
    /// \param[in] plan     Existing plan.
    ///
    /// \note This function is thread-safe as long as \p input and \p output are only accessed by
    ///       one single thread. However \p plan can be access by multiple threads concurrently.
    /// \note The arrays used to create \p plan should be similar to \p input and \p output.
    ///       The shape should be the same. The input and output arrays are the same (in-place) or different
    ///       (out-of-place) if the plan was originally created to be in-place or out-of-place, respectively.
    ///       The alignment should be the same as well.
    template<typename T>
    NOA_IH void c2c(Complex<T>* input, Complex<T>* output, const Plan<T>& plan) {
        NOA_PROFILE_FUNCTION();
        if constexpr (std::is_same_v<T, float>) {
            fftwf_execute_dft(plan.get(),
                              reinterpret_cast<fftwf_complex*>(input),
                              reinterpret_cast<fftwf_complex*>(output));
        } else {
            fftw_execute_dft(plan.get(),
                             reinterpret_cast<fftw_complex*>(input),
                             reinterpret_cast<fftw_complex*>(output));
        }
    }
}

// -- "One time" transforms -- //
namespace noa::cpu::fft {
    /// Computes the forward R2C transform.
    /// \tparam T           float, double.
    /// \param[in] inputs   On the \b host. Real space array.
    /// \param[out] outputs On the \b host. Non-redundant non-centered FFT(s). Can be equal to \p inputs.
    /// \param shape        Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches      Number of contiguous batches.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void r2c(T* inputs, Complex<T>* outputs, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        Plan fast_plan(inputs, outputs, shape, batches, fft::ESTIMATE);
        execute(fast_plan);
    }

    /// Computes the in-place R2C transform.
    /// \tparam T           float, double.
    /// \param[in] data     On the \b host. Input should be the real space array with appropriate padding.
    /// \param shape        Logical {fast, medium, slow} shape of \p data.
    /// \param batches      Number of contiguous batches.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void r2c(T* data, size3_t shape, size_t batches) {
        r2c(data, reinterpret_cast<Complex<T>*>(data), shape, batches);
    }

    /// Computes the backward C2R transform.
    /// \tparam T           float, double.
    /// \param[in] inputs   On the \b host. Non-redundant non-centered FFT(s).
    /// \param[out] outputs On the \b host. Real space array. Can be equal to \p inputs.
    /// \param shape        Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches      Number of contiguous batches.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2r(Complex<T>* inputs, T* outputs, size3_t shape, size_t batches) {
        NOA_PROFILE_FUNCTION();
        Plan fast_plan(inputs, outputs, shape, batches, fft::ESTIMATE);
        execute(fast_plan);
    }

    /// Computes the in-place C2R transform.
    /// \tparam T           float, double.
    /// \param[in] data     On the \b host. Input should be the non-redundant non-centered FFT(s).
    /// \param shape        Logical {fast, medium, slow} shape of \p data.
    /// \param batches      Number of contiguous batches.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2r(Complex<T>* data, size3_t shape, size_t batches) {
        c2r(data, reinterpret_cast<T*>(data), shape, batches);
    }

    /// Computes the C2C transform.
    /// \tparam T           float, double.
    /// \param[in] inputs   On the \b host. Input complex data.
    /// \param[out] outputs On the \b host. Non-centered FFT(s). Can be equal to \p inputs.
    /// \param shape        Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches      Number of contiguous batches.
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2c(Complex<T>* inputs, Complex<T>* outputs, size3_t shape, size_t batches, Sign sign) {
        NOA_PROFILE_FUNCTION();
        Plan fast_plan(inputs, outputs, shape, batches, sign, fft::ESTIMATE);
        execute(fast_plan);
    }

    /// Computes the in-place C2C transform.
    /// \tparam T           float, double.
    /// \param[in] data     On the \b host.
    /// \param shape        Logical {fast, medium, slow} shape of \p data.
    /// \param batches      Number of contiguous batches.
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2c(Complex<T>* data, size3_t shape, size_t batches, Sign sign) {
        c2c(data, data, shape, batches, sign);
    }
}
