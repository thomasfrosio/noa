/// \file noa/cpu/fft/Transforms.h
/// \brief Fast Fourier Transforms.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <fftw3.h>

#include "noa/common/Definitions.h"
#include "noa/common/Profiler.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"
#include "noa/cpu/fft/Plan.h"

// -- Execute -- //
namespace noa::cpu::fft {
    /// Executes the \p plan.
    /// \param[in,out] stream Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note It is safe to execute the same plan in parallel by multiple threads. However, since a given plan operates
    ///       by default on a fixed array, one needs to use one of the new-array functions so that different threads
    ///       compute the transform on different data.
    template<typename T>
    NOA_IH void execute(const Plan<T>& plan, Stream& stream) {
        stream.enqueue([&]() {
            NOA_PROFILE_FUNCTION();
            if constexpr (std::is_same_v<T, float>)
                fftwf_execute(plan.get());
            else
                fftw_execute(plan.get());
        });
    }
}

// -- New-array transforms -- //
namespace noa::cpu::fft {
    /// Computes the forward R2C transform encoded in \p plan.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b host. Should match the input used to create \p plan.
    /// \param[out] output      On the \b host. Should match the output used to create \p plan.
    /// \param[in] plan         Existing plan.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note This function is thread-safe as long as \p input and \p output are only accessed by
    ///       one single thread. However \p plan can be access by multiple threads concurrently.
    /// \note The arrays used to create \p plan should be similar to \p input and \p output.
    ///       The shape should be the same. The input and output arrays are the same (in-place) or different
    ///       (out-of-place) if the plan was originally created to be in-place or out-of-place, respectively.
    ///       The alignment should be the same as well.
    template<typename T>
    NOA_IH void r2c(T* input, Complex<T>* output, const Plan<T>& plan, Stream& stream) {
        auto ptr = plan.get();
        stream.enqueue([=]() {
            NOA_PROFILE_FUNCTION();
            if constexpr (std::is_same_v<T, float>)
                fftwf_execute_dft_r2c(ptr, input, reinterpret_cast<fftwf_complex*>(output));
            else
                fftw_execute_dft_r2c(ptr, input, reinterpret_cast<fftw_complex*>(output));
        });
    }

    /// Computes the inverse C2R transform encoded in \p plan.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b host. Should match the input used to create \p plan.
    /// \param[out] output      On the \b host. Should match the output used to create \p plan.
    /// \param[in] plan         Existing plan.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note This function is thread-safe as long as \p input and \p output are only accessed by
    ///       one single thread. However \p plan can be access by multiple threads concurrently.
    /// \note The arrays used to create \p plan should be similar to \p input and \p output.
    ///       The shape should be the same. The input and output arrays are the same (in-place) or different
    ///       (out-of-place) if the plan was originally created to be in-place or out-of-place, respectively.
    ///       The alignment should be the same as well.
    template<typename T>
    NOA_IH void c2r(Complex<T>* input, T* output, const Plan<T>& plan, Stream& stream) {
        auto ptr = plan.get();
        stream.enqueue([=]() {
            NOA_PROFILE_FUNCTION();
            if constexpr (std::is_same_v<T, float>)
                fftwf_execute_dft_c2r(ptr, reinterpret_cast<fftwf_complex*>(input), output);
            else
                fftw_execute_dft_c2r(ptr, reinterpret_cast<fftw_complex*>(input), output);
        });
    }

    /// Computes the C2C transform encoded in \p plan.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b host. Should match the input used to create \p plan.
    /// \param[out] output      On the \b host. Should match the output used to create \p plan.
    /// \param[in] plan         Existing plan.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note This function is thread-safe as long as \p input and \p output are only accessed by
    ///       one single thread. However \p plan can be access by multiple threads concurrently.
    /// \note The arrays used to create \p plan should be similar to \p input and \p output.
    ///       The shape should be the same. The input and output arrays are the same (in-place) or different
    ///       (out-of-place) if the plan was originally created to be in-place or out-of-place, respectively.
    ///       The alignment should be the same as well.
    template<typename T>
    NOA_IH void c2c(Complex<T>* input, Complex<T>* output, const Plan<T>& plan, Stream& stream) {
        auto ptr = plan.get();
        stream.enqueue([=]() {
            NOA_PROFILE_FUNCTION();
            if constexpr (std::is_same_v<T, float>) {
                fftwf_execute_dft(ptr,
                                  reinterpret_cast<fftwf_complex*>(input),
                                  reinterpret_cast<fftwf_complex*>(output));
            } else {
                fftw_execute_dft(ptr,
                                 reinterpret_cast<fftw_complex*>(input),
                                 reinterpret_cast<fftw_complex*>(output));
            }
        });
    }
}

// -- "One time" transforms -- //
namespace noa::cpu::fft {
    /// Computes the forward R2C transform.
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b host. Real space array.
    /// \param[out] outputs     On the \b host. Non-redundant non-centered FFT(s). Can be equal to \p inputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void r2c(T* inputs, Complex<T>* outputs, size3_t shape, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        Plan fast_plan(inputs, outputs, shape, batches, fft::ESTIMATE, stream);
        execute(fast_plan, stream);
        stream.synchronize();
    }

    /// Computes the forward R2C transform.
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b host. Real space array.
    /// \param input_pitch      Pitch, in real elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Non-redundant non-centered FFT(s). Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in complex elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void r2c(T* inputs, size3_t input_pitch, Complex<T>* outputs, size3_t output_pitch,
                    size3_t shape, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        Plan fast_plan(inputs, input_pitch, outputs, output_pitch, shape, batches, fft::ESTIMATE, stream);
        execute(fast_plan, stream);
        stream.synchronize();
    }

    /// Computes the in-place R2C transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host. Input should be the real space array with appropriate padding.
    /// \param shape            Logical {fast, medium, slow} shape of \p data.
    /// \param batches          Number of contiguous batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void r2c(T* data, size3_t shape, size_t batches, Stream& stream) {
        r2c(data, reinterpret_cast<Complex<T>*>(data), shape, batches, stream);
    }

    /// Computes the in-place R2C transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host. Input should be the real space array with appropriate padding.
    /// \param pitch            Pitch, in real elements, of \p data.
    /// \param shape            Logical {fast, medium, slow} shape of \p data.
    /// \param batches          Number of batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \note Since the transform is in-place, it must be able to hold the complex non-redundant transform.
    ///       As such, \p pitch.x must have the appropriate padding, and thus be divisible by 2.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void r2c(T* data, size3_t pitch, size3_t shape, size_t batches, Stream& stream) {
        NOA_ASSERT(!(pitch.x % 2)); // must be even to match the complex indexing
        NOA_ASSERT(pitch.x >= shape.x + 1 + int(!(pitch.x % 2))); // at least 1 (if odd) or 2 (if even) real element
        size3_t complex_pitch = {pitch.x / 2, pitch.y, pitch.z};
        r2c(data, pitch, reinterpret_cast<Complex<T>*>(data), complex_pitch, shape, batches, stream);
    }

    /// Computes the backward C2R transform.
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b host. Non-redundant non-centered FFT(s).
    /// \param[out] outputs     On the \b host. Real space array. Can be equal to \p inputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2r(Complex<T>* inputs, T* outputs, size3_t shape, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        Plan fast_plan(inputs, outputs, shape, batches, fft::ESTIMATE, stream);
        execute(fast_plan, stream);
        stream.synchronize();
    }

    /// Computes the backward C2R transform.
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b host. Non-redundant non-centered FFT(s).
    /// \param input_pitch      Pitch, in complex elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Real space array. Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in real elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2r(Complex<T>* inputs, size3_t input_pitch, T* outputs, size3_t output_pitch,
                    size3_t shape, size_t batches, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        Plan fast_plan(inputs, input_pitch, outputs, output_pitch, shape, batches, fft::ESTIMATE, stream);
        execute(fast_plan, stream);
        stream.synchronize();
    }

    /// Computes the in-place C2R transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host. Input should be the non-redundant non-centered FFT(s).
    /// \param shape            Logical {fast, medium, slow} shape of \p data.
    /// \param batches          Number of contiguous batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2r(Complex<T>* data, size3_t shape, size_t batches, Stream& stream) {
        c2r(data, reinterpret_cast<T*>(data), shape, batches, stream);
    }

    /// Computes the in-place C2R transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host. Input should be the non-redundant non-centered FFT(s).
    /// \param pitch            Pitch, in complex elements, of \p data.
    /// \param shape            Logical {fast, medium, slow} shape of \p data.
    /// \param batches          Number of batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \note Since the transform is in-place, it must be able to hold the complex non-redundant transform.
    ///       As such, \p pitch must have the appropriate padding, thus be divisible by 2.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2r(Complex<T>* data, size3_t pitch, size3_t shape, size_t batches, Stream& stream) {
        size3_t real_pitch = {pitch.x * 2, pitch.y, pitch.z};
        c2r(data, pitch, reinterpret_cast<T*>(data), real_pitch, shape, batches, stream);
    }

    /// Computes the C2C transform.
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b host. Input complex data.
    /// \param[out] outputs     On the \b host. Non-centered FFT(s). Can be equal to \p inputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2c(Complex<T>* inputs, Complex<T>* outputs, size3_t shape, size_t batches, Sign sign, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        Plan fast_plan(inputs, outputs, shape, batches, sign, fft::ESTIMATE, stream);
        execute(fast_plan, stream);
        stream.synchronize();
    }

    /// Computes the C2C transform.
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b host. Input complex data.
    /// \param input_pitch      Pitch, in complex elements, of \p inputs.
    /// \param[out] outputs     On the \b host. Non-centered FFT(s). Can be equal to \p inputs.
    /// \param output_pitch     Pitch, in complex elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of batches.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2c(Complex<T>* inputs, size3_t input_pitch, Complex<T>* outputs, size3_t output_pitch,
                    size3_t shape, size_t batches, Sign sign, Stream& stream) {
        NOA_PROFILE_FUNCTION();
        Plan fast_plan(inputs, input_pitch, outputs, output_pitch, shape, batches, sign, fft::ESTIMATE, stream);
        execute(fast_plan, stream);
        stream.synchronize();
    }

    /// Computes the in-place C2C transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host.
    /// \param shape            Logical {fast, medium, slow} shape of \p data.
    /// \param batches          Number of contiguous batches.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2c(Complex<T>* data, size3_t shape, size_t batches, Sign sign, Stream& stream) {
        c2c(data, data, shape, batches, sign, stream);
    }

    /// Computes the in-place C2C transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host.
    /// \param pitch            Pitch, in complex elements, of \p data.
    /// \param shape            Logical {fast, medium, slow} shape of \p data.
    /// \param batches          Number of batches.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///                         The stream is synchronized when the function returns.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2c(Complex<T>* data, size3_t pitch, size3_t shape, size_t batches, Sign sign, Stream& stream) {
        c2c(data, pitch, data, pitch, shape, batches, sign, stream);
    }
}
