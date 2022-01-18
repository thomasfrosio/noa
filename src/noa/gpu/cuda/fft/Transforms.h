/// \file noa/gpu/cuda/fft/Transforms.h
/// \brief Fast Fourier Transforms.
/// \author Thomas - ffyr2w
/// \date 19 Jun 2021

#pragma once

#include <cufft.h>

#include "noa/common/Definitions.h"
#include "noa/gpu/cuda/Types.h"
#include "noa/gpu/cuda/Stream.h"
#include "noa/gpu/cuda/fft/Plan.h"
#include "noa/gpu/cuda/fft/Exception.h"

namespace noa::cuda::fft {
    using namespace ::noa::fft;

    /// Computes the forward R2C transform encoded in \p plan.
    /// \tparam T           float, double.
    /// \param[in] input    On the \b device. Should match the config encoded in \p plan.
    /// \param[out] output  On the \b device. Should match the config encoded in \p plan.
    /// \param[in] plan     Existing plan.
    /// \note This functions is asynchronous with respect to the host and may return before completion.
    ///       All operations are enqueued to the stream associated with the \p plan.
    template<typename T>
    NOA_IH void r2c(T* input, Complex<T>* output, const Plan<T>& plan) {
        if constexpr (std::is_same_v<T, float>)
            NOA_THROW_IF(cufftExecR2C(plan.get(), input, reinterpret_cast<cufftComplex*>(output)));
        else
            NOA_THROW_IF(cufftExecD2Z(plan.get(), input, reinterpret_cast<cufftDoubleComplex*>(output)));
    }

    /// Computes the backward C2R transform encoded in \p plan.
    /// \tparam T           float, double.
    /// \param[in] input    On the \b device. Should match the config encoded in \p plan.
    /// \param[out] output  On the \b device. Should match the config encoded in \p plan.
    /// \param[in] plan     Existing plan.
    /// \note This functions is asynchronous with respect to the host and may return before completion.
    ///       All operations are enqueued to the stream associated with the \p plan.
    template<typename T>
    NOA_IH void c2r(Complex<T>* input, T* output, const Plan<T>& plan) {
        if constexpr (std::is_same_v<T, float>)
            NOA_THROW_IF(cufftExecC2R(plan.get(), reinterpret_cast<cufftComplex*>(input), output));
        else
            NOA_THROW_IF(cufftExecZ2D(plan.get(), reinterpret_cast<cufftDoubleComplex*>(input), output));
    }

    /// Computes the C2C transform using the \p plan.
    /// \tparam T           float, double.
    /// \param[in] input    On the \b device. Should match the config encoded in \p plan.
    /// \param[out] output  On the \b device. Should match the config encoded in \p plan.
    /// \param[in] plan     Existing plan.
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \note This functions is asynchronous with respect to the host and may return before completion.
    ///       All operations are enqueued to the stream associated with the \p plan.
    template<typename T>
    NOA_IH void c2c(Complex<T>* input, Complex<T>* output, const Plan<T>& plan, Sign sign) {
        if constexpr (std::is_same_v<T, float>) {
            NOA_THROW_IF(cufftExecC2C(plan.get(),
                                      reinterpret_cast<cufftComplex*>(input),
                                      reinterpret_cast<cufftComplex*>(output), sign));
        } else {
            NOA_THROW_IF(cufftExecZ2Z(plan.get(),
                                      reinterpret_cast<cufftDoubleComplex*>(input),
                                      reinterpret_cast<cufftDoubleComplex*>(output), sign));
        }
    }
}

// -- "One time" transforms -- //
namespace noa::cuda::fft {
    /// Computes the forward R2C transform.
    /// \tparam T           float, double.
    /// \param[in] inputs       On the \b device. Real space array(s).
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device. Non-redundant non-centered FFT(s).
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p outputs can be equal to \p inputs. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void r2c(T* inputs, size_t input_pitch, Complex<T>* outputs, size_t output_pitch,
                    size3_t shape, size_t batches, Stream& stream) {
        Plan<T> fast_plan(fft::R2C, shape, batches, input_pitch, output_pitch, stream);
        r2c(inputs, outputs, fast_plan);
    }

    /// Computes the forward R2C transform.
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b device. Real space array(s).
    /// \param[out] outputs     On the \b device. Non-redundant non-centered FFT.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p outputs can be equal to \p inputs. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void r2c(T* inputs, Complex<T>* outputs, size3_t shape, size_t batches, Stream& stream) {
        Plan<T> fast_plan(fft::R2C, shape, batches, stream);
        r2c(inputs, outputs, fast_plan);
    }

    /// Computes the backward C2R transform.
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b device. Non-redundant non-centered FFT(s).
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device. Real space array(s).
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p outputs can be equal to \p inputs. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void c2r(Complex<T>* inputs, size_t input_pitch, T* outputs, size_t output_pitch,
                    size3_t shape, size_t batches, Stream& stream) {
        Plan<T> fast_plan(fft::C2R, shape, batches, input_pitch, output_pitch, stream);
        c2r(inputs, outputs, fast_plan);
    }

    /// Computes the backward C2R transform.
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b device. Non-redundant non-centered FFT.
    /// \param[out] outputs     On the \b device. Real space array(s).
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p outputs can be equal to \p inputs. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void c2r(Complex<T>* inputs, T* outputs, size3_t shape, size_t batches, Stream& stream) {
        Plan<T> fast_plan(fft::C2R, shape, batches, stream);
        c2r(inputs, outputs, fast_plan);
    }

    /// Computes the C2C transform.
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b device.
    /// \param input_pitch      Pitch, in elements, of \p inputs.
    /// \param[out] outputs     On the \b device.
    /// \param output_pitch     Pitch, in elements, of \p outputs.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p outputs can be equal to \p inputs. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void c2c(Complex<T>* inputs, size_t input_pitch,
                    Complex<T>* outputs, size_t output_pitch,
                    size3_t shape, size_t batches, Sign sign, Stream& stream) {
        Plan<T> fast_plan(fft::C2C, shape, batches, input_pitch, output_pitch, stream);
        c2c(inputs, outputs, fast_plan, sign);
    }

    /// Computes the C2C transform.
    /// \tparam T               float, double.
    /// \param[in] inputs       On the \b device.
    /// \param[out] outputs     On the \b device.
    /// \param shape            Logical {fast, medium, slow} shape of \p inputs and \p outputs.
    /// \param batches          Number of contiguous batches.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p outputs can be equal to \p inputs. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void c2c(Complex<T>* inputs, Complex<T>* outputs, size3_t shape, size_t batches,
                    Sign sign, Stream& stream) {
        Plan<T> fast_plan(fft::C2C, shape, batches, stream);
        c2c(inputs, outputs, fast_plan, sign);
    }
}
