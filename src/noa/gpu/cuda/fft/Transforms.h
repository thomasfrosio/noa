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
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Real space array(s).
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param[out] output      On the \b device. Non-redundant non-centered FFT(s).
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p output can be equal to \p input. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void r2c(T* input, size4_t input_stride,
                    Complex<T>* output, size4_t output_stride,
                    size4_t shape, Stream& stream) {
        Plan<T> plan(fft::R2C, input_stride, output_stride, shape, stream);
        r2c(input, output, plan);
    }

    /// Computes the forward R2C transform.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Real space array(s).
    /// \param[out] output      On the \b device. Non-redundant non-centered FFT.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p output can be equal to \p input. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void r2c(T* input, Complex<T>* output, size4_t shape, Stream& stream) {
        Plan<T> plan(fft::R2C, shape, stream);
        r2c(input, output, plan);
    }

    /// Computes the in-place R2C transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host. Input should be the real space array with appropriate padding.
    /// \param shape            Rightmost shape of \p data.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void r2c(T* data, size4_t shape, Stream& stream) {
        r2c(data, reinterpret_cast<Complex<T>*>(data), shape, stream);
    }

    /// Computes the in-place R2C transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host. Input should be the real space array with appropriate padding.
    /// \param stride           Rightmost strides, in real elements, of \p data.
    /// \param shape            Rightmost shape of \p data.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note Since the transform is in-place, it must be able to hold the complex non-redundant transform.
    ///       As such, the innermost dimension must have the appropriate padding. See fft::Plan for more details
    template<typename T>
    NOA_IH void r2c(T* data, size4_t stride, size4_t shape, Stream& stream) {
        NOA_ASSERT(!(stride.pitch(2) % 2));
        NOA_ASSERT(stride.pitch(2) >= shape[3] + 1 + size_t(!(shape[3] % 2)));

        const size4_t complex_stride{stride[0] / 2, stride[1] / 2, stride[2] / 2, stride[3]};
        r2c(data, stride, reinterpret_cast<Complex<T>*>(data), complex_stride, shape, stream);
    }

    /// Computes the backward C2R transform.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Non-redundant non-centered FFT(s).
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param[out] output      On the \b device. Real space array(s).
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p output can be equal to \p input. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void c2r(Complex<T>* input, size4_t input_stride,
                    T* output, size4_t output_stride,
                    size4_t shape, Stream& stream) {
        Plan<T> plan(fft::C2R, input_stride, output_stride, shape, stream);
        c2r(input, output, plan);
    }

    /// Computes the backward C2R transform.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Non-redundant non-centered FFT(s).
    /// \param[out] output      On the \b device. Real space array(s).
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p output can be equal to \p input. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void c2r(Complex<T>* input, T* output, size4_t shape, Stream& stream) {
        Plan<T> plan(fft::C2R, shape, stream);
        c2r(input, output, plan);
    }

    /// Computes the in-place C2R transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host. Input should be the non-redundant non-centered FFT(s).
    /// \param shape            Rightmost shape of \p data.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note \note This function is asynchronous relative to the host and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2r(Complex<T>* data, size4_t shape, Stream& stream) {
        c2r(data, reinterpret_cast<T*>(data), shape, stream);
    }

    /// Computes the in-place C2R transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host. Input should be the non-redundant non-centered FFT(s).
    /// \param stride           Rightmost strides, in complex elements, of \p data.
    /// \param shape            Rightmost shape of \p data.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note \note This function is asynchronous relative to the host and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2r(Complex<T>* data, size4_t stride, size4_t shape, Stream& stream) {
        const size4_t real_stride{stride[0] * 2, stride[1] * 2, stride[2] * 2, stride[3]};
        c2r(data, stride, reinterpret_cast<T*>(data), real_stride, shape, stream);
    }

    /// Computes the C2C transform.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device.
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param[out] output      On the \b device.
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p output can be equal to \p input.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2c(Complex<T>* input, size4_t input_stride,
                    Complex<T>* output, size4_t output_stride,
                    size4_t shape, Sign sign, Stream& stream) {
        Plan<T> fast_plan(fft::C2C, input_stride, output_stride, shape, stream);
        c2c(input, output, fast_plan, sign);
    }

    /// Computes the C2C transform.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device.
    /// \param[out] output      On the \b device.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p output can be equal to \p input.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2c(Complex<T>* input, Complex<T>* output, size4_t shape, Sign sign, Stream& stream) {
        Plan<T> fast_plan(fft::C2C, shape, stream);
        c2c(input, output, fast_plan, sign);
    }

    /// Computes the in-place C2C transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host.
    /// \param shape            Rightmost shape of \p data.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2c(Complex<T>* data, size4_t shape, Sign sign, Stream& stream) {
        c2c(data, data, shape, sign, stream);
    }

    /// Computes the in-place C2C transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host.
    /// \param stride           Rightmost strides, in complex elements, of \p data.
    /// \param shape            Rightmost shape of \p data.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2c(Complex<T>* data, size4_t stride, size4_t shape, Sign sign, Stream& stream) {
        c2c(data, stride, data, stride, shape, sign, stream);
    }
}
