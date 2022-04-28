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
#include "noa/gpu/cuda/math/Ewise.h"

namespace noa::cuda::fft {
    using Norm = noa::fft::Norm;
    using Sign = noa::fft::Sign;
}

namespace noa::cuda::fft::details {
    template<bool HALF, typename T>
    void normalize(const shared_t<T[]>& array, size4_t stride, size4_t shape, Sign sign, Norm norm, Stream& stream) {
        using real_t = noa::traits::value_type_t<T>;
        const size3_t shape_{shape[1], shape[2], shape[3]};
        const auto count = static_cast<real_t>(noa::math::prod(shape_));
        const auto scale = norm == Norm::NORM_ORTHO ? noa::math::sqrt(count) : count;
        if (sign == Sign::FORWARD && (norm == Norm::NORM_FORWARD || norm == Norm::NORM_ORTHO)) {
            math::ewise(array, stride, 1 / scale, array, stride,
                        HALF ? shape.fft() : shape, noa::math::multiply_t{}, stream);
        } else if (sign == Sign::BACKWARD && (norm == Norm::NORM_BACKWARD || norm == Norm::NORM_ORTHO)) {
            math::ewise(array, stride, 1 / scale, array, stride,
                        shape, noa::math::multiply_t{}, stream);
        }
    }
}

namespace noa::cuda::fft {
    using namespace ::noa::fft;

    /// Computes the forward R2C transform encoded in \p plan.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Should match the config encoded in \p plan.
    /// \param[out] output      On the \b device. Should match the config encoded in \p plan.
    /// \param[in] plan         Valid plan.
    /// \param[in,out] stream   Stream on which to enqueue this function (and all operations associated with \p plan)
    /// \note This functions is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void r2c(const shared_t<T[]>& input,
                    const shared_t<Complex<T>[]>& output,
                    const Plan<T>& plan, Stream& stream) {
        NOA_THROW_IF(cufftSetStream(plan.get(), stream.get()));
        if constexpr (std::is_same_v<T, float>)
            NOA_THROW_IF(cufftExecR2C(plan.get(), input.get(), reinterpret_cast<cufftComplex*>(output.get())));
        else
            NOA_THROW_IF(cufftExecD2Z(plan.get(), input.get(), reinterpret_cast<cufftDoubleComplex*>(output.get())));
    }

    /// Computes the backward C2R transform encoded in \p plan.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Should match the config encoded in \p plan.
    /// \param[out] output      On the \b device. Should match the config encoded in \p plan.
    /// \param[in] plan         Valid plan.
    /// \param[in,out] stream   Stream on which to enqueue this function (and all operations associated with \p plan)
    /// \note This functions is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void c2r(const shared_t<Complex<T>[]>& input,
                    const shared_t<T[]>& output,
                    const Plan<T>& plan, Stream& stream) {
        NOA_THROW_IF(cufftSetStream(plan.get(), stream.get()));
        if constexpr (std::is_same_v<T, float>)
            NOA_THROW_IF(cufftExecC2R(plan.get(), reinterpret_cast<cufftComplex*>(input.get()), output.get()));
        else
            NOA_THROW_IF(cufftExecZ2D(plan.get(), reinterpret_cast<cufftDoubleComplex*>(input.get()), output.get()));
    }

    /// Computes the C2C transform using the \p plan.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Should match the config encoded in \p plan.
    /// \param[out] output      On the \b device. Should match the config encoded in \p plan.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param[in] plan         Valid plan.
    /// \param[in,out] stream   Stream on which to enqueue this function (and all operations associated with \p plan)
    /// \note This functions is asynchronous with respect to the host and may return before completion.
    template<typename T>
    NOA_IH void c2c(const shared_t<Complex<T>[]>& input,
                    const shared_t<Complex<T>[]>& output,
                    Sign sign, const Plan<T>& plan, Stream& stream) {
        NOA_THROW_IF(cufftSetStream(plan.get(), stream.get()));
        if constexpr (std::is_same_v<T, float>) {
            NOA_THROW_IF(cufftExecC2C(plan.get(),
                                      reinterpret_cast<cufftComplex*>(input.get()),
                                      reinterpret_cast<cufftComplex*>(output.get()), sign));
        } else {
            NOA_THROW_IF(cufftExecZ2Z(plan.get(),
                                      reinterpret_cast<cufftDoubleComplex*>(input.get()),
                                      reinterpret_cast<cufftDoubleComplex*>(output.get()), sign));
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
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p output can be equal to \p input. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void r2c(const shared_t<T[]>& input, size4_t input_stride,
                    const shared_t<Complex<T>[]>& output, size4_t output_stride,
                    size4_t shape, Norm norm, Stream& stream) {
        const Plan<T> plan{fft::R2C, input_stride, output_stride, shape, stream.device()};
        r2c(input, output, plan, stream);
        details::normalize<true>(output, output_stride, shape, Sign::FORWARD, norm, stream);
    }

    /// Computes the forward R2C transform.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Real space array(s).
    /// \param[out] output      On the \b device. Non-redundant non-centered FFT.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p output can be equal to \p input. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void r2c(const shared_t<T[]>& input,
                    const shared_t<Complex<T>[]>& output,
                    size4_t shape, Norm norm, Stream& stream) {
        Plan<T> plan(fft::R2C, shape, stream.device());
        r2c(input, output, plan, stream);
        details::normalize<true>(output, shape.fft().stride(), shape, Sign::FORWARD, norm, stream);
    }

    /// Computes the in-place R2C transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host. Input should be the real space array with appropriate padding.
    /// \param shape            Rightmost shape of \p data.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void r2c(const shared_t<T[]>& data, size4_t shape, Norm norm, Stream& stream) {
        r2c(data, std::reinterpret_pointer_cast<Complex<T>[]>(data), shape, norm, stream);
    }

    /// Computes the in-place R2C transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host. Input should be the real space array with appropriate padding.
    /// \param stride           Rightmost strides, in real elements, of \p data.
    /// \param shape            Rightmost shape of \p data.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note Since the transform is in-place, it must be able to hold the complex non-redundant transform.
    ///       As such, the innermost dimension must have the appropriate padding. See fft::Plan for more details
    template<typename T>
    NOA_IH void r2c(const shared_t<T[]>& data, size4_t stride, size4_t shape, Norm norm, Stream& stream) {
        // Since it is in-place, the pitch (in real elements) in the innermost dimension should be:
        //  1: even, since complex elements take 2 real elements
        //  2: have at least 1 (if odd) or 2 (if even) extract real element in the innermost dimension
        NOA_ASSERT(!(stride.pitch()[2] % 2));
        NOA_ASSERT(stride.pitch()[2] >= shape[3] + 1 + size_t(!(shape[3] % 2)));

        const size4_t complex_stride{stride[0] / 2, stride[1] / 2, stride[2] / 2, stride[3]};
        r2c(data, stride, std::reinterpret_pointer_cast<Complex<T>[]>(data), complex_stride, shape, norm, stream);
    }

    /// Computes the backward C2R transform.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Non-redundant non-centered FFT(s).
    /// \param input_stride     Rightmost stride, in elements, of \p input.
    /// \param[out] output      On the \b device. Real space array(s).
    /// \param output_stride    Rightmost stride, in elements, of \p output.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p output can be equal to \p input. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void c2r(const shared_t<Complex<T>[]>& input, size4_t input_stride,
                    const shared_t<T[]>& output, size4_t output_stride,
                    size4_t shape, Norm norm, Stream& stream) {
        const Plan<T> plan{fft::C2R, input_stride, output_stride, shape, stream.device()};
        c2r(input, output, plan, stream);
        details::normalize<false>(output, output_stride, shape, Sign::BACKWARD, norm, stream);
    }

    /// Computes the backward C2R transform.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device. Non-redundant non-centered FFT(s).
    /// \param[out] output      On the \b device. Real space array(s).
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p output can be equal to \p input. See \c fft::Plan<float> for more details.
    /// \note This function is asynchronous relative to the host and may return before completion.
    template<typename T>
    NOA_IH void c2r(const shared_t<Complex<T>[]>& input,
                    const shared_t<T[]>& output,
                    size4_t shape, Norm norm, Stream& stream) {
        const Plan<T> plan{fft::C2R, shape, stream.device()};
        c2r(input, output, plan, stream);
        details::normalize<false>(output, shape.stride(), shape, Sign::BACKWARD, norm, stream);
    }

    /// Computes the in-place C2R transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host. Input should be the non-redundant non-centered FFT(s).
    /// \param shape            Rightmost shape of \p data.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note \note This function is asynchronous relative to the host and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2r(const shared_t<Complex<T>[]>& data, size4_t shape, Norm norm, Stream& stream) {
        c2r(data, std::reinterpret_pointer_cast<T[]>(data), shape, norm, stream);
    }

    /// Computes the in-place C2R transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host. Input should be the non-redundant non-centered FFT(s).
    /// \param stride           Rightmost strides, in complex elements, of \p data.
    /// \param shape            Rightmost shape of \p data.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note \note This function is asynchronous relative to the host and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2r(const shared_t<Complex<T>[]>& data, size4_t stride, size4_t shape, Norm norm, Stream& stream) {
        const size4_t real_stride{stride[0] * 2, stride[1] * 2, stride[2] * 2, stride[3]};
        c2r(data, stride, std::reinterpret_pointer_cast<T[]>(data), real_stride, shape, norm, stream);
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
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p output can be equal to \p input.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2c(const shared_t<Complex<T>[]>& input, size4_t input_stride,
                    const shared_t<Complex<T>[]>& output, size4_t output_stride,
                    size4_t shape, Sign sign, Norm norm, Stream& stream) {
        const Plan<T> fast_plan{fft::C2C, input_stride, output_stride, shape, stream.device()};
        c2c(input, output, sign, fast_plan, stream);
        details::normalize<false>(output, output_stride, shape, sign, norm, stream);
    }

    /// Computes the C2C transform.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b device.
    /// \param[out] output      On the \b device.
    /// \param shape            Rightmost shape of \p input and \p output.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    ///
    /// \note \p output can be equal to \p input.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2c(const shared_t<Complex<T>[]>& input,
                    const shared_t<Complex<T>[]>& output,
                    size4_t shape, Sign sign, Norm norm, Stream& stream) {
        const Plan<T> fast_plan{fft::C2C, shape, stream.device()};
        c2c(input, output, sign, fast_plan, stream);
        details::normalize<false>(output, shape.stride(), shape, sign, norm, stream);
    }

    /// Computes the in-place C2C transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host.
    /// \param shape            Rightmost shape of \p data.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2c(const shared_t<Complex<T>[]>& data, size4_t shape, Sign sign, Norm norm, Stream& stream) {
        c2c(data, data, shape, sign, norm, stream);
    }

    /// Computes the in-place C2C transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host.
    /// \param stride           Rightmost strides, in complex elements, of \p data.
    /// \param shape            Rightmost shape of \p data.
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note This function is asynchronous relative to the host and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    NOA_IH void c2c(const shared_t<Complex<T>[]>& data, size4_t stride, size4_t shape,
                    Sign sign, Norm norm, Stream& stream) {
        c2c(data, stride, data, stride, shape, sign, norm, stream);
    }
}
