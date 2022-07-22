/// \file noa/cpu/fft/Transforms.h
/// \brief Fast Fourier Transforms.
/// \author Thomas - ffyr2w
/// \date 18 Jun 2021

#pragma once

#include <fftw3.h>

#include "noa/common/Definitions.h"
#include "noa/common/Types.h"
#include "noa/cpu/Stream.h"
#include "noa/cpu/fft/Plan.h"
#include "noa/cpu/math/Ewise.h"

namespace noa::cpu::fft {
    using Norm = noa::fft::Norm;
}

namespace noa::cpu::fft::details {
    template<bool HALF, typename T>
    void normalize(const shared_t<T[]>& array, size4_t strides, size4_t shape, Sign sign, Norm norm, Stream& stream) {
        using real_t = noa::traits::value_type_t<T>;
        const size3_t shape_{shape[1], shape[2], shape[3]};
        const auto count = static_cast<real_t>(noa::math::prod(shape_));
        const auto scale = norm == fft::NORM_ORTHO ? noa::math::sqrt(count) : count;
        if (sign == Sign::FORWARD && (norm == fft::NORM_FORWARD || norm == fft::NORM_ORTHO)) {
            math::ewise(array, strides, 1 / scale, array, strides,
                        HALF ? shape.fft() : shape, noa::math::multiply_t{}, stream);
        } else if (sign == Sign::BACKWARD && (norm == fft::NORM_BACKWARD || norm == fft::NORM_ORTHO)) {
            math::ewise(array, strides, 1 / scale, array, strides,
                        shape, noa::math::multiply_t{}, stream);
        }
    }
}

// -- Execute -- //
namespace noa::cpu::fft {
    /// Executes the \p plan.
    /// \note It is safe to execute the same plan in parallel by multiple threads. However, since a given plan operates
    ///       by default on a fixed array, one needs to use one of the new-array functions so that different threads
    ///       compute the transform on different data.
    template<typename T>
    inline void execute(const Plan<T>& plan) {
        if constexpr (std::is_same_v<T, float>)
            fftwf_execute(plan.get());
        else
            fftw_execute(plan.get());
    }

    /// Executes the \p plan.
    /// \param[in,out] stream Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note It is safe to execute the same plan in parallel by multiple threads. However, since a given plan operates
    ///       by default on a fixed array, one needs to use one of the new-array functions so that different threads
    ///       compute the transform on different data.
    template<typename T>
    inline void execute(const Plan<T>& plan, Stream& stream) {
        const auto ptr = plan.get();
        stream.enqueue([=]() {
            if constexpr (std::is_same_v<T, float>)
                fftwf_execute(ptr);
            else
                fftw_execute(ptr);
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
    inline void r2c(const shared_t<T[]>& input,
                    const shared_t<Complex<T>[]>& output,
                    const Plan<T>& plan, Stream& stream) {
        const auto ptr = plan.get();
        stream.enqueue([=]() {
            if constexpr (std::is_same_v<T, float>)
                fftwf_execute_dft_r2c(ptr, input.get(), reinterpret_cast<fftwf_complex*>(output.get()));
            else
                fftw_execute_dft_r2c(ptr, input.get(), reinterpret_cast<fftw_complex*>(output.get()));
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
    inline void c2r(const shared_t<Complex<T>[]>& input,
                    const shared_t<T[]>& output,
                    const Plan<T>& plan, Stream& stream) {
        const auto ptr = plan.get();
        stream.enqueue([=]() {
            if constexpr (std::is_same_v<T, float>)
                fftwf_execute_dft_c2r(ptr, reinterpret_cast<fftwf_complex*>(input.get()), output.get());
            else
                fftw_execute_dft_c2r(ptr, reinterpret_cast<fftw_complex*>(input.get()), output.get());
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
    inline void c2c(const shared_t<Complex<T>[]>& input,
                    const shared_t<Complex<T>[]>& output,
                    const Plan<T>& plan, Stream& stream) {
        const auto ptr = plan.get();
        stream.enqueue([=]() {
            if constexpr (std::is_same_v<T, float>) {
                fftwf_execute_dft(ptr,
                                  reinterpret_cast<fftwf_complex*>(input.get()),
                                  reinterpret_cast<fftwf_complex*>(output.get()));
            } else {
                fftw_execute_dft(ptr,
                                 reinterpret_cast<fftw_complex*>(input.get()),
                                 reinterpret_cast<fftw_complex*>(output.get()));
            }
        });
    }
}

// -- "One time" transforms -- //
namespace noa::cpu::fft {
    /// Computes the forward R2C transform.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b host. Real space array.
    /// \param[out] output      On the \b host. Non-redundant non-centered FFT(s). Can be equal to \p input.
    /// \param shape            BDHW shape of \p input and \p output. Batched 2D/3D array(s) or column/row vector(s).
    /// \param flag             Any (combination) of the FFT flags.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    inline void r2c(const shared_t<T[]>& input,
                    const shared_t<Complex<T>[]>& output,
                    size4_t shape, uint flag, Norm norm, Stream& stream) {
        stream.enqueue([=]() mutable {
            const Plan fast_plan(input.get(), output.get(), shape, flag, stream.threads());
            execute(fast_plan);
            details::normalize<true>(output, shape.fft().strides(), shape, Sign::FORWARD, norm, stream);
        });
    }

    /// Computes the forward R2C transform.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b host. Real space array.
    /// \param input_strides    BDHW strides, in real elements, of \p input.
    /// \param[out] output      On the \b host. Non-redundant non-centered FFT(s). Can be equal to \p input.
    /// \param output_strides   BDHW strides, in complex elements, of \p output.
    /// \param shape            BDHW shape of \p input and \p output. Batched 2D/3D array(s) or column/row vector(s).
    /// \param flag             Any (combination) of the FFT flags.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    inline void r2c(const shared_t<T[]>& input, size4_t input_strides,
                    const shared_t<Complex<T>[]>& output, size4_t output_strides,
                    size4_t shape, uint flag, Norm norm, Stream& stream) {
        stream.enqueue([=]() mutable {
            const Plan fast_plan(input.get(), input_strides, output.get(), output_strides, shape, flag, stream.threads());
            execute(fast_plan);
            details::normalize<true>(output, output_strides, shape, Sign::FORWARD, norm, stream);
        });
    }

    /// Computes the in-place R2C transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host. Input should be the real space array with appropriate padding.
    /// \param shape            BDHW shape of \p data. Batched 2D/3D array(s) or column/row vector(s).
    /// \param flag             Any (combination) of the FFT flags.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    inline void r2c(const shared_t<T[]>& data, size4_t shape, uint flag, Norm norm, Stream& stream) {
        r2c(data, std::reinterpret_pointer_cast<Complex<T>[]>(data), shape, flag, norm, stream);
    }

    /// Computes the in-place R2C transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host. Input should be the real space array with appropriate padding.
    /// \param strides          BDHW strides, in real elements, of \p data.
    /// \param shape            BDHW shape of \p data. Batched 2D/3D array(s) or column/row vector(s).
    /// \param flag             Any (combination) of the FFT flags.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \note Since the transform is in-place, it must be able to hold the complex non-redundant transform.
    ///       As such, the rows must have the appropriate padding. See fft::Plan for more details
    template<typename T>
    inline void r2c(const shared_t<T[]>& data, size4_t strides, size4_t shape, uint flag, Norm norm, Stream& stream) {
        // Since it is in-place, the pitch (in real elements) of the rows:
        //  1: is even, since complex elements take 2 real elements
        //  2: has at least 1 (if odd) or 2 (if even) extract real element
        NOA_ASSERT(!(strides.pitches()[2] % 2));
        NOA_ASSERT(strides.pitches()[2] >= shape[3] + 1 + size_t(!(shape[3] % 2)));

        const size4_t complex_strides{strides[0] / 2, strides[1] / 2, strides[2] / 2, strides[3]};
        r2c(data, strides, std::reinterpret_pointer_cast<Complex<T>[]>(data),
            complex_strides, shape, flag, norm, stream);
    }

    /// Computes the backward C2R transform.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b host. Non-redundant non-centered FFT(s).
    /// \param[out] output      On the \b host. Real space array. Can be equal to \p input.
    /// \param shape            BDHW shape of \p input and \p output. Batched 2D/3D array(s) or column/row vector(s).
    /// \param flag             Any (combination) of the FFT flags.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    inline void c2r(const shared_t<Complex<T>[]>& input,
                    const shared_t<T[]>& output,
                    size4_t shape, uint flag, Norm norm, Stream& stream) {
        stream.enqueue([=]() mutable {
            const Plan fast_plan(input.get(), output.get(), shape, flag, stream.threads());
            execute(fast_plan);
            details::normalize<false>(output, shape.strides(), shape, Sign::BACKWARD, norm, stream);
        });
    }

    /// Computes the backward C2R transform.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b host. Non-redundant non-centered FFT(s).
    /// \param input_strides    BDHW strides, in complex elements, of \p input.
    /// \param[out] output      On the \b host. Real space array. Can be equal to \p input.
    /// \param output_strides   BDHW strides, in real elements, of \p output.
    /// \param shape            BDHW shape of \p input and \p output. Batched 2D/3D array(s) or column/row vector(s).
    /// \param flag             Any (combination) of the FFT flags.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    inline void c2r(const shared_t<Complex<T>[]>& input, size4_t input_strides,
                    const shared_t<T[]>& output, size4_t output_strides,
                    size4_t shape, uint flag, Norm norm, Stream& stream) {
        stream.enqueue([=]() mutable {
            const Plan fast_plan(input.get(), input_strides,
                                 output.get(), output_strides,
                                 shape, flag, stream.threads());
            execute(fast_plan);
            details::normalize<false>(output, output_strides, shape, Sign::BACKWARD, norm, stream);
        });
    }

    /// Computes the in-place C2R transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host. Input should be the non-redundant non-centered FFT(s).
    /// \param shape            BDHW shape of \p data. Batched 2D/3D array(s) or column/row vector(s).
    /// \param flag             Any (combination) of the FFT flags.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    inline void c2r(const shared_t<Complex<T>[]>& data, size4_t shape,
                    uint flag, Norm norm, Stream& stream) {
        c2r(data, std::reinterpret_pointer_cast<T[]>(data), shape, flag, norm, stream);
    }

    /// Computes the in-place C2R transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host. Input should be the non-redundant non-centered FFT(s).
    /// \param strides          BDHW strides, in complex elements, of \p data.
    /// \param shape            BDHW shape of \p data. Batched 2D/3D array(s) or column/row vector(s).
    /// \param flag             Any (combination) of the FFT flags.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    inline void c2r(const shared_t<Complex<T>[]>& data, size4_t strides, size4_t shape,
                    uint flag, Norm norm, Stream& stream) {
        const size4_t real_strides{strides[0] * 2, strides[1] * 2, strides[2] * 2, strides[3]};
        c2r(data, strides, std::reinterpret_pointer_cast<T[]>(data), real_strides, shape, flag, norm, stream);
    }

    /// Computes the C2C transform.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b host. Input complex data.
    /// \param[out] output      On the \b host. Non-centered FFT(s). Can be equal to \p input.
    /// \param shape            BDHW shape of \p input and \p output. Batched 2D/3D array(s) or column/row vector(s).
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param flag             Any (combination) of the FFT flags.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    inline void c2c(const shared_t<Complex<T>[]>& input,
                    const shared_t<Complex<T>[]>& output,
                    size4_t shape, Sign sign, uint flag, Norm norm, Stream& stream) {
        stream.enqueue([=]() mutable {
            const Plan fast_plan(input.get(), output.get(), shape, sign, flag, stream.threads());
            execute(fast_plan);
            details::normalize<false>(output, shape.strides(), shape, sign, norm, stream);
        });
    }

    /// Computes the C2C transform.
    /// \tparam T               float, double.
    /// \param[in] input        On the \b host. Input complex data.
    /// \param input_strides    BDHW strides, in complex elements, of \p input.
    /// \param[out] output      On the \b host. Non-centered FFT(s). Can be equal to \p input.
    /// \param output_strides   BDHW strides, in complex elements, of \p output.
    /// \param shape            BDHW shape of \p input and \p output. Batched 2D/3D array(s) or column/row vector(s).
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param flag             Any (combination) of the FFT flags.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    inline void c2c(const shared_t<Complex<T>[]>& input, size4_t input_strides,
                    const shared_t<Complex<T>[]>& output, size4_t output_strides,
                    size4_t shape, Sign sign, uint flag, Norm norm, Stream& stream) {
        stream.enqueue([=]() mutable {
            const Plan fast_plan(input.get(), input_strides, output.get(), output_strides, shape,
                                 sign, flag, stream.threads());
            execute(fast_plan);
            details::normalize<false>(output, output_strides, shape, sign, norm, stream);
        });
    }

    /// Computes the in-place C2C transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host.
    /// \param shape            BDHW shape of \p data. Batched 2D/3D array(s) or column/row vector(s).
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param flag             Any (combination) of the FFT flags.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    inline void c2c(const shared_t<Complex<T>[]>& data, size4_t shape,
                    Sign sign, uint flag, Norm norm, Stream& stream) {
        c2c(data, data, shape, sign, flag, norm, stream);
    }

    /// Computes the in-place C2C transform.
    /// \tparam T               float, double.
    /// \param[in] data         On the \b host.
    /// \param strides          BDHW strides, in complex elements, of \p data.
    /// \param shape            BDHW shape of \p data. Batched 2D/3D array(s) or column/row vector(s).
    /// \param sign             Sign of the exponent in the formula that defines the Fourier transform.
    ///                         It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param flag             Any (combination) of the FFT flags.
    /// \param norm             Normalization mode.
    /// \param[in,out] stream   Stream on which to enqueue this function.
    /// \note Depending on the stream, this function may be asynchronous and may return before completion.
    /// \see fft::Plan for more details.
    template<typename T>
    inline void c2c(const shared_t<Complex<T>[]>& data, size4_t strides, size4_t shape,
                    Sign sign, uint flag, Norm norm, Stream& stream) {
        c2c(data, strides, data, strides, shape, sign, flag, norm, stream);
    }
}
