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
    void normalize(const shared_t<T[]>& array, size4_t strides, size4_t shape, Sign sign, Norm norm, Stream& stream) {
        using real_t = noa::traits::value_type_t<T>;
        const size3_t shape_{shape[1], shape[2], shape[3]};
        const auto count = static_cast<real_t>(noa::math::prod(shape_));
        const auto scale = norm == Norm::NORM_ORTHO ? noa::math::sqrt(count) : count;
        if (sign == Sign::FORWARD && (norm == Norm::NORM_FORWARD || norm == Norm::NORM_ORTHO)) {
            math::ewise(array, strides, 1 / scale, array, strides,
                        HALF ? shape.fft() : shape, noa::math::multiply_t{}, stream);
        } else if (sign == Sign::BACKWARD && (norm == Norm::NORM_BACKWARD || norm == Norm::NORM_ORTHO)) {
            math::ewise(array, strides, 1 / scale, array, strides,
                        shape, noa::math::multiply_t{}, stream);
        }
    }
}

namespace noa::cuda::fft {
    using namespace ::noa::fft;

    // Plan creation (and the cuFFT APIs in general) is thread safe. However, plans and output data
    // should only be accessed by one (host) thread at a time.
    // The shape should be the same. The input and output arrays are the same (in-place) or different
    // (out-of-place) if the plan was originally created to be in-place or out-of-place, respectively.
    // The alignment should be the same as well.

    template<typename T>
    inline void r2c(const shared_t<T[]>& input,
                    const shared_t<Complex<T>[]>& output,
                    const Plan<T>& plan, Stream& stream) {
        NOA_THROW_IF(cufftSetStream(plan.get(), stream.get()));
        if constexpr (std::is_same_v<T, float>)
            NOA_THROW_IF(cufftExecR2C(plan.get(), input.get(), reinterpret_cast<cufftComplex*>(output.get())));
        else
            NOA_THROW_IF(cufftExecD2Z(plan.get(), input.get(), reinterpret_cast<cufftDoubleComplex*>(output.get())));
        stream.attach(input, output, plan.share());
    }

    template<typename T>
    inline void c2r(const shared_t<Complex<T>[]>& input,
                    const shared_t<T[]>& output,
                    const Plan<T>& plan, Stream& stream) {
        NOA_THROW_IF(cufftSetStream(plan.get(), stream.get()));
        if constexpr (std::is_same_v<T, float>)
            NOA_THROW_IF(cufftExecC2R(plan.get(), reinterpret_cast<cufftComplex*>(input.get()), output.get()));
        else
            NOA_THROW_IF(cufftExecZ2D(plan.get(), reinterpret_cast<cufftDoubleComplex*>(input.get()), output.get()));
        stream.attach(input, output, plan.share());
    }

    template<typename T>
    inline void c2c(const shared_t<Complex<T>[]>& input,
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
        stream.attach(input, output, plan.share());
    }
}

// -- "One time" transforms -- //
namespace noa::cuda::fft {
    template<typename T>
    inline void r2c(const shared_t<T[]>& input,
                    const shared_t<Complex<T>[]>& output,
                    size4_t shape, Norm norm, Stream& stream) {
        Plan<T> plan(fft::R2C, shape, stream.device());
        r2c(input, output, plan, stream);
        details::normalize<true>(output, shape.fft().strides(), shape, Sign::FORWARD, norm, stream);
    }

    template<typename T>
    inline void r2c(const shared_t<T[]>& input, size4_t input_strides,
                    const shared_t<Complex<T>[]>& output, size4_t output_strides,
                    size4_t shape, Norm norm, Stream& stream) {
        const Plan<T> plan(fft::R2C, input_strides, output_strides, shape, stream.device());
        r2c(input, output, plan, stream);
        details::normalize<true>(output, output_strides, shape, Sign::FORWARD, norm, stream);
    }

    template<typename T>
    inline void r2c(const shared_t<T[]>& data, size4_t shape, Norm norm, Stream& stream) {
        r2c(data, std::reinterpret_pointer_cast<Complex<T>[]>(data), shape, norm, stream);
    }

    template<typename T>
    inline void r2c(const shared_t<T[]>& data, size4_t strides, size4_t shape, Norm norm, Stream& stream) {
        // Since it is in-place, the pitch (in real elements) of the rows:
        //  1: is even, since complex elements take 2 real elements
        //  2: has at least 1 (if odd) or 2 (if even) extract real element
        NOA_ASSERT(!(strides.pitches()[2] % 2));
        NOA_ASSERT(strides.pitches()[2] >= shape[3] + 1 + size_t(!(shape[3] % 2)));

        const size4_t complex_strides{strides[0] / 2, strides[1] / 2, strides[2] / 2, strides[3]};
        r2c(data, strides, std::reinterpret_pointer_cast<Complex<T>[]>(data), complex_strides, shape, norm, stream);
    }

    template<typename T>
    inline void c2r(const shared_t<Complex<T>[]>& input, size4_t input_strides,
                    const shared_t<T[]>& output, size4_t output_strides,
                    size4_t shape, Norm norm, Stream& stream) {
        const Plan<T> plan(fft::C2R, input_strides, output_strides, shape, stream.device());
        c2r(input, output, plan, stream);
        details::normalize<false>(output, output_strides, shape, Sign::BACKWARD, norm, stream);
    }

    template<typename T>
    inline void c2r(const shared_t<Complex<T>[]>& input,
                    const shared_t<T[]>& output,
                    size4_t shape, Norm norm, Stream& stream) {
        const Plan<T> plan(fft::C2R, shape, stream.device());
        c2r(input, output, plan, stream);
        details::normalize<false>(output, shape.strides(), shape, Sign::BACKWARD, norm, stream);
    }

    template<typename T>
    inline void c2r(const shared_t<Complex<T>[]>& data, size4_t shape, Norm norm, Stream& stream) {
        c2r(data, std::reinterpret_pointer_cast<T[]>(data), shape, norm, stream);
    }

    template<typename T>
    inline void c2r(const shared_t<Complex<T>[]>& data, size4_t strides, size4_t shape, Norm norm, Stream& stream) {
        const size4_t real_strides{strides[0] * 2, strides[1] * 2, strides[2] * 2, strides[3]};
        c2r(data, strides, std::reinterpret_pointer_cast<T[]>(data), real_strides, shape, norm, stream);
    }

    template<typename T>
    inline void c2c(const shared_t<Complex<T>[]>& input, size4_t input_strides,
                    const shared_t<Complex<T>[]>& output, size4_t output_strides,
                    size4_t shape, Sign sign, Norm norm, Stream& stream) {
        const Plan<T> fast_plan(fft::C2C, input_strides, output_strides, shape, stream.device());
        c2c(input, output, sign, fast_plan, stream);
        details::normalize<false>(output, output_strides, shape, sign, norm, stream);
    }

    template<typename T>
    inline void c2c(const shared_t<Complex<T>[]>& input,
                    const shared_t<Complex<T>[]>& output,
                    size4_t shape, Sign sign, Norm norm, Stream& stream) {
        const Plan<T> fast_plan(fft::C2C, shape, stream.device());
        c2c(input, output, sign, fast_plan, stream);
        details::normalize<false>(output, shape.strides(), shape, sign, norm, stream);
    }

    template<typename T>
    inline void c2c(const shared_t<Complex<T>[]>& data, size4_t shape, Sign sign, Norm norm, Stream& stream) {
        c2c(data, data, shape, sign, norm, stream);
    }

    template<typename T>
    inline void c2c(const shared_t<Complex<T>[]>& data, size4_t strides, size4_t shape,
                    Sign sign, Norm norm, Stream& stream) {
        c2c(data, strides, data, strides, shape, sign, norm, stream);
    }
}
