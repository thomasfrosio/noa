#pragma once

#include <cufft.h>

#include "noa/gpu/cuda/Types.hpp"
#include "noa/gpu/cuda/Stream.hpp"
#include "noa/gpu/cuda/fft/Plan.hpp"
#include "noa/gpu/cuda/fft/Exception.hpp"
#include "noa/gpu/cuda/Ewise.hpp"

namespace noa::cuda::fft {
    using Norm = noa::fft::Norm;
    using Sign = noa::fft::Sign;
}

namespace noa::cuda::fft::details {
    template<typename Real>
    bool is_aligned_to_complex(Real* ptr) {
        // This is apparently not guaranteed to work by the standard.
        // But this should work in all modern and mainstream platforms.
        constexpr size_t ALIGNMENT = alignof(Complex<Real>);
        return !(reinterpret_cast<std::uintptr_t>(ptr) % ALIGNMENT);
    }

    template<bool HALF, typename T>
    void normalize(T* array, const Strides4<i64>& strides, const Shape4<i64>& shape,
                   Sign sign, Norm norm, Stream& stream) {
        using real_t = noa::traits::value_type_t<T>;
        const auto count = static_cast<real_t>(noa::math::product(shape.pop_front()));
        const auto scale = norm == Norm::ORTHO ? noa::math::sqrt(count) : count;
        if (sign == Sign::FORWARD && (norm == Norm::FORWARD || norm == Norm::ORTHO)) {
            ewise_binary(array, strides, 1 / scale, array, strides,
                         HALF ? shape.fft() : shape, noa::multiply_t{}, stream);
        } else if (sign == Sign::BACKWARD && (norm == Norm::BACKWARD || norm == Norm::ORTHO)) {
            ewise_binary(array, strides, 1 / scale, array, strides,
                         shape, noa::multiply_t{}, stream);
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

    template<typename Real>
    void r2c(Real* input, Complex<Real>* output, const Plan<Real>& plan, Stream& stream) {
        NOA_CHECK(details::is_aligned_to_complex(input),
                  "cufft requires both the input and output to be aligned to the complex type. This might not "
                  "be the case for the real input when operating on a subregion starting at an odd offset. "
                  "Hint: copy the real array to a new array or add adequate padding.");
        NOA_THROW_IF(cufftSetStream(plan.get(), stream.get()));
        if constexpr (std::is_same_v<Real, f32>)
            NOA_THROW_IF(cufftExecR2C(plan.get(), input, reinterpret_cast<cufftComplex*>(output)));
        else
            NOA_THROW_IF(cufftExecD2Z(plan.get(), input, reinterpret_cast<cufftDoubleComplex*>(output)));
        stream.enqueue_attach(plan.share());
    }

    template<typename Real>
    void c2r(Complex<Real>* input, Real* output, const Plan<Real>& plan, Stream& stream) {
        NOA_CHECK(details::is_aligned_to_complex(output),
                  "cufft requires both the input and output to be aligned to the complex type. This might not "
                  "be the case for the real output when operating on a subregion starting at an odd offset. "
                  "Hint: copy the real array to a new array or add adequate padding.");
        NOA_THROW_IF(cufftSetStream(plan.get(), stream.get()));
        if constexpr (std::is_same_v<Real, f32>)
            NOA_THROW_IF(cufftExecC2R(plan.get(), reinterpret_cast<cufftComplex*>(input), output));
        else
            NOA_THROW_IF(cufftExecZ2D(plan.get(), reinterpret_cast<cufftDoubleComplex*>(input), output));
        stream.enqueue_attach(plan.share());
    }

    template<typename Real>
    void c2c(Complex<Real>* input, Complex<Real>* output, Sign sign, const Plan<Real>& plan, Stream& stream) {
        NOA_THROW_IF(cufftSetStream(plan.get(), stream.get()));
        if constexpr (std::is_same_v<Real, f32>) {
            NOA_THROW_IF(cufftExecC2C(plan.get(),
                                      reinterpret_cast<cufftComplex*>(input),
                                      reinterpret_cast<cufftComplex*>(output),
                                      noa::traits::to_underlying(sign)));
        } else {
            NOA_THROW_IF(cufftExecZ2Z(plan.get(),
                                      reinterpret_cast<cufftDoubleComplex*>(input),
                                      reinterpret_cast<cufftDoubleComplex*>(output),
                                      noa::traits::to_underlying(sign)));
        }
        stream.enqueue_attach(plan.share());
    }
}

namespace noa::cuda::fft {
    template<typename T>
    void r2c(T* input, Complex<T>* output, const Shape4<i64>& shape, Norm norm, bool cache_plan, Stream& stream) {
        Plan<T> plan(fft::R2C, shape, stream.device(), cache_plan);
        r2c(input, output, plan, stream);
        details::normalize<true>(output, shape.fft().strides(), shape, Sign::FORWARD, norm, stream);
    }

    template<typename T>
    void r2c(T* input, const Strides4<i64>& input_strides,
             Complex<T>* output, const Strides4<i64>& output_strides,
             const Shape4<i64>& shape, Norm norm, bool cache_plan, Stream& stream) {
        const Plan<T> plan(fft::R2C, input_strides, output_strides, shape, stream.device(), cache_plan);
        r2c(input, output, plan, stream);
        details::normalize<true>(output, output_strides, shape, Sign::FORWARD, norm, stream);
    }

    template<typename T>
    void r2c(T* data, const Shape4<i64>& shape, Norm norm, bool cache_plan, Stream& stream) {
        r2c(data, reinterpret_cast<Complex<T>*>(data), shape, norm, cache_plan, stream);
    }

    template<typename T>
    void r2c(T* data, const Strides4<i64>& strides, const Shape4<i64>& shape, Norm norm,
             bool cache_plan, Stream& stream) {
        // Since it is in-place, the physical width (in real elements):
        //  1: is even, since complex elements take 2 real elements.
        //  2: has at least 1 (if odd) or 2 (if even) extract real element.
        NOA_ASSERT(!(strides.physical_shape()[2] % 2));
        NOA_ASSERT(strides.physical_shape()[2] >= shape[3] + 1 + static_cast<i64>(!(shape[3] % 2)));

        const auto complex_strides = Strides4<i64>{strides[0] / 2, strides[1] / 2, strides[2] / 2, strides[3]};
        r2c(data, strides, reinterpret_cast<Complex<T>*>(data), complex_strides, shape, norm, cache_plan, stream);
    }

    template<typename T>
    void c2r(Complex<T>* input, const Strides4<i64>& input_strides,
             T* output, const Strides4<i64>& output_strides,
             const Shape4<i64>& shape, Norm norm, bool cache_plan, Stream& stream) {
        const Plan<T> plan(fft::C2R, input_strides, output_strides, shape, stream.device(), cache_plan);
        c2r(input, output, plan, stream);
        details::normalize<false>(output, output_strides, shape, Sign::BACKWARD, norm, stream);
    }

    template<typename T>
    void c2r(Complex<T>* input, T* output, const Shape4<i64>& shape, Norm norm, bool cache_plan, Stream& stream) {
        const Plan<T> plan(fft::C2R, shape, stream.device(), cache_plan);
        c2r(input, output, plan, stream);
        details::normalize<false>(output, shape.strides(), shape, Sign::BACKWARD, norm, stream);
    }

    template<typename T>
    void c2r(Complex<T>* data, const Shape4<i64>& shape, Norm norm, bool cache_plan, Stream& stream) {
        c2r(data, reinterpret_cast<T*>(data), shape, norm, cache_plan, stream);
    }

    template<typename T>
    void c2r(Complex<T>* data, const Strides4<i64>& strides, const Shape4<i64>& shape,
             Norm norm, bool cache_plan, Stream& stream) {
        const auto real_strides = Strides4<i64>{strides[0] * 2, strides[1] * 2, strides[2] * 2, strides[3]};
        c2r(data, strides, reinterpret_cast<T*>(data), real_strides, shape, norm, cache_plan, stream);
    }

    template<typename T>
    void c2c(Complex<T>* input, const Strides4<i64>& input_strides,
             Complex<T>* output, const Strides4<i64>& output_strides,
             const Shape4<i64>& shape, Sign sign, Norm norm, bool cache_plan, Stream& stream) {
        const Plan<T> fast_plan(fft::C2C, input_strides, output_strides, shape, stream.device(), cache_plan);
        c2c(input, output, sign, fast_plan, stream);
        details::normalize<false>(output, output_strides, shape, sign, norm, stream);
    }

    template<typename T>
    void c2c(Complex<T>* input, Complex<T>* output, const Shape4<i64>& shape,
             Sign sign, Norm norm, bool cache_plan, Stream& stream) {
        const Plan<T> fast_plan(fft::C2C, shape, stream.device(), cache_plan);
        c2c(input, output, sign, fast_plan, stream);
        details::normalize<false>(output, shape.strides(), shape, sign, norm, stream);
    }

    template<typename T>
    void c2c(Complex<T>* data, const Shape4<i64>& shape, Sign sign, Norm norm, bool cache_plan, Stream& stream) {
        c2c(data, data, shape, sign, norm, cache_plan, stream);
    }

    template<typename T>
    void c2c(Complex<T>* data, const Strides4<i64>& strides, const Shape4<i64>& shape,
             Sign sign, Norm norm, bool cache_plan, Stream& stream) {
        c2c(data, strides, data, strides, shape, sign, norm, cache_plan, stream);
    }
}
