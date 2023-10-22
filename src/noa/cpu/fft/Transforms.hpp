#pragma once

#include "noa/core/Types.hpp"
#include "noa/cpu/EwiseBinary.hpp"
#include "noa/cpu/fft/Plan.hpp"

#if defined(NOA_IS_OFFLINE)
#include <fftw3.h>

namespace noa::cpu::fft {
    using Norm = noa::fft::Norm;
    using Sign = noa::fft::Sign;
}

namespace noa::cpu::fft::guts {
    template<bool HALF, typename T>
    void normalize(
            T* array, const Strides4<i64>& strides, const Shape4<i64>& shape,
            Sign sign, Norm norm, i64 threads
    ) {
        using real_t = nt::value_type_t<T>;
        const auto count = static_cast<real_t>(noa::product(shape.pop_front()));
        const auto scale = norm == Norm::ORTHO ? noa::sqrt(count) : count;

        if (sign == Sign::FORWARD and (norm == Norm::FORWARD or norm == Norm::ORTHO)) {
            noa::cpu::ewise_binary(
                    array, strides, 1 / scale, array, strides,
                    HALF ? shape.rfft() : shape, noa::multiply_t{}, threads);
        } else if (sign == Sign::BACKWARD and (norm == Norm::BACKWARD or norm == Norm::ORTHO)) {
            noa::cpu::ewise_binary(
                    array, strides, 1 / scale, array, strides,
                    shape, noa::multiply_t{}, threads);
        }
    }
}

// -- Execute -- //
namespace noa::cpu::fft {
    // Executes the plan.
    // It is safe to execute the same plan in parallel by multiple threads. However, since a given plan operates
    // by default on a fixed array, one needs to use one of the new-array functions so that different threads
    // compute the transform on different data.

    template<typename T>
    void execute(const Plan<T>& plan) {
        if constexpr (std::is_same_v<T, f32>)
            fftwf_execute(plan.get());
        else
            fftw_execute(plan.get());
    }
}

// -- New-array transforms -- //
namespace noa::cpu::fft {
    // These functions are thread-safe as long as "input" and "output" are only accessed by
    // one single thread. However, "plan" can be accessed by multiple threads concurrently.
    // The arrays used to create "plan" should be similar to "input" and "output".
    // The shape should be the same. The input and output arrays are the same (in-place) or different
    // (out-of-place) if the plan was originally created to be in-place or out-of-place, respectively.
    // The alignment should be the same as well.

    template<typename T>
    void r2c(T* input, Complex<T>* output, const Plan<T>& plan) {
        if constexpr (std::is_same_v<T, f32>)
            fftwf_execute_dft_r2c(plan.get(), input, reinterpret_cast<fftwf_complex*>(output));
        else
            fftw_execute_dft_r2c(plan.get(), input, reinterpret_cast<fftw_complex*>(output));
    }

    template<typename T>
    void c2r(Complex<T>* input, T* output, const Plan<T>& plan) {
        if constexpr (std::is_same_v<T, f32>)
            fftwf_execute_dft_c2r(plan.get(), reinterpret_cast<fftwf_complex*>(input), output);
        else
            fftw_execute_dft_c2r(plan.get(), reinterpret_cast<fftw_complex*>(input), output);
    }

    template<typename T>
    void c2c(Complex<T>* input,Complex<T>* output, const Plan<T>& plan) {
        if constexpr (std::is_same_v<T, f32>) {
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
    template<typename T>
    void r2c(T* input, Complex<T>* output, const Shape4<i64>& shape, u32 flag, Norm norm, i64 threads) {
        const Plan fast_plan(input, output, shape, flag, threads);
        execute(fast_plan);
        guts::normalize<true>(output, shape.rfft().strides(), shape, Sign::FORWARD, norm, threads);
    }

    template<typename T>
    void r2c(T* input, const Strides4<i64>& input_strides,
             Complex<T>* output, const Strides4<i64>& output_strides,
             const Shape4<i64>& shape, u32 flag, Norm norm, i64 threads) {
        const Plan fast_plan(input, input_strides, output, output_strides, shape, flag, threads);
        execute(fast_plan);
        guts::normalize<true>(output, output_strides, shape, Sign::FORWARD, norm, threads);
    }

    template<typename T>
    void r2c(T* data, const Shape4<i64>& shape, u32 flag, Norm norm, i64 threads) {
        r2c(data, reinterpret_cast<Complex<T>*>(data), shape, flag, norm, threads);
    }

    template<typename T>
    void r2c(T* data, const Strides4<i64>& strides, const Shape4<i64>& shape,
             u32 flag, Norm norm, i64 threads) {
        // Since it is in-place, the physical width (in real elements):
        //  1: is even, since complex elements take 2 real elements.
        //  2: has at least 1 (if odd) or 2 (if even) extract real element.
        NOA_ASSERT(!(strides.physical_shape()[2] % 2));
        NOA_ASSERT(strides.physical_shape()[2] >= shape[3] + 1 + static_cast<i64>(!(shape[3] % 2)));

        const auto complex_strides = Strides4<i64>{strides[0] / 2, strides[1] / 2, strides[2] / 2, strides[3]};
        r2c(data, strides, reinterpret_cast<Complex<T>*>(data), complex_strides, shape, flag, norm, threads);
    }

    template<typename T>
    void c2r(Complex<T>* input, T* output, const Shape4<i64>& shape, u32 flag, Norm norm, i64 threads) {
        const Plan fast_plan(input, output, shape, flag, threads);
        execute(fast_plan);
        guts::normalize<false>(output, shape.strides(), shape, Sign::BACKWARD, norm, threads);
    }

    template<typename T>
    void c2r(Complex<T>* input, const Strides4<i64>& input_strides,
             T* output, const Strides4<i64>& output_strides,
             const Shape4<i64>& shape, u32 flag, Norm norm, i64 threads) {
        const Plan fast_plan(input, input_strides, output, output_strides, shape, flag, threads);
        execute(fast_plan);
        guts::normalize<false>(output, output_strides, shape, Sign::BACKWARD, norm, threads);
    }

    template<typename T>
    void c2r(Complex<T>* data, const Shape4<i64>& shape, u32 flag, Norm norm, i64 threads) {
        c2r(data, reinterpret_cast<T*>(data), shape, flag, norm, threads);
    }

    template<typename T>
    void c2r(Complex<T>* data, const Strides4<i64>& strides, const Shape4<i64>& shape,
             u32 flag, Norm norm, i64 threads) {
        const auto real_strides = Strides4<i64>{strides[0] * 2, strides[1] * 2, strides[2] * 2, strides[3]};
        c2r(data, strides, reinterpret_cast<T*>(data), real_strides, shape, flag, norm, threads);
    }

    template<typename T>
    void c2c(Complex<T>* input, Complex<T>* output, const Shape4<i64>& shape,
             Sign sign, u32 flag, Norm norm, i64 threads) {
        const Plan fast_plan(input, output, shape, sign, flag, threads);
        execute(fast_plan);
        guts::normalize<false>(output, shape.strides(), shape, sign, norm, threads);
    }

    template<typename T>
    void c2c(Complex<T>* input, const Strides4<i64>& input_strides,
             Complex<T>* output, const Strides4<i64>& output_strides,
             const Shape4<i64>& shape, Sign sign, u32 flag, Norm norm, i64 threads) {
        const Plan fast_plan(input, input_strides, output, output_strides, shape, sign, flag, threads);
        execute(fast_plan);
        guts::normalize<false>(output, output_strides, shape, sign, norm, threads);
    }

    template<typename T>
    void c2c(Complex<T>* data, const Shape4<i64>& shape,
             Sign sign, u32 flag, Norm norm, i64 threads) {
        c2c(data, data, shape, sign, flag, norm, threads);
    }

    template<typename T>
    void c2c(Complex<T>* data, const Strides4<i64>& strides, const Shape4<i64>& shape,
             Sign sign, u32 flag, Norm norm, i64 threads) {
        c2c(data, strides, data, strides, shape, sign, flag, norm, threads);
    }
}
#endif
