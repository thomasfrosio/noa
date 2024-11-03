#pragma once

#include "noa/unified/Array.hpp"
#include "noa/unified/Factory.hpp"
#include "noa/unified/Ewise.hpp"

#include "noa/cpu/fft/Transforms.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/fft/Transforms.hpp"
#endif

namespace noa::fft {
    static constexpr Norm NORM_DEFAULT = Norm::FORWARD;

    struct FFTOptions {
        /// Normalization mode.
        Norm norm = NORM_DEFAULT;

        /// Whether this transform should be cached.
        bool cache_plan = true;
    };
}

namespace noa::fft::guts {
    template<typename T>
    void normalize(T&& array, const Shape4<i64>& shape, Sign sign, Norm norm) {
        using real_t = nt::mutable_value_type_twice_t<T>;
        const auto count = static_cast<real_t>(product(shape.pop_front()));
        const auto scale = norm == Norm::ORTHO ? sqrt(count) : count;
        if ((sign == Sign::FORWARD and (norm == Norm::FORWARD or norm == Norm::ORTHO)) or
            (sign == Sign::BACKWARD and (norm == Norm::BACKWARD or norm == Norm::ORTHO))) {
            ewise({}, std::forward<T>(array), Scale{1 / scale});
        }
    }
}

namespace noa::fft {
    /// Computes the forward r2c transform of (batched) 2d/3d array(s) or column/row vector(s).
    /// \param[in] input    Real space array.
    /// \param[out] output  Non-redundant non-centered, aka "h" layout, FFT(s).
    /// \note In-place transforms are allowed if the \p input is appropriately padded to account
    ///       for the extra one (if odd) or two (if even) real element along the width dimension.
    template<nt::varray_decay_of_almost_any<f32, f64> Input,
             nt::varray_decay_of_any<Complex<nt::mutable_value_type_t<Input>>> Output>
    void r2c(Input&& input, Output&& output, FFTOptions options = {}) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        const auto logical_shape = input.shape();
        check(vall(Equal{}, output.shape(), logical_shape.rfft()),
              "Given the real input with a shape of {}, the non-redundant shape of the complex output "
              "should be {}, but got {}", logical_shape, logical_shape.rfft(), output.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto n_threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, real = std::forward<Input>(input)] {
                constexpr auto flags = noa::cpu::fft::ESTIMATE | noa::cpu::fft::PRESERVE_INPUT;
                noa::cpu::fft::r2c(
                        real.get(), real.strides(),
                        output.get(), output.strides(),
                        real.shape(), flags, n_threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::fft::r2c(
                input.get(), input.strides(),
                output.get(), output.strides(),
                input.shape(), options.cache_plan,
                cuda_stream);
            cuda_stream.enqueue_attach(std::forward<Input>(input), output);
            #else
            panic_no_gpu_backend();
            #endif
        }
        guts::normalize(std::forward<Output>(output), logical_shape, Sign::FORWARD, options.norm);
    }

    /// Computes the forward r2c transform of (batched) 2d/3d array(s) or column/row vector(s).
    /// \param[in] input    Real space array.
    /// \return Non-redundant non-centered, aka "h" layout, FFT(s).
    template<nt::varray_decay_of_almost_any<f32, f64> Input>
    [[nodiscard]] auto r2c(Input&& input, FFTOptions options = {}) {
        using real_t = nt::mutable_value_type_twice_t<Input>;
        Array<Complex<real_t>> output(input.shape().rfft(), input.options());
        r2c(std::forward<Input>(input), output, options);
        return output;
    }

    /// Computes the backward c2r transform.
    /// \param[in,out] input    Non-redundant non-centered, aka "h" layout, FFT(s).
    /// \param[out] output      Real space array.
    /// \note In-place transforms are allowed if the \p output is appropriately padded to account
    ///       for the extra one (if odd) or two (if even) real element in the width dimension.
    /// \note For multidimensional c2r transforms, the input is not preserved.
    template<nt::varray_decay_of_almost_any<c32, c64> Input,
             nt::varray_decay_of_any<nt::mutable_value_type_twice_t<Input>> Output>
    void c2r(Input&& input, Output&& output, FFTOptions options = {}) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        const auto logical_shape = output.shape();
        check(vall(Equal{}, input.shape(), logical_shape.rfft()),
              "Given the real output with a shape of {}, the non-redundant shape of the complex input "
              "should be {}, but got {}", logical_shape, logical_shape.rfft(), input.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, complex = std::forward<Input>(input)] {
                constexpr auto flags = noa::cpu::fft::ESTIMATE;
                noa::cpu::fft::c2r(
                    complex.get(), complex.strides(),
                    output.get(), output.strides(),
                    output.shape(), flags, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::fft::c2r(
                input.get(), input.strides(),
                output.get(), output.strides(),
                output.shape(), options.cache_plan, cuda_stream);
            cuda_stream.enqueue_attach(std::forward<Input>(input), output);
            #else
            panic_no_gpu_backend();
            #endif
        }
        guts::normalize(std::forward<Output>(output), logical_shape, Sign::BACKWARD, options.norm);
    }

    /// Computes the backward c2r transform.
    /// \param[in,out] input    Non-redundant non-centered, aka "h" layout, FFT(s).
    /// \param shape            BDHW logical shape of \p input.
    /// \return Real space array.
    /// \note For multidimensional c2r transforms, the input is not preserved.
    template<nt::varray_decay_of_almost_any<c32, c64> Input>
    [[nodiscard]] auto c2r(Input&& input, const Shape4<i64> shape, FFTOptions options = {}) {
        using real_t = nt::mutable_value_type_twice_t<Input>;
        Array<real_t> output(shape, input.options());
        c2r(std::forward<Input>(input), output, options);
        return output;
    }

    /// Computes the c2c transform.
    /// \param[in] input    Input complex data.
    /// \param[out] output  Non-centered, aka "f" layout, FFT(s).
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c Sign::FORWARD) or +1 (\c Sign::BACKWARD).
    /// \note In-place transforms are allowed.
    template<nt::varray_decay_of_almost_any<c32, c64> Input,
             nt::varray_decay_of_any<nt::mutable_value_type_t<Input>> Output>
    void c2c(Input&& input, Output&& output, Sign sign, FFTOptions options = {}) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        const auto logical_shape = input.shape();
        check(vall(Equal{}, logical_shape, output.shape()),
              "The input and output shape should match (no broadcasting allowed), but got input {} and output {}",
              logical_shape, output.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=, i = std::forward<Input>(input)] {
                constexpr auto flags = noa::cpu::fft::ESTIMATE | noa::cpu::fft::PRESERVE_INPUT;
                noa::cpu::fft::c2c(
                    i.get(), i.strides(),
                    output.get(), output.strides(),
                    i.shape(), sign, flags, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::fft::c2c(
                input.get(), input.strides(),
                output.get(), output.strides(),
                input.shape(), sign, options.cache_plan, cuda_stream);
            cuda_stream.enqueue_attach(std::forward<Input>(input), output);
            #else
            panic_no_gpu_backend();
            #endif
        }
        guts::normalize(std::forward<Output>(output), logical_shape, sign, options.norm);
    }

    /// Computes the c2c transform.
    /// \param[in] input    Input complex data.
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c Sign::FORWARD) or +1 (\c Sign::BACKWARD).
    /// \return Non-centered, aka "f" layout, FFT(s).
    template<nt::varray_decay_of_almost_any<c32, c64> Input>
    [[nodiscard]] auto c2c(Input&& input, Sign sign, FFTOptions options = {}) {
        using complex_t = nt::mutable_value_type_t<Input>;
        Array<complex_t> output(input.shape(), input.options());
        c2c(std::forward<Input>(input), output, sign, options);
        return output;
    }
}
