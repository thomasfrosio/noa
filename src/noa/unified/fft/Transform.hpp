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
    void normalize(const T& array, const Shape4<i64>& shape, Sign sign, Norm norm) {
        using real_t = nt::value_type_t<T>;
        const auto count = static_cast<real_t>(product(shape.pop_front()));
        const auto scale = norm == Norm::ORTHO ? sqrt(count) : count;
        if ((sign == Sign::FORWARD and (norm == Norm::FORWARD or norm == Norm::ORTHO)) or
            (sign == Sign::BACKWARD and (norm == Norm::BACKWARD or norm == Norm::ORTHO))) {
            ewise(wrap(array, 1 / scale), array, Multiply{});
        }
    }
}

namespace noa::fft {
    /// Computes the forward R2C transform of (batched) 2d/3d array(s) or column/row vector(s).
    /// \param[in] input    Real space array.
    /// \param[out] output  Non-redundant non-centered FFT(s).
    /// \note In-place transforms are allowed if the \p input is appropriately padded to account
    ///       for the extra one (if odd) or two (if even) real element along the width dimension.
    template<typename Input, typename Output>
    requires (nt::is_varray_of_almost_any_v<Input, f32, f64> and
              nt::is_varray_of_any_v<Output, c32, c64>)
    void r2c(const Input& input, const Output& output, FFTOptions options) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(all(output.shape() == input.shape().rfft()),
              "Given the real input with a shape of {}, the non-redundant shape of the complex output "
              "should be {}, but got {}", input.shape(), input.shape().rfft(), output.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:{}, output:{}",
              input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                constexpr auto flags = noa::cpu::fft::ESTIMATE | noa::cpu::fft::PRESERVE_INPUT;
                noa::cpu::fft::r2c(
                        input.get(), input.strides(),
                        output.get(), output.strides(),
                        input.shape(), flags, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::fft::r2c(
                    input.get(), input.strides(),
                    output.get(), output.strides(),
                    input.shape(), options.cache_plan,
                    cuda_stream);
            cuda_stream.enqueue_attach(input, output);
            #else
            panic("No GPU backend detected");
            #endif
        }
        guts::normalize(output, output.shape(), Sign::FORWARD, options.norm);
    }

    /// Computes the forward R2C transform of (batched) 2D/3D array(s) or column/row vector(s).
    /// \param[in] input    Real space array.
    /// \return Non-redundant non-centered FFT(s).
    template<typename Input> requires nt::is_varray_of_almost_any_v<Input, f32, f64>
    [[nodiscard]] auto r2c(const Input& real, FFTOptions options) {
        using real_t = nt::mutable_value_type_twice_t<Input>;
        Array<Complex<real_t>> output(real.shape().rfft(), real.options());
        r2c(real, output, options);
        return output;
    }

    /// Computes the backward C2R transform.
    /// \param[in,out] input    Non-redundant non-centered FFT(s).
    /// \param[out] output      Real space array.
    /// \note In-place transforms are allowed if the \p output is appropriately padded to account
    ///       for the extra one (if odd) or two (if even) real element in the width dimension.
    /// \note For multidimensional C2R transforms, the input is not preserved.
    template<typename Input, typename Output>
    requires (nt::is_varray_of_almost_any_v<Input, c32, c64> and
              nt::is_varray_of_any_v<Output, f32, f64>)
    void c2r(const Input& input, const Output& output, FFTOptions options) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(all(input.shape() == output.shape().rfft()),
              "Given the real output with a shape of {}, the non-redundant shape of the complex input "
              "should be {}, but got {}", output.shape(), output.shape().rfft(), input.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:{}, output:{}",
              input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                constexpr auto flags = cpu::fft::ESTIMATE;
                noa::cpu::fft::c2r(
                        input.get(), input.strides(),
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
            cuda_stream.enqueue_attach(input, output);
            #else
            panic("No GPU backend detected");
            #endif
        }
        guts::normalize(output, output.shape(), Sign::BACKWARD, options.norm);
    }

    /// Computes the backward C2R transform.
    /// \param[in,out] input    Non-redundant non-centered FFT(s).
    /// \param shape            BDHW logical shape of \p input.
    /// \return Real space array.
    /// \note For multidimensional C2R transforms, the input is not preserved.
    template<typename Input> requires nt::is_varray_of_almost_any_v<Input, c32, c64>
    [[nodiscard]] auto c2r(const Input& input, const Shape4<i64> shape, FFTOptions options) {
        using real_t = nt::mutable_value_type_twice_t<Input>;
        Array<real_t> output(shape, input.options());
        c2r(input, output, options);
        return output;
    }

    /// Computes the C2C transform.
    /// \param[in] input    Input complex data.
    /// \param[out] output  Non-centered FFT(s).
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c Sign::FORWARD) or +1 (\c Sign::BACKWARD).
    /// \note In-place transforms are allowed.
    template<typename Input, typename Output>
    requires (nt::is_varray_of_almost_any_v<Input, c32, c64> and
              nt::is_varray_of_any_v<Output, c32, c64>)
    void c2c(const Input& input, const Output& output, Sign sign, FFTOptions options) {
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(all(input.shape() == output.shape()),
              "The input and output shape should match (no broadcasting allowed), but got input {} and output {}",
              input.shape(), output.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:{}, output:{}",
              input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                constexpr auto flags = cpu::fft::ESTIMATE | cpu::fft::PRESERVE_INPUT;
                noa::cpu::fft::c2c(
                        input.get(), input.strides(),
                        output.get(), output.strides(),
                        input.shape(), sign, flags, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            noa::cuda::fft::c2c(
                    input.get(), input.strides(),
                    output.get(), output.strides(),
                    input.shape(), sign, options.cache_plan, cuda_stream);
            cuda_stream.enqueue_attach(input, output);
            #else
            panic("No GPU backend detected");
            #endif
        }
    }

    /// Computes the C2C transform.
    /// \param[in] input    Input complex data.
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c Sign::FORWARD) or +1 (\c Sign::BACKWARD).
    /// \return Non-centered FFT(s).
    template<typename Input> requires nt::is_varray_of_almost_any_v<Input, c32, c64>
    [[nodiscard]] auto c2c(const Input& input, Sign sign, FFTOptions options) {
        using complex_t = nt::mutable_value_type_t<Input>;
        Array<complex_t> output(input.shape(), input.options());
        c2c(input, output, sign, options);
        return output;
    }
}
