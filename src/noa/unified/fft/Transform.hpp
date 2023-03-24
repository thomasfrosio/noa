#pragma once

#include "noa/cpu/fft/Transforms.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/fft/Transforms.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::fft {
    static constexpr Norm NORM_DEFAULT = Norm::FORWARD;

    /// Computes the forward R2C transform of (batched) 2D/3D array(s) or column/row vector(s).
    /// \param[in] input    Real space array.
    /// \param[out] output  Non-redundant non-centered FFT(s).
    /// \param norm         Normalization mode.
    /// \note In-place transforms are allowed if the \p input is appropriately padded to account
    ///       for the extra one (if odd) or two (if even) real element along the width dimension.
    template<typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64> &&
             noa::traits::is_array_or_view_of_any_v<Output, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, noa::traits::value_type_t<Output>>>>
    void r2c(const Input& input, const Output& output, Norm norm = NORM_DEFAULT) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(noa::all(output.shape() == input.shape().fft()),
                  "Given the real input with a shape of {}, the non-redundant shape of the complex output "
                  "should be {}, but got {}", input.shape(), input.shape().fft(), output.shape());

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::fft::r2c(input.get(), input.strides(),
                              output.get(), output.strides(),
                              input.shape(), cpu::fft::ESTIMATE | cpu::fft::PRESERVE_INPUT,
                              norm, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::fft::r2c(input.get(), input.strides(),
                           output.get(), output.strides(),
                           input.shape(), norm, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Computes the forward R2C transform of (batched) 2D/3D array(s) or column/row vector(s).
    /// \param[in] input    Real space array.
    /// \param norm         Normalization mode.
    /// \return Non-redundant non-centered FFT(s).
    template<typename Input, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, f32, f64>>>
    [[nodiscard]] auto r2c(const Input& input, Norm norm = NORM_DEFAULT) {
        using real_t = typename Input::value_type;
        Array<Complex<real_t>> output(input.shape().fft(), input.options());
        r2c(input, output, norm);
        return output;
    }

    /// Computes the backward C2R transform.
    /// \param[in,out] input    Non-redundant non-centered FFT(s).
    /// \param[out] output      Real space array.
    /// \param norm             Normalization mode.
    /// \note In-place transforms are allowed if the \p output is appropriately padded to account
    ///       for the extra one (if odd) or two (if even) real element in the width dimension.
    /// \note For multidimensional C2R transforms, the input is not preserved.
    template<typename Input, typename Output, typename = std::enable_if_t<
            noa::traits::is_array_or_view_of_almost_any_v<Input, c32, c64> &&
            noa::traits::is_array_or_view_of_any_v<Output, f32, f64> &&
            noa::traits::are_almost_same_value_type_v<noa::traits::value_type_t<Input>, Output>>>
    void c2r(const Input& input, const Output& output, Norm norm = NORM_DEFAULT) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(noa::all(input.shape() == output.shape().fft()),
                  "Given the real output with a shape of {}, the non-redundant shape of the complex input "
                  "should be {}, but got {}", output.shape(), output.shape().fft(), input.shape());

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::fft::c2r(input.get(), input.strides(),
                              output.get(), output.strides(),
                              output.shape(), cpu::fft::ESTIMATE,
                              norm, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::fft::c2r(input.get(), input.strides(),
                           output.get(), output.strides(),
                           output.shape(), norm, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Computes the backward C2R transform.
    /// \param[in,out] input    Non-redundant non-centered FFT(s).
    /// \param shape            BDHW logical shape of \p input.
    /// \param norm             Normalization mode.
    /// \return Real space array.
    /// \note For multidimensional C2R transforms, the input is not preserved.
    template<typename Input, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, c32, c64>>>
    [[nodiscard]] auto c2r(const Input& input, const Shape4<i64> shape, Norm norm = NORM_DEFAULT) {
        using real_t = noa::traits::value_type_t<typename Input::value_type>;
        Array<real_t> output(shape, input.options());
        c2r(input, output, norm);
        return output;
    }

    /// Computes the C2C transform.
    /// \param[in] input    Input complex data.
    /// \param[out] output  Non-centered FFT(s).
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c Sign::FORWARD) or +1 (\c Sign::BACKWARD).
    /// \param norm         Normalization mode.
    /// \note In-place transforms are allowed.
    template<typename Input, typename Output, typename = std::enable_if_t<
            noa::traits::is_array_or_view_of_almost_any_v<Input, c32, c64> &&
            noa::traits::is_array_or_view_of_any_v<Output, c32, c64> &&
            noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void c2c(const Input& input, const Output& output, Sign sign, Norm norm = NORM_DEFAULT) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(noa::all(input.shape() == output.shape()),
                  "The input and output shape should match (no broadcasting allowed), but got input {} and output {}",
                  input.shape(), output.shape());

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::fft::c2c(input.get(), input.strides(),
                              output.get(), output.strides(),
                              input.shape(), sign, cpu::fft::ESTIMATE | cpu::fft::PRESERVE_INPUT,
                              norm, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::fft::c2c(input.get(), input.strides(),
                           output.get(), output.strides(),
                           input.shape(), sign, norm, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Computes the C2C transform.
    /// \param[in] input    Input complex data.
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c Sign::FORWARD) or +1 (\c Sign::BACKWARD).
    /// \param norm         Normalization mode.
    /// \return Non-centered FFT(s).
    template<typename Input, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, c32, c64>>>
    [[nodiscard]] auto c2c(const Input& input, Sign sign, Norm norm = NORM_DEFAULT) {
        using complex_t = typename Input::value_type;
        Array<complex_t> output(input.shape(), input.options());
        c2c(input, output, sign, norm);
        return output;
    }
}
