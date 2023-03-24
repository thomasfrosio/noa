#pragma once

#include "noa/cpu/math/Complex.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/math/Complex.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::math {
    /// Extracts the real and imaginary part of complex numbers.
    /// \param[in] input    Complex array to decompose.
    /// \param[out] real    Real elements.
    /// \param[out] imag    Imaginary elements.
    template<typename Complex, typename Real, typename Imag, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_complex_v<Complex> &&
             noa::traits::are_array_or_view_of_real_v<Real, Imag> &&
             noa::traits::are_almost_same_value_type_v<noa::traits::value_type_t<Complex>, Real, Imag>>>
    void decompose(const Complex& input, const Real& real, const Imag& imag) {
        NOA_CHECK(!input.is_empty() && !real.is_empty() && !imag.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(real, imag), "The output arrays should not overlap");

        const auto& output_shape = real.shape();
        NOA_CHECK(all(output_shape == imag.shape()),
                  "The real and imaginary arrays should have the same shape, but got real:{} and imag:{}",
                  output_shape, imag.shape());

        auto input_strides = input.strides();
        if (!noa::indexing::broadcast(input.shape(), input_strides, output_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output_shape);
        }

        const Device device = real.device();
        NOA_CHECK(device == input.device() && device == imag.device(),
                  "The input and output arrays must be on the same device, but got input:{}, real:{} and imag:{}",
                  input.device(), device, imag.device());

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::math::decompose(input.get(), input_strides,
                                     real.get(), real.strides(),
                                     imag.get(), imag.strides(),
                                     output_shape, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::decompose(input.get(), input_strides,
                                  real.get(), real.strides(),
                                  imag.get(), imag.strides(),
                                  output_shape, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), real.share(), imag.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Extracts the real part of complex numbers.
    /// \param[in] input    Complex array to decompose.
    /// \param[out] real    Real elements.
    template<typename Complex, typename Real, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_complex_v<Complex> &&
             noa::traits::is_array_or_view_of_real_v<Real> &&
             noa::traits::are_almost_same_value_type_v<noa::traits::value_type_t<Complex>, Real>>>
    void real(const Complex& input, const Real& real) {
        NOA_CHECK(!input.is_empty() && !real.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(real, input), "The arrays should not overlap");

        auto input_strides = input.strides();
        if (!noa::indexing::broadcast(input.shape(), input_strides, real.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), real.shape());
        }

        const Device device = real.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and real:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::math::real(input.get(), input_strides,
                                real.get(), real.strides(),
                                real.shape(), threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::real(input.get(), input_strides,
                             real.get(), real.strides(),
                             real.shape(), cuda_stream);
            cuda_stream.enqueue_attach(input.share(), real.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Extracts the imaginary part of complex numbers.
    /// \param[in] input    Complex array to decompose.
    /// \param[out] imag    Imaginary elements.
    template<typename Complex, typename Imag, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_complex_v<Complex> &&
             noa::traits::is_array_or_view_of_real_v<Imag> &&
             noa::traits::are_almost_same_value_type_v<noa::traits::value_type_t<Complex>, Imag>>>
    void imag(const Complex& input, const Imag& imag) {
        NOA_CHECK(!input.is_empty() && !imag.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(imag, input), "The arrays should not overlap");

        auto input_strides = input.strides();
        if (!noa::indexing::broadcast(input.shape(), input_strides, imag.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), imag.shape());
        }

        const Device device = imag.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{} and imag:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::math::imag(input.get(), input_strides,
                                imag.get(), imag.strides(),
                                imag.shape(), threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::imag(input.get(), input_strides,
                             imag.get(), imag.strides(),
                             imag.shape(), cuda_stream);
            cuda_stream.enqueue_attach(input.share(), imag.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Fuses the real and imaginary components.
    /// \param[in] real Real elements to interleave.
    /// \param[in] imag Imaginary elements to interleave.
    /// \param output   Complex array.
    template<typename Complex, typename Real, typename Imag, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_complex_v<Complex> &&
             noa::traits::are_array_or_view_of_real_v<Real, Imag> &&
             noa::traits::are_almost_same_value_type_v<noa::traits::value_type_t<Complex>, Real, Imag>>>
    void complex(const Real& real, const Imag& imag, const Complex& output) {
        NOA_CHECK(!output.is_empty() && !real.is_empty() && !imag.is_empty(), "Empty array detected");

        const auto& output_shape = output.shape();
        auto real_strides = real.strides();
        if (!noa::indexing::broadcast(real.shape(), real_strides, output_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      real.shape(), output_shape);
        }
        auto imag_strides = imag.strides();
        if (!noa::indexing::broadcast(imag.shape(), imag_strides, output_shape)) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      imag.shape(), output_shape);
        }

        const Device device = output.device();
        NOA_CHECK(device == real.device() && device == imag.device(),
                  "The input and output arrays must be on the same device, but got real:{}, imag:{} and output:{}",
                  real.device(), imag.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::math::complex(real.get(), real_strides,
                                   imag.get(), imag_strides,
                                   output.get(), output.strides(),
                                   output_shape, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::math::complex(real.get(), real_strides,
                                imag.get(), imag_strides,
                                output.get(), output.strides(),
                                output_shape, cuda_stream);
            cuda_stream.enqueue_attach(output.share(), real.share(), imag.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}

namespace noa::math {
    /// Extracts the real and imaginary part of complex numbers.
    template<typename Complex, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_complex_v<Complex>>>
    [[nodiscard]] auto decompose(const Complex& input) {
        using complex_t = noa::traits::value_type_t<Complex>;
        using real_t = noa::traits::value_type_t<complex_t>;
        Array<real_t> real(input.shape(), input.options());
        Array<real_t> imag(input.shape(), input.options());
        decompose(input, real, imag);
        return std::pair{real, imag};
    }

    /// Extracts the real part of complex numbers.
    template<typename Complex, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_complex_v<Complex>>>
    [[nodiscard]] auto real(const Complex& input) {
        using complex_t = noa::traits::value_type_t<Complex>;
        using real_t = noa::traits::value_type_t<complex_t>;
        Array<real_t> output(input.shape(), input.options());
        real(input, output);
        return output;
    }

    /// Extracts the imaginary part of complex numbers.
    template<typename Complex, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_complex_v<Complex>>>
    [[nodiscard]] auto imag(const Complex& input) {
        using complex_t = noa::traits::value_type_t<Complex>;
        using real_t = noa::traits::value_type_t<complex_t>;
        Array<real_t> output(input.shape(), input.options());
        imag(input, output);
        return output;
    }

    /// Fuses the real and imaginary components.
    template<typename Real, typename Imag, typename = std::enable_if_t<
             noa::traits::are_array_or_view_of_real_v<Real, Imag> &&
             noa::traits::are_almost_same_value_type_v<Real, Imag>>>
    [[nodiscard]] auto complex(const Real& real, const Imag& imag) {
        using real_t = noa::traits::mutable_value_type_t<Real>;
        using complex_t = Complex<real_t>;
        Array<complex_t> output(real.shape(), real.options());
        complex(real, imag, output);
        return output;
    }
}
