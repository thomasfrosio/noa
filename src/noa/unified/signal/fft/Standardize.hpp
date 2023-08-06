#pragma once

#include "noa/cpu/signal/fft/Standardize.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/Standardize.hpp"
#endif

#include "noa/unified/Array.hpp"
#include "noa/unified/fft/Transform.hpp"

namespace noa::signal::fft {
    using Remap = noa::fft::Remap;
    using Norm = noa::fft::Norm;

    /// Standardizes (mean=0, stddev=1) a real-space signal, by modifying its Fourier coefficients.
    /// \tparam REMAP       Remapping operator. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[in] input    Input FFT.
    /// \param[out] output  Output FFT. Can be equal to \p input.
    ///                     The C2R transform of \p output has its mean set to 0 and its stddev set to 1.
    /// \param shape        BDHW logical shape of \p input and \p output.
    /// \param norm         Normalization mode of \p input.
    template<Remap REMAP, typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::is_varray_of_almost_any_v<Input, c32, c64> &&
             noa::traits::is_varray_of_any_v<Output, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output> &&
             (REMAP == Remap::H2H || REMAP == Remap::HC2HC || REMAP == Remap::F2F || REMAP == Remap::FC2FC)>>
    void standardize_ifft(const Input& input, const Output& output, const Shape4<i64>& shape,
                          Norm norm = noa::fft::NORM_DEFAULT) {
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        constexpr bool IS_FULL = REMAP == Remap::F2F || REMAP == Remap::FC2FC;
        const auto actual_shape = IS_FULL ? shape : shape.rfft();
        NOA_CHECK(noa::all(input.shape() == actual_shape) && noa::all(output.shape() == actual_shape),
                  "The input {} and output {} {}redundant FFTs don't match the expected shape {}",
                  input.shape(), output.shape(), IS_FULL ? "" : "non-", actual_shape);

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::signal::fft::standardize_ifft<REMAP>(
                        input.get(), input.strides(),
                        output.get(), output.strides(),
                        shape, norm, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::signal::fft::standardize_ifft<REMAP>(
                    input.get(), input.strides(),
                    output.get(), output.strides(),
                    shape, norm, stream.cuda());
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
