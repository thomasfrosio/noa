#pragma once

#include "noa/cpu/fft/Remap.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/fft/Remap.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::fft {
    /// Remaps FFT(s).
    /// \param remap        Remapping operation.
    /// \param[in] input    Input fft to remap.
    /// \param[out] output  Remapped fft.
    /// \param shape        BDHW logical shape.
    /// \note If \p remap is \c H2HC, \p input can be equal to \p output, iff the height and depth are even or 1.
    template<typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::are_array_or_view_of_real_or_complex_v<Input, Output> &&
             noa::traits::are_almost_same_value_type_v<Input, Output>>>
    void remap(Remap remap, const Input& input, const Output& output, const Shape4<i64>& shape) {
        const auto u8_remap = static_cast<u8>(remap);
        const bool is_src_full = u8_remap & Layout::SRC_FULL;
        const bool is_dst_full = u8_remap & Layout::DST_FULL;

        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(noa::all(input.shape() == (is_src_full ? shape : shape.rfft())),
                  "Given the {} remap, the input fft is expected to have a physical shape of {}, but got {}",
                  remap, is_src_full ? shape : shape.rfft(), input.shape());
        NOA_CHECK(noa::all(output.shape() == (is_dst_full ? shape : shape.rfft())),
                  "Given the {} remap, the output fft is expected to have a physical shape of {}, but got {}",
                  remap, is_dst_full ? shape : shape.rfft(), output.shape());

        NOA_CHECK(!noa::indexing::are_overlapped(input, output) ||
                  (remap == fft::H2HC && input.get() == output.get() &&
                   (shape[2] == 1 || !(shape[2] % 2)) &&
                   (shape[1] == 1 || !(shape[1] % 2))),
                  "In-place remapping is only available with {} and when the depth and height dimensions "
                  "have an even number of elements, but got remap {} and shape {}", fft::H2HC, remap, shape);

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::fft::remap(remap, input.get(), input.strides(),
                                output.get(), output.strides(),
                                shape, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::fft::remap(remap, input.get(), input.strides(),
                             output.get(), output.strides(),
                             shape, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Remaps fft(s).
    template<typename Input, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_real_or_complex_v<Input>>>
    [[nodiscard]] auto remap(Remap remap, const Input& input, const Shape4<i64>& shape) {
        const auto output_shape = noa::traits::to_underlying(remap) & Layout::DST_FULL ? shape : shape.rfft();
        using value_t = typename Input::value_type;
        Array<value_t> output(output_shape, input.options());
        fft::remap(remap, input, output, shape);
        return output;
    }
}
