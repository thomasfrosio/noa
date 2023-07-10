#pragma once

#include "noa/cpu/fft/Resize.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/fft/Resize.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::fft::details {
    using Remap = ::noa::fft::Remap;
    template<Remap REMAP>
    constexpr bool is_valid_resize =
            REMAP == Remap::H2H || REMAP == Remap::F2F ||
            REMAP == Remap::HC2HC || REMAP == Remap::FC2FC;
}

// TODO Rescale values like in IMOD?

namespace noa::fft {
    /// Crops or zero-pads FFT(s).
    /// \tparam REMAP       FFT Remap. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[in] input    FFT to resize.
    /// \param input_shape  BDHW logical shape of \p input.
    /// \param[out] output  Resized FFT.
    /// \param output_shape BDHW logical shape of \p output.
    ///
    /// \note The batch dimension cannot be resized.
    /// \note If \p REMAP is H2H or F2C, this function can either crop or pad, but cannot do both.
    template<Remap REMAP, typename Input, typename Output, typename = std::enable_if_t<
             noa::traits::are_array_or_view_of_real_or_complex_v<Input, Output> &&
             noa::traits::are_almost_same_value_type_v<Input, Output> &&
             details::is_valid_resize<REMAP>>>
    void resize(const Input& input, const Shape4<i64>& input_shape,
                const Output& output, const Shape4<i64>& output_shape) {

        constexpr bool IS_FULL = noa::traits::to_underlying(REMAP) & Layout::SRC_FULL;
        NOA_CHECK(!input.is_empty() && !output.is_empty(), "Empty array detected");
        NOA_CHECK(!noa::indexing::are_overlapped(input, output), "Input and output arrays should not overlap");
        NOA_CHECK(input.shape()[0] == output.shape()[0], "The batch dimension cannot be resized");

        NOA_CHECK(noa::all(input.shape() == (IS_FULL ? input_shape : input_shape.rfft())),
                  "Given the {} remap, the input fft is expected to have a physical shape of {}, but got {}",
                  REMAP, IS_FULL ? input_shape : input_shape.rfft(), input.shape());
        NOA_CHECK(noa::all(output.shape() == (IS_FULL ? output_shape : output_shape.rfft())),
                  "Given the {} remap, the output fft is expected to have a physical shape of {}, but got {}",
                  REMAP, IS_FULL ? output_shape : output_shape.rfft(), output.shape());

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::fft::resize<REMAP>(
                        input.get(), input.strides(), input_shape,
                        output.get(), output.strides(), output_shape, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::fft::resize<REMAP>(
                    input.get(), input.strides(), input_shape,
                    output.get(), output.strides(), output_shape, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Returns a cropped or zero-padded FFT.
    /// \tparam REMAP       FFT Remap. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[in] input    FFT to resize.
    /// \param input_shape  BDHW logical shape of \p input.
    /// \param output_shape BDHW logical shape of the output.
    ///
    /// \note The batch dimension cannot be resized.
    /// \note If \p REMAP is H2H or F2C, this function can either crop or pad, but cannot do both.
    template<Remap REMAP, typename Input, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_real_or_complex_v<Input> &&
             details::is_valid_resize<REMAP>>>
    [[nodiscard]] auto resize(const Input& input,
                              const Shape4<i64>& input_shape,
                              const Shape4<i64>& output_shape) {
        using value_t = typename Input::value_type;
        Array<value_t> output(noa::traits::to_underlying(REMAP) & Layout::DST_FULL ?
                              output_shape : output_shape.rfft(),
                              input.options());
        fft::resize<REMAP>(input, input_shape, output, output_shape);
        return output;
    }
}
