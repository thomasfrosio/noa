#pragma once

#include "noa/cpu/fft/Resize.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/fft/Resize.h"
#endif

#include "noa/unified/Array.h"

namespace noa::fft {
    /// Crops or zero-pads an FFT.
    /// \tparam REMAP       FFT Remap. Only H2H and F2F are currently supported.
    /// \tparam T           half_t, float, double, chalf_t, cfloat_t, cdouble_t.
    /// \param[in] input    FFT to resize.
    /// \param input_shape  Rightmost logical shape of \p input.
    /// \param[out] output  Resized FFT.
    /// \param output_shape Rightmost logical shape of \p output.
    ///                     All dimensions should either be <= or >= than \p input_shape.
    /// \note The outermost dimension cannot be resized, i.e. \p input_shape[0] == \p output_shape[0].
    template<Remap REMAP, typename T>
    void resize(const Array<T>& input, size4_t input_shape, const Array<T>& output, size4_t output_shape) {
        NOA_CHECK(all(input.shape() == input_shape.fft()),
                  "The non-redundant FFT with a shape of [logical:{}, pitch:{}] is expected, but got pitch of {}",
                  input_shape, input_shape.fft(), input.shape());
        NOA_CHECK(all(output.shape() == output_shape.fft()),
                  "The non-redundant FFT with a shape of [logical:{}, pitch:{}] is expected, but got pitch of {}",
                  output_shape, output_shape.fft(), output.shape());

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::fft::resize<REMAP>(input.share(), input.stride(), input_shape,
                                    output.share(), output.stride(), output_shape, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::fft::resize<REMAP>(input.share(), input.stride(), input_shape,
                                     output.share(), output.stride(), output_shape, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
