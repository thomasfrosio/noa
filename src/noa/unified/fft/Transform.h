#pragma once

#include "noa/cpu/fft/Transforms.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/fft/Transforms.h"
#endif

#include "noa/unified/Array.h"

namespace noa::fft {
    /// Computes the forward R2C transform.
    /// \tparam T           float, double.
    /// \param[in] input    Real space array.
    /// \param[out] output  Non-redundant non-centered FFT(s).
    /// \param norm         Normalization mode.
    /// \note In-place transforms are allowed if the \p input is appropriately padded to account
    ///       for the extra one or two real element in the innermost dimension. See cpu::fft::Plan for more details.
    template<typename T>
    void r2c(const Array<T>& input, const Array<Complex<T>>& output, Norm norm = NORM_FORWARD) {
        NOA_CHECK(all(output.shape() == input.shape().fft()),
                  "Given the real input with a shape of {}, the non-redundant shape of the complex output "
                  "should be {}, but got {}", input.shape(), input.shape().fft(), output.shape());

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::fft::r2c(input.share(), input.stride(),
                          output.share(), output.stride(),
                          input.shape(), cpu::fft::ESTIMATE | cpu::fft::PRESERVE_INPUT,
                          norm, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::fft::r2c(input.share(), input.stride(),
                           output.share(), output.stride(),
                           input.shape(), norm, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Computes the backward C2R transform.
    /// \tparam T               float, double.
    /// \param[in,out] input    Non-redundant non-centered FFT(s).
    /// \param[out] output      Real space array.
    /// \param norm             Normalization mode.
    /// \note In-place transforms are allowed if the \p output is appropriately padded to account
    ///       for the extra one or two real element in the innermost dimension. See cpu::fft::Plan for more details.
    /// \note For multidimensional C2R transforms, the input is not preserved.
    template<typename T>
    void c2r(const Array<Complex<T>>& input, const Array<T>& output, Norm norm = NORM_FORWARD) {
        NOA_CHECK(all(input.shape() == output.shape().fft()),
                  "Given the real output with a shape of {}, the non-redundant shape of the complex input "
                  "should be {}, but got {}", output.shape(), output.shape().fft(), input.shape());

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::fft::c2r(input.share(), input.stride(),
                          output.share(), output.stride(),
                          input.shape(), cpu::fft::ESTIMATE | cpu::fft::PRESERVE_INPUT,
                          norm, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::fft::c2r(input.share(), input.stride(),
                           output.share(), output.stride(),
                           input.shape(), norm, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Computes the C2C transform.
    /// \tparam T           float, double.
    /// \param[in] input    Input complex data.
    /// \param[out] output  Non-centered FFT(s).
    /// \param sign         Sign of the exponent in the formula that defines the Fourier transform.
    ///                     It can be −1 (\c fft::FORWARD) or +1 (\c fft::BACKWARD).
    /// \param norm         Normalization mode.
    /// \note In-place transforms are allowed.
    template<typename T>
    void c2c(const Array<Complex<T>>& input, const Array<Complex<T>>& output, Sign sign, Norm norm = NORM_FORWARD) {
        NOA_CHECK(all(input.shape() == input.shape()),
                  "The input and output shape should match (no broadcasting allowed), but got input {} and output {}",
                  input.shape(), output.shape());

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, but got input:{}, output:{}",
                  input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::fft::c2c(input.share(), input.stride(),
                          output.share(), output.stride(),
                          input.shape(), sign, cpu::fft::ESTIMATE | cpu::fft::PRESERVE_INPUT,
                          norm, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::fft::c2c(input.share(), input.stride(),
                           output.share(), output.stride(),
                           input.shape(), sign, norm, stream.cuda());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
