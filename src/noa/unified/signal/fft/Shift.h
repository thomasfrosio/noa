#pragma once

#include "noa/cpu/signal/fft/Shift.h"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/Shift.h"
#endif

#include "noa/unified/Array.h"

namespace noa::signal::fft {
    using Remap = noa::fft::Remap;

    /// Phase-shifts a non-redundant 2D (batched) FFT.
    /// \tparam REMAP           Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \tparam T               cfloat_t or cdouble_t.
    /// \param[in] input        2D FFT to phase-shift. If empty, the phase-shifts are saved in \p output.
    /// \param[out] output      Non-redundant phase-shifted 2D FFT.
    /// \param[in] shifts       Rightmost 2D real-space forward shift to apply (as phase shift).
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are usually from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are not phase-shifted.
    /// \note \p input and \p output can be equal if no remapping is done, i.e. H2H or HC2HC.
    template<Remap REMAP, typename T>
    void shift2D(const Array<T>& input, const Array<T>& output,
                 const Array<float2_t>& shifts, float cutoff = 0.5f) {
        size4_t input_stride = input.stride();
        if (!input.empty() && !indexing::broadcast(input.shape(), input_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        if (REMAP == Remap::H2HC || REMAP == Remap::HC2H)
            NOA_CHECK(input.get() != output.get(), "In-place phase-shift is not supported with {} remap", REMAP);

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, "
                  "but got input:{} and output:{}", input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(shifts.dereferencable(), "The matrices should be accessible to the host");
            cpu::signal::fft::shift2D<REMAP>(
                    input.share(), input.stride(),
                    output.share(), output.stride(), output.shape(),
                    shifts.share(), cutoff, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::shift2D<REMAP>(
                    input.share(), input.stride(),
                    output.share(), output.stride(), output.shape(),
                    shifts.share(), cutoff, stream.cpu());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    ///  Phase-shifts a non-redundant 2D (batched) FFT.
    /// \see This function is has the same features and limitations than the overload above.
    template<Remap REMAP, typename T>
    void shift2D(const Array<T>& input, const Array<T>& output,
                 float2_t shift, float cutoff = 0.5f) {
        size4_t input_stride = input.stride();
        if (!input.empty() && !indexing::broadcast(input.shape(), input_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        if (REMAP == Remap::H2HC || REMAP == Remap::HC2H)
            NOA_CHECK(input.get() != output.get(), "In-place phase-shift is not supported with {} remap", REMAP);

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, "
                  "but got input:{} and output:{}", input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::fft::shift2D<REMAP>(
                    input.share(), input.stride(),
                    output.share(), output.stride(), output.shape(),
                    shift, cutoff, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::shift2D<REMAP>(
                    input.share(), input.stride(),
                    output.share(), output.stride(), output.shape(),
                    shift, cutoff, stream.cpu());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Phase-shifts a non-redundant 3D (batched) FFT transform.
    /// \tparam REMAP           Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \tparam T               cfloat_t or cdouble_t.
    /// \param[in] input        3D FFT to phase-shift. If empty, the phase-shifts are saved in \p output.
    /// \param[out] output      Non-redundant phase-shifted 3D FFT.
    /// \param[in] shifts       Rightmost 3D real-space forward shift to apply (as phase shift).
    /// \param cutoff           Maximum output frequency to consider, in cycle/pix.
    ///                         Values are usually from 0 (DC) to 0.5 (Nyquist).
    ///                         Frequencies higher than this value are not phase-shifted.
    /// \note \p input and \p output can be equal if no remapping is done, i.e. H2H or HC2HC.
    template<Remap REMAP, typename T>
    void shift3D(const Array<T>& input, const Array<T>& output,
                 const Array<float3_t>& shifts, float cutoff = 0.5f) {
        size4_t input_stride = input.stride();
        if (!input.empty() && !indexing::broadcast(input.shape(), input_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        if (REMAP == Remap::H2HC || REMAP == Remap::HC2H)
            NOA_CHECK(input.get() != output.get(), "In-place phase-shift is not supported with {} remap", REMAP);

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, "
                  "but got input:{} and output:{}", input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            NOA_CHECK(shifts.dereferencable(), "The matrices should be accessible to the host");
            cpu::signal::fft::shift3D<REMAP>(
                    input.share(), input.stride(),
                    output.share(), output.stride(), output.shape(),
                    shifts.share(), cutoff, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::shift3D<REMAP>(
                    input.share(), input.stride(),
                    output.share(), output.stride(), output.shape(),
                    shifts.share(), cutoff, stream.cpu());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    ///  Phase-shifts a non-redundant 3D (batched) FFT.
    /// \see This function is has the same features and limitations than the overload above.
    template<Remap REMAP, typename T>
    void shift3D(const Array<T>& input, const Array<T>& output,
                 float3_t shift, float cutoff = 0.5f) {
        size4_t input_stride = input.stride();
        if (!input.empty() && !indexing::broadcast(input.shape(), input_stride, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        if (REMAP == Remap::H2HC || REMAP == Remap::HC2H)
            NOA_CHECK(input.get() != output.get(), "In-place phase-shift is not supported with {} remap", REMAP);

        const Device device = output.device();
        NOA_CHECK(device == input.device(),
                  "The input and output arrays must be on the same device, "
                  "but got input:{} and output:{}", input.device(), device);

        Stream& stream = Stream::current(device);
        if (device.cpu()) {
            cpu::signal::fft::shift3D<REMAP>(
                    input.share(), input.stride(),
                    output.share(), output.stride(), output.shape(),
                    shift, cutoff, stream.cpu());
        } else {
            #ifdef NOA_ENABLE_CUDA
            cuda::signal::fft::shift3D<REMAP>(
                    input.share(), input.stride(),
                    output.share(), output.stride(), output.shape(),
                    shift, cutoff, stream.cpu());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
