#pragma once

#include "noa/cpu/signal/fft/PhaseShift.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/PhaseShift.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::signal::fft::details {
    using Remap = noa::fft::Remap;

    template<Remap REMAP, typename Input, typename Output, typename Shift>
    void check_phase_shift_parameters(const Input& input, const Output& output,
                                      const Shape4<i64>& shape, const Shift& shifts) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");
        NOA_CHECK(noa::all(output.shape() == shape.rfft()),
                  "Given the logical shape {}, the expected non-redundant shape should be {}, but got {}",
                  shape, shape.rfft(), output.shape());

        if (!input.is_empty()) {
            NOA_CHECK(output.device() == input.device(),
                      "The input and output arrays must be on the same device, but got input:{}, output:{}",
                      input.device(), output.device());
            NOA_CHECK(REMAP == Remap::H2H || REMAP == Remap::HC2HC || input.get() != output.get(),
                      "In-place remapping is not allowed");
        }

        if constexpr (noa::traits::is_array_or_view_v<Shift>) {
            NOA_CHECK(indexing::is_contiguous_vector(shifts) && shifts.elements() == output.shape()[0],
                      "The input shift(s) should be entered as a 1D contiguous vector, with one shift per output batch, "
                      "but got shift {} and output {}", shifts.shape(), output.shape());
            NOA_CHECK(output.device() == shifts.device(),
                      "The shift and output arrays must be on the same device, but got shifts:{}, output:{}",
                      shifts.device(), output.device());
        }
    }

    template<typename Shift>
    auto extract_shift(const Shift& shift) {
        if constexpr (traits::is_realX_v<Shift>) {
            return shift;
        } else {
            using ptr_t = const noa::traits::value_type_t<Shift>*;
            return ptr_t(shift.get());
        }
    }
}

namespace noa::signal::fft {
    using Remap = noa::fft::Remap;

    /// Phase-shifts a non-redundant 2D (batched) FFT.
    /// \tparam REMAP       Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \param[in] input    2D FFT to phase-shift. If empty, the phase-shifts are saved in \p output.
    /// \param[out] output  Non-redundant phase-shifted 2D FFT.
    /// \param shape        BDHW logical shape.
    /// \param[in] shifts   HW 2D phase-shift to apply. A single value or a contiguous vector with one shift per batch.
    /// \param cutoff       Maximum output frequency to consider, in cycle/pix.
    ///                     Values are usually from 0 (DC) to 0.5 (Nyquist).
    ///                     Frequencies higher than this value are not phase-shifted.
    /// \note \p input and \p output can be equal if no remapping is done, i.e. H2H or HC2HC.
    template<Remap REMAP, typename Output, typename Shift,
             typename Input = View<const noa::traits::value_type_t<Output>>, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, c32, c64> &&
             noa::traits::is_array_or_view_of_any_v<Output, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output> &&
             (noa::traits::is_array_or_view_of_almost_any_v<Shift, Vec2<f32>> || std::is_same_v<Shift, Vec2<f32>>) &&
             (REMAP == Remap::H2H || REMAP == Remap::H2HC || REMAP == Remap::HC2H || REMAP == Remap::HC2HC)>>
    void phase_shift_2d(const Input& input, const Output& output, const Shape4<i64>& shape,
                        const Shift& shifts, f32 cutoff = 1) {

        details::check_phase_shift_parameters<REMAP>(input, output, shape, shifts);
        auto input_strides = input.strides();
        if (!input.is_empty() && !noa::indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::signal::fft::phase_shift_2d<REMAP>(
                        input.get(), input_strides,
                        output.get(), output.strides(), shape,
                        details::extract_shift(shifts),
                        cutoff, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::signal::fft::phase_shift_2d<REMAP>(
                    input.get(), input_strides,
                    output.get(), output.strides(), shape,
                    details::extract_shift(shifts),
                    cutoff, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            if constexpr (noa::traits::is_array_or_view_v<Shift>)
                cuda_stream.enqueue_attach(shifts.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Phase-shifts a non-redundant 3D (batched) FFT transform.
    /// \tparam REMAP       Remap operation. Should be H2H, H2HC, HC2HC or HC2H.
    /// \param[in] input    3D FFT to phase-shift. If empty, the phase-shifts are saved in \p output.
    /// \param[out] output  Non-redundant phase-shifted 3D FFT.
    /// \param shape        BDHW logical shape.
    /// \param[in] shifts   DHW 3D phase-shift to apply. A single value or a contiguous vector with one shift per batch.
    /// \param cutoff       Maximum output frequency to consider, in cycle/pix.
    ///                     Values are usually from 0 (DC) to 0.5 (Nyquist).
    ///                     Frequencies higher than this value are not phase-shifted.
    /// \note \p input and \p output can be equal if no remapping is done, i.e. H2H or HC2HC.
    template<Remap REMAP, typename Output, typename Shift,
             typename Input = View<const noa::traits::value_type_t<Output>>, typename = std::enable_if_t<
             noa::traits::is_array_or_view_of_almost_any_v<Input, c32, c64> &&
             noa::traits::is_array_or_view_of_any_v<Output, c32, c64> &&
             noa::traits::are_almost_same_value_type_v<Input, Output> &&
             (noa::traits::is_array_or_view_of_almost_any_v<Shift, Vec3<f32>> || std::is_same_v<Shift, Vec3<f32>>)&&
             (REMAP == Remap::H2H || REMAP == Remap::H2HC || REMAP == Remap::HC2H || REMAP == Remap::HC2HC)>>
    void phase_shift_3d(const Input& input, const Output& output, const Shape4<i64>& shape,
                        const Shift& shifts, f32 cutoff = 1) {

        details::check_phase_shift_parameters<REMAP>(input, output, shape, shifts);
        auto input_strides = input.strides();
        if (!input.is_empty() && !noa::indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.thread_limit();
            cpu_stream.enqueue([=]() {
                cpu::signal::fft::phase_shift_3d<REMAP>(
                        input.get(), input_strides,
                        output.get(), output.strides(), shape,
                        details::extract_shift(shifts),
                        cutoff, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::signal::fft::phase_shift_3d<REMAP>(
                    input.get(), input_strides,
                    output.get(), output.strides(), shape,
                    details::extract_shift(shifts),
                    cutoff, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            if constexpr (noa::traits::is_array_or_view_v<Shift>)
                cuda_stream.enqueue_attach(shifts.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
