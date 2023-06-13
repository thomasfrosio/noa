#pragma once

#include "noa/cpu/signal/fft/CTF.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/CTF.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::signal::fft::details {
    using Remap = noa::fft::Remap;
    using Layout = noa::fft::Layout;
    namespace nt = noa::traits;
    namespace ns = noa::signal;

    template<typename CTF>
    struct ctf_parser_t {
        using iso32_type = ns::fft::CTFIsotropic<f32>;
        using iso64_type = ns::fft::CTFIsotropic<f64>;
        using aniso32_type = ns::fft::CTFAnisotropic<f32>;
        using aniso64_type = ns::fft::CTFAnisotropic<f64>;

        static constexpr bool IS_ISOTROPIC =
                nt::is_any_v<CTF, iso32_type, iso64_type> ||
                nt::is_array_or_view_of_any_v<CTF, iso32_type, iso64_type>;
        static constexpr bool IS_ANISOTROPIC =
                nt::is_any_v<CTF, aniso32_type, aniso64_type> ||
                nt::is_array_or_view_of_any_v<CTF, aniso32_type, aniso64_type>;
        static constexpr bool IS_VALID = IS_ISOTROPIC || IS_ANISOTROPIC;
    };

    template<Remap REMAP, typename Input, typename Output, typename CTF>
    constexpr bool is_valid_ctf_v =
            ctf_parser_t<CTF>::IS_VALID &&
            (REMAP == Remap::H2H || REMAP == Remap::HC2H || REMAP == Remap::H2HC || REMAP == Remap::HC2HC ||
             REMAP == Remap::F2F || REMAP == Remap::FC2F || REMAP == Remap::F2FC || REMAP == Remap::FC2FC) &&
            (noa::traits::are_same_value_type_v<noa::traits::value_type_t<Input>, noa::traits::value_type_t<Output>> &&
             ((noa::traits::are_same_value_type_v<Input, Output> &&
               noa::traits::are_array_or_view_of_real_or_complex_v<Input, Output>) ||
              (noa::traits::is_array_or_view_of_complex_v<Input> &&
               noa::traits::is_array_or_view_of_real_v<Output>)));

    template<Remap REMAP>
    constexpr Remap remove_input_layout() {
        constexpr u8 REMAP_u8 = noa::traits::to_underlying(REMAP);
        constexpr bool IS_CENTERED = REMAP_u8 & Layout::DST_CENTERED;
        constexpr bool IS_HALF = REMAP_u8 & Layout::DST_HALF;
        constexpr Remap NEW_REMAP =
                IS_CENTERED && IS_HALF ? Remap::HC2HC :
                !IS_CENTERED && IS_HALF ? Remap::H2H :
                IS_CENTERED ? Remap::FC2FC : Remap::F2F;
        return NEW_REMAP;
    }

    template<noa::fft::Remap REMAP, typename Input, typename Output, typename CTF>
    void ctf_check_parameters(
            const Input& input,
            const Output& output,
            const Shape4<i64>& shape,
            const CTF& ctf) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");

        constexpr auto u8_REMAP = static_cast<u8>(REMAP);
        constexpr bool IS_HALF = u8_REMAP & noa::fft::Layout::DST_HALF;
        constexpr bool IS_REMAPPED =
                (u8_REMAP & noa::fft::Layout::SRC_CENTERED) !=
                (u8_REMAP & noa::fft::Layout::DST_CENTERED);

        const auto expected_shape = IS_HALF ? shape.rfft() : shape;
        NOA_CHECK(noa::all(output.shape() == expected_shape),
                  "The output shape doesn't match the expected shape, got output={} and expected={}",
                  output.shape(), expected_shape);

        if constexpr (!std::is_empty_v<Input>) {
            if (!input.is_empty()) {
                NOA_CHECK(input.device() == output.device(),
                          "The input and output arrays must be on the same device, but got input={} and output={}",
                          input.device(), output.device());
                NOA_CHECK(!IS_REMAPPED || !noa::indexing::are_overlapped(input, output),
                          "This function cannot execute an in-place multiplication and a remapping");
            }
        }

        if constexpr (noa::traits::is_array_or_view_v<CTF>) {
            NOA_CHECK(!ctf.is_empty() && noa::indexing::is_contiguous_vector(ctf) && ctf.size() == shape[0],
                      "The CTFs should be specified as a contiguous vector with {} elements, "
                      "but got shape {} and strides {}",
                      shape[0], ctf.shape(), ctf.strides());
        }
        if constexpr (ctf_parser_t<CTF>::IS_ANISOTROPIC) {
            NOA_CHECK(shape.ndim() == 2,
                      "Only (batched) 2d arrays are supported with anisotropic CTFs, but got shape {}",
                      shape);
        }
    }

    template<typename CTF>
    auto extract_ctf(const CTF& ctf) {
        if constexpr (nt::is_array_or_view_v<CTF>) {
            using ptr_t = const noa::traits::value_type_t<CTF>*;
            return ptr_t(ctf.get());
        } else {
            return ctf;
        }
    }
}

namespace noa::signal::fft {

    /// Computes the isotropic ctf.
    /// \tparam REMAP                   Output layout. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[out] output              1d, 2d, or 3d ctf.
    /// \param shape                    Logical BDHW shape.
    /// \param ctf                      Isotropic ctf. A contiguous vector of ctfs can be passed. In this case, there
    ///                                 should be one ctf per output batch. If a single value is passed, it is applied
    ///                                 to every batch.
    /// \param ctf_abs                  Whether the absolute of the ctf should be computed.
    /// \param ctf_square               Whether the square of the ctf should be computed.
    /// \param fftfreq_range            Frequency [start, end] range of the output, in cycle/pixels.
    ///                                 If the end is negative (default), set it to the highest frequencies
    ///                                 for the given dimensions, i.e. the entire rfft/fft range is selected.
    /// \param fftfreq_range_endpoint   Whether the \p frequency_range 's end should be included in the range.
    template<noa::fft::Remap REMAP,
            typename Output, typename CTF,
            typename = std::enable_if_t<
                    details::is_valid_ctf_v<REMAP, View<noa::traits::value_type_t<Output>>, Output, CTF> &&
                    details::ctf_parser_t<CTF>::IS_ISOTROPIC>>
    void ctf_isotropic(
            const Output& output,
            const Shape4<i64>& shape,
            const CTF& ctf,
            bool ctf_abs = false,
            bool ctf_square = false,
            Vec2<f32> fftfreq_range = {0, -1},
            bool fftfreq_range_endpoint = true
    ) {
        details::ctf_check_parameters<REMAP>(Empty{}, output, shape, ctf);

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::signal::fft::ctf_isotropic<REMAP>(
                        output.get(), output.strides(), shape,
                        details::extract_ctf(ctf), ctf_square, ctf_abs,
                        fftfreq_range, fftfreq_range_endpoint, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::signal::fft::ctf_isotropic<REMAP>(
                    output.get(), output.strides(), shape,
                    details::extract_ctf(ctf), ctf_square, ctf_abs,
                    fftfreq_range, fftfreq_range_endpoint, cuda_stream);
            cuda_stream.enqueue_attach(output.share());
            if constexpr (noa::traits::is_array_or_view_v<CTF>)
                cuda_stream.enqueue_attach(ctf.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    /// Computes the isotropic ctf.
    /// \tparam REMAP       Remapping operation. Should be H2H, HC2H, H2HC, HC2HC, F2F, FC2F, F2FC or FC2FC.
    /// \param[in] input    1d, 2d, or 3d array to multiply with the ctf.
    ///                     If empty, the ctf is directly written to \p output.
    ///                     If complex and \p output is real, \c output=abs(input)^2*ctf is computed.
    /// \param[out] output  1d, 2d, or 3d ctf or ctf-multiplied array.
    ///                     If no remapping is done, it can be equal to \p input.
    /// \param shape        Logical BDHW shape.
    /// \param ctf          Isotropic ctf. A contiguous vector of ctfs can be passed. In this case, there
    ///                     should be one ctf per output batch. If a single value is passed, it is applied
    ///                     to every batch.
    /// \param ctf_abs      Whether the absolute of the ctf should be computed.
    /// \param ctf_square   Whether the square of the ctf should be computed.
    template<noa::fft::Remap REMAP,
             typename Output, typename CTF,
             typename Input = View<noa::traits::value_type_t<Output>>,
             typename = std::enable_if_t<
                     details::is_valid_ctf_v<REMAP, Input, Output, CTF> &&
                     details::ctf_parser_t<CTF>::IS_ISOTROPIC>>
    void ctf_isotropic(
            const Input& input,
            const Output& output,
            const Shape4<i64>& shape,
            const CTF& ctf,
            bool ctf_abs = false,
            bool ctf_square = false
    ) {
        if (input.is_empty()) {
            return ctf_isotropic<details::remove_input_layout<REMAP>()>(
                    output, shape, ctf, ctf_abs, ctf_square);
        }
        details::ctf_check_parameters<REMAP>(input, output, shape, ctf);

        auto input_strides = input.strides();
        if (!input.is_empty() && !noa::indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::signal::fft::ctf_isotropic<REMAP>(
                        input.get(), input_strides,
                        output.get(), output.strides(), shape,
                        details::extract_ctf(ctf), ctf_square, ctf_abs, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::signal::fft::ctf_isotropic<REMAP>(
                    input.get(), input_strides,
                    output.get(), output.strides(), shape,
                    details::extract_ctf(ctf), ctf_square, ctf_abs, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            if constexpr (noa::traits::is_array_or_view_v<CTF>)
                cuda_stream.enqueue_attach(ctf.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }

    // TODO
    template<noa::fft::Remap REMAP, typename Input, typename Output, typename CTF, typename = std::enable_if_t<
             details::is_valid_ctf_v<REMAP, Input, Output, CTF> && details::ctf_parser_t<CTF>::IS_ANISOTROPIC>>
    void ctf_anisotropic(
            const Input& input,
            const Output& output,
            const Shape4<i64>& shape,
            const CTF& ctf,
            bool ctf_square = false,
            bool ctf_abs = false
    ) {
        details::ctf_check_parameters<REMAP>(input, output, shape, ctf);

        auto input_strides = input.strides();
        if (!input.is_empty() && !noa::indexing::broadcast(input.shape(), input_strides, output.shape())) {
            NOA_THROW("Cannot broadcast an array of shape {} into an array of shape {}",
                      input.shape(), output.shape());
        }

        const Device device = output.device();
        Stream& stream = Stream::current(device);
        if (device.is_cpu()) {
            auto& cpu_stream = stream.cpu();
            const auto threads = cpu_stream.threads();
            cpu_stream.enqueue([=]() {
                cpu::signal::fft::ctf_anisotropic<REMAP>(
                        input.get(), input_strides,
                        output.get(), output.strides(), shape,
                        details::extract_ctf(ctf), ctf_square, ctf_abs, threads);
            });
        } else {
            #ifdef NOA_ENABLE_CUDA
            auto& cuda_stream = stream.cuda();
            cuda::signal::fft::ctf_anisotropic<REMAP>(
                    input.get(), input_strides,
                    output.get(), output.strides(), shape,
                    details::extract_ctf(ctf), ctf_square, ctf_abs, cuda_stream);
            cuda_stream.enqueue_attach(input.share(), output.share());
            if constexpr (noa::traits::is_array_or_view_v<CTF>)
                cuda_stream.enqueue_attach(ctf.share());
            #else
            NOA_THROW("No GPU backend detected");
            #endif
        }
    }
}
