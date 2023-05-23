#pragma once

#include "noa/cpu/signal/fft/CTF.hpp"
#ifdef NOA_ENABLE_CUDA
#include "noa/gpu/cuda/signal/fft/CTF.hpp"
#endif

#include "noa/unified/Array.hpp"

namespace noa::signal::fft::details {
    using Remap = noa::fft::Remap;
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
            (noa::traits::are_same_value_type_v<Input, Output> &&
             ((noa::traits::are_all_same_v<Input, Output> &&
               noa::traits::are_real_or_complex_v<Input, Output>) ||
              (noa::traits::is_complex_v<Input> &&
               noa::traits::is_real_v<Output>)));

    template<noa::fft::Remap REMAP, typename Input, typename Output, typename CTF>
    void ctf_check_parameters(
            const Input& input,
            const Output& output,
            const Shape4<i64>& shape,
            const CTF& ctf) {
        NOA_CHECK(!output.is_empty(), "Empty array detected");

        constexpr bool IS_HALF = static_cast<u8>(REMAP) & noa::fft::Layout::DST_HALF;
        const auto expected_shape = IS_HALF ? shape.rfft() : shape;
        NOA_CHECK(noa::all(output.shape() == expected_shape),
                  "The output shape doesn't match the expected shape, got output={} and expected={}",
                  output.shape(), expected_shape);

        if (!input.is_empty()) {
            NOA_CHECK(input.device() == output.device(),
                      "The input and output arrays must be on the same device, but got input={} and output={}",
                      input.device(), output.device());
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
    template<noa::fft::Remap REMAP, typename Input, typename Output, typename CTF, typename = std::enable_if_t<
             details::is_valid_ctf_v<REMAP, Input, Output, CTF> && details::ctf_parser_t<CTF>::IS_ISOTROPIC>>
    void ctf_isotropic(
            const Input& input,
            const Output& output,
            const Shape4<i64>& shape,
            const CTF& ctf,
            bool ctf_square = false,
            bool ctf_abs = false
    ) {
        details::ctf_check_parameters(input, output, shape, ctf);

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
        details::ctf_check_parameters(input, output, shape, ctf);

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
