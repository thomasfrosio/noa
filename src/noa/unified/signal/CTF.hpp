#pragma once

#include "noa/core/fft/RemapInterface.hpp"
#include "noa/core/signal/CTF.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Iwise.hpp"

namespace noa::signal::guts {
    using Remap = noa::fft::Remap;
    using Layout = noa::fft::Layout;

    template<Remap REMAP, typename Input, typename Output, typename CTF, bool ISOTROPIC>
    struct CTFConcept {
        static constexpr bool is_valid_isotropic_ctf =
                nt::is_ctf_isotropic_v<CTF> or
                (nt::is_varray_v<CTF> and nt::is_ctf_isotropic_v<nt::value_type_t<CTF>>);

        static constexpr bool is_valid_anisotropic_ctf =
                nt::is_ctf_anisotropic_v<CTF> or
                (nt::is_varray_v<CTF> and nt::is_ctf_anisotropic_v<nt::value_type_t<CTF>>);

        static constexpr bool is_valid_ctf =
                (ISOTROPIC and is_valid_isotropic_ctf) or
                (not ISOTROPIC and is_valid_anisotropic_ctf);

        static constexpr bool is_valid_remap =
                (REMAP == Remap::H2H or REMAP == Remap::HC2H or REMAP == Remap::H2HC or REMAP == Remap::HC2HC or
                 REMAP == Remap::F2F or REMAP == Remap::FC2F or REMAP == Remap::F2FC or REMAP == Remap::FC2FC);

        static constexpr bool is_valid_varrays =
                (nt::are_varray_of_real_v<Input, Output> or nt::are_varray_of_complex_v<Input, Output>) or
                (nt::is_varray_of_complex_v<Input> and nt::is_varray_of_real_v<Output>);

        static constexpr bool value = is_valid_ctf and is_valid_remap and is_valid_varrays;
    };

    template<Remap REMAP>
    constexpr Remap remove_input_layout() {
        constexpr u8 REMAP_u8 = to_underlying(REMAP);
        constexpr bool IS_CENTERED = REMAP_u8 & Layout::DST_CENTERED;
        constexpr bool IS_HALF = REMAP_u8 & Layout::DST_HALF;
        constexpr Remap NEW_REMAP =
                IS_CENTERED and IS_HALF ? Remap::HC2HC :
                not IS_CENTERED and IS_HALF ? Remap::H2H :
                IS_CENTERED ? Remap::FC2FC : Remap::F2F;
        return NEW_REMAP;
    }

    template<noa::fft::Remap REMAP, typename Input, typename Output, typename CTF>
    void ctf_check_parameters(
            const Input& input,
            const Output& output,
            const Shape4<i64>& shape,
            const CTF& ctf) {
        check(!output.is_empty(), "Empty array detected");

        constexpr auto u8_REMAP = static_cast<u8>(REMAP);
        constexpr bool IS_HALF = u8_REMAP & noa::fft::Layout::DST_HALF;
        constexpr bool IS_REMAPPED =
                (u8_REMAP & noa::fft::Layout::SRC_CENTERED) !=
                (u8_REMAP & noa::fft::Layout::DST_CENTERED);

        const auto expected_shape = IS_HALF ? shape.rfft() : shape;
        check(all(output.shape() == expected_shape),
              "The output shape doesn't match the expected shape, got output={} and expected={}",
              output.shape(), expected_shape);

        if constexpr (not std::is_empty_v<Input>) {
            if (not input.is_empty()) {
                check(input.device() == output.device(),
                      "The input and output arrays must be on the same device, but got input={} and output={}",
                      input.device(), output.device());
                check(not IS_REMAPPED or not ni::are_overlapped(input, output),
                      "This function cannot execute an in-place multiplication and a remapping");
            }
        }

        if constexpr (nt::is_varray_v<CTF>) {
            check(not ctf.is_empty() and ni::is_contiguous_vector(ctf) and ctf.elements() == shape[0],
                  "The CTFs should be specified as a contiguous vector with {} elements, "
                  "but got shape {} and strides {}",
                  shape[0], ctf.shape(), ctf.strides());
            check(ctf.device() == output.device(),
                  "The ctf and output arrays must be on the same device, but got ctf={} and output={}",
                  ctf.device(), output.device());
        }
        if constexpr (nt::is_ctf_anisotropic_v<CTF> or nt::is_ctf_anisotropic_v<nt::value_type_t<CTF>>) {
            check(shape.ndim() == 2,
                  "Only (batched) 2d arrays are supported with anisotropic CTFs, but got shape {}",
                  shape);
        }
    }

    template<typename CTF>
    auto extract_ctf(const CTF& ctf) {
        if constexpr (nt::is_varray_v<CTF>) {
            return AccessorRestrictContiguous<const nt::value_type_t<CTF>, 1, i64>(ctf.get());
        } else {
            return ctf;
        }
    }
}

namespace noa::signal {
    struct CTFSpectrumOptions {
        /// Frequency [start, end] range of the output, in cycle/pixels, along its the cartesian axes.
        /// If the end is negative (default), set it to the highest frequencies for the given dimensions,
        /// i.e. the entire rfft/fft range is selected.
        Vec2<f64> fftfreq_range{0, -1};

        /// Whether the \p frequency_range 's end should be included in the range.
        bool fftfreq_range_endpoint{true};

        /// Whether the absolute of the ctf should be computed.
        bool ctf_abs{};

        /// Whether the square of the ctf should be computed.
        /// If ctf_abs=true, computes the abs square.
        bool ctf_squared{};
    };

    /// Computes the isotropic ctf.
    /// \details This function can compute the entire fft or rfft spectrum, centered or not (this is with the
    ///          default frequency range). It can also generate a specific frequency range (defined like a
    ///          linspace range, i.e. start, stop, endpoint).
    /// \tparam REMAP       Output layout. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[out] output  1d, 2d, or 3d ctf.
    /// \param shape        Logical BDHW shape.
    /// \param ctf          Isotropic ctf. A contiguous vector of ctfs can be passed. In this case, there should be
    ///                     one ctf per output batch. If a single value is passed, it is applied to every batch.
    /// \param options      Spectrum and ctf options.
    template<noa::fft::RemapInterface REMAP, typename Output, typename CTF>
    requires guts::CTFConcept<REMAP.remap, View<nt::value_type_t<Output>>, Output, CTF, true>::value
    void ctf_isotropic(
            const Output& output,
            Shape4<i64> shape,
            const CTF& ctf,
            const CTFSpectrumOptions& options = {}
    ) {
        guts::ctf_check_parameters<REMAP>(Empty{}, output, shape, ctf);
        const Device device = output.device();

        using noa::fft::Remap;
        using value_t = nt::value_type_t<Output>;
        using coord_t = nt::value_type_twice_t<CTF>;
        using ctf_t = decltype(guts::extract_ctf(ctf));
        constexpr bool is_rfft = REMAP.remap == Remap::H2H or REMAP.remap == Remap::HC2HC;

        switch (shape.ndim()) {
            case 1: {
                using output_accessor_t = Accessor<value_t, 2, i64>;
                using op_t = CTFGenerator<REMAP.remap, 1, coord_t, i64, Empty, output_accessor_t, ctf_t>;
                auto index = ni::non_empty_dhw_dimension(shape);
                auto output_accessor = output_accessor_t(output.get(), output.strides().filter(0, index));
                auto op = op_t(output_accessor, shape.filter(index), guts::extract_ctf(ctf),
                               options.ctf_abs, options.ctf_squared,
                               options.fftfreq_range, options.fftfreq_range_endpoint);

                auto iwise_shape = shape.filter(0, index);
                if constexpr (is_rfft)
                    iwise_shape = iwise_shape.rfft();
                iwise(iwise_shape, device, op, output, ctf);
                break;
            }
            case 2: {
                // Reorder HW dimensions to rightmost.
                auto output_strides = output.strides();
                const auto order = ni::order(output_strides.filter(2, 3), shape.filter(2, 3));
                if (any(order != Vec2<i64>{0, 1})) {
                    std::swap(output_strides[2], output_strides[3]);
                    std::swap(shape[2], shape[3]);
                }

                using output_accessor_t = Accessor<value_t, 3, i64>;
                using op_t = CTFGenerator<REMAP.remap, 2, coord_t, i64, Empty, output_accessor_t, ctf_t>;
                auto output_accessor = output_accessor_t(output.get(), output_strides.filter(0, 2, 3));
                auto op = op_t(output_accessor, shape.filter(2, 3), guts::extract_ctf(ctf),
                               options.ctf_abs, options.ctf_squared,
                               options.fftfreq_range, options.fftfreq_range_endpoint);

                auto iwise_shape = shape.filter(0, 2, 3);
                if constexpr (is_rfft)
                    iwise_shape = iwise_shape.rfft();
                iwise(iwise_shape, device, op, output, ctf);
                break;
            }
            case 3: {
                // Reorder BHW dimensions to rightmost.
                auto output_strides = output.strides();
                const auto order = ni::order(output_strides.pop_front(), shape.pop_front());
                if (any(order != Vec3<i64>{0, 1, 2})) {
                    const auto order_3d = (order + 1).push_front(0);
                    output_strides = ni::reorder(output_strides, order_3d);
                    shape = ni::reorder(shape, order_3d);
                }

                using output_accessor_t = Accessor<value_t, 4, i64>;
                using op_t = CTFGenerator<REMAP.remap, 3, coord_t, i64, Empty, output_accessor_t, ctf_t>;
                auto output_accessor = output_accessor_t(output.get(), output_strides);
                auto op = op_t(output_accessor, shape.pop_front(), guts::extract_ctf(ctf),
                               options.ctf_abs, options.ctf_squared,
                               options.fftfreq_range, options.fftfreq_range_endpoint);

                auto iwise_shape = shape;
                if constexpr (is_rfft)
                    iwise_shape = iwise_shape.rfft();
                iwise(iwise_shape, device, op, output, ctf);
                break;
            }
        }
    }

    /// Computes the isotropic ctf.
    /// \tparam REMAP       Remapping operation. Should be H2H, HC2H, H2HC, HC2HC, F2F, FC2F, F2FC or FC2FC.
    /// \param[in] input    1d, 2d, or 3d array(s) to multiply with the ctf.
    ///                     If empty, the ctf is directly written to \p output.
    ///                     If complex and \p output is real, the power spectrum is first computed is computed.
    /// \param[out] output  1d, 2d, or 3d ctf or ctf-multiplied array, depending on \p input.
    ///                     If no remapping is done, it can be equal to \p input.
    /// \param shape        Logical BDHW shape.
    /// \param ctf          Isotropic ctf. A contiguous vector of ctfs can be passed. In this case, there
    ///                     should be one ctf per output batch. If a single value is passed, it is applied
    ///                     to every batch.
    /// \param options      Spectrum and CTF options. Note that the frequency range options are ignored,
    ///                     if the input is not empty.
    template<noa::fft::RemapInterface REMAP,
             typename Output, typename CTF,
             typename Input = View<nt::value_type_t<Output>>>
    requires guts::CTFConcept<REMAP.remap, Input, Output, CTF, true>::value
    void ctf_isotropic(
            const Input& input,
            const Output& output,
            Shape4<i64> shape,
            const CTF& ctf,
            const CTFSpectrumOptions& options = {}
    ) {
        if (input.is_empty()) {
            return ctf_isotropic<guts::remove_input_layout<REMAP>()>(
                    output, shape, ctf, options);
        }
        guts::ctf_check_parameters<REMAP>(input, output, shape, ctf);

        auto input_strides = input.strides();
        if (not input.is_empty() and not ni::broadcast(input.shape(), input_strides, output.shape())) {
            panic("Cannot broadcast an array of shape {} into an array of shape {}",
                  input.shape(), output.shape());
        }

        const Device device = output.device();

        using noa::fft::Remap;
        using noa::fft::Layout;
        using input_value_t = nt::mutable_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;
        using coord_t = nt::value_type_twice_t<CTF>;
        using ctf_t = decltype(guts::extract_ctf(ctf));
        constexpr bool is_rfft = static_cast<u8>(REMAP.remap) & Layout::SRC_HALF;

        switch (shape.ndim()) {
            case 1: {
                auto index = ni::non_empty_dhw_dimension(shape);
                auto iwise_shape = shape.filter(0, index);
                if constexpr (is_rfft)
                    iwise_shape = iwise_shape.rfft();

                using intput_accessor_t = Accessor<const input_value_t, 2, i64>;
                using output_accessor_t = Accessor<output_value_t, 2, i64>;
                using op_t = CTFGenerator<REMAP.remap, 1, coord_t, i64, intput_accessor_t, output_accessor_t, ctf_t>;
                auto op = op_t(input_accessor_t(input.get(), input_strides.filter(0, index)),
                               output_accessor_t(output.get(), output.strides().filter(0, index)),
                               shape.filter(index), guts::extract_ctf(ctf),
                               options.ctf_abs, options.ctf_squared);
                iwise(iwise_shape, device, op, input, output, ctf);
                break;
            }
            case 2: {
                // Reorder HW dimensions to rightmost.
                auto output_strides = output.strides();
                const auto order = ni::order(output_strides.filter(2, 3), shape.filter(2, 3));
                if (any(order != Vec2<i64>{0, 1})) {
                    std::swap(input_strides[2], input_strides[3]);
                    std::swap(output_strides[2], output_strides[3]);
                    std::swap(shape[2], shape[3]);
                }
                using intput_accessor_t = Accessor<const input_value_t, 3, i64>;
                using output_accessor_t = Accessor<output_value_t, 3, i64>;
                using op_t = CTFGenerator<REMAP.remap, 2, coord_t, i64, intput_accessor_t, output_accessor_t, ctf_t>;
                auto op = op_t(intput_accessor_t(input.get(), input_strides.filter(0, 2, 3)),
                               output_accessor_t(output.get(), output_strides.filter(0, 2, 3)),
                               shape.filter(2, 3), guts::extract_ctf(ctf),
                               options.ctf_abs, options.ctf_squared);

                auto iwise_shape = shape.filter(0, 2, 3);
                if constexpr (is_rfft)
                    iwise_shape = iwise_shape.rfft();
                iwise(iwise_shape, device, op, input, output, ctf);
                break;
            }
            case 3: {
                // Reorder BHW dimensions to rightmost.
                auto output_strides = output.strides();
                const auto order = ni::order(output_strides.pop_front(), shape.pop_front());
                if (any(order != Vec3<i64>{0, 1, 2})) {
                    const auto order_3d = (order + 1).push_front(0);
                    input_strides = ni::reorder(input_strides, order_3d);
                    output_strides = ni::reorder(output_strides, order_3d);
                    shape = ni::reorder(shape, order_3d);
                }

                using intput_accessor_t = Accessor<const input_value_t, 4, i64>;
                using output_accessor_t = Accessor<output_value_t, 4, i64>;
                using op_t = CTFGenerator<REMAP.remap, 3, coord_t, i64, Empty, output_accessor_t, ctf_t>;
                auto op = op_t(intput_accessor_t(input.get(), input_strides),
                               output_accessor_t(output.get(), output_strides),
                               shape.pop_front(), guts::extract_ctf(ctf),
                               options.ctf_abs, options.ctf_squared,
                               options.fftfreq_range, options.fftfreq_range_endpoint);

                auto iwise_shape = shape;
                if constexpr (is_rfft)
                    iwise_shape = iwise_shape.rfft();
                iwise(iwise_shape, device, op, input, output, ctf);
                break;
            }
        }
    }

    /// Computes the anisotropic ctf.
    /// \details This function can compute the entire fft or rfft spectrum, centered or not (this is with the
    ///          default frequency range). It can also generate a specific frequency range (defined like a
    ///          linspace range, i.e. start, stop, endpoint).
    /// \tparam REMAP                   Output layout. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[out] output              2d ctf.
    /// \param shape                    Logical BDHW shape.
    /// \param ctf                      Anisotropic ctf. A contiguous vector of ctfs can be passed. In this case, there
    ///                                 should be one ctf per output batch. If a single value is passed, it is applied
    ///                                 to every batch.
    /// \param ctf_abs                  Whether the absolute of the ctf should be computed.
    /// \param ctf_square               Whether the square of the ctf should be computed.
    /// \param fftfreq_range            Frequency [start, end] range of the output, in cycle/pixels,
    ///                                 along its the cartesian axes. If the end is negative (default), set it to
    ///                                 the highest frequencies for the given dimensions, i.e. the entire rfft/fft
    ///                                 range is selected.
    /// \param fftfreq_range_endpoint   Whether the \p frequency_range 's end should be included in the range.
    template<noa::fft::RemapInterface REMAP, typename Output, typename CTF>
    requires guts::CTFConcept<REMAP.remap, View<nt::value_type_t<Output>>, Output, CTF, false>::value
    void ctf_anisotropic(
            const Output& output,
            Shape4<i64> shape,
            const CTF& ctf,
            const CTFSpectrumOptions& options
    ) {
        guts::ctf_check_parameters<REMAP.remap>(Empty{}, output, shape, ctf);

        using noa::fft::Remap;
        using value_t = nt::value_type_t<Output>;
        using coord_t = nt::value_type_twice_t<CTF>;
        using ctf_t = decltype(guts::extract_ctf(ctf));
        constexpr bool is_rfft = REMAP.remap == Remap::H2H or REMAP.remap == Remap::HC2HC;

        // Reorder HW dimensions to rightmost.
        auto output_strides = output.strides();
        const auto order = ni::order(output_strides.filter(2, 3), shape.filter(2, 3));
        if (any(order != Vec2<i64>{0, 1})) {
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
        }

        using output_accessor_t = Accessor<value_t, 3, i64>;
        using op_t = CTFGenerator<REMAP.remap, 2, coord_t, i64, Empty, output_accessor_t, ctf_t>;
        auto output_accessor = output_accessor_t(output.get(), output_strides.filter(0, 2, 3));
        auto op = op_t(output_accessor, shape.filter(2, 3), guts::extract_ctf(ctf),
                       options.ctf_abs, options.ctf_squared,
                       options.fftfreq_range, options.fftfreq_range_endpoint);

        auto iwise_shape = shape.filter(0, 2, 3);
        if constexpr (is_rfft)
            iwise_shape = iwise_shape.rfft();
        iwise(iwise_shape, output.device(), op, output, ctf);

    }

    /// Computes the anisotropic ctf.
    /// \tparam REMAP       Remapping operation. Should be H2H, HC2H, H2HC, HC2HC, F2F, FC2F, F2FC or FC2FC.
    /// \param[in] input    2d array to multiply with the ctf.
    ///                     If empty, the ctf is directly written to \p output.
    ///                     If complex and \p output is real, \c output=abs(input)^2*ctf is computed.
    /// \param[out] output  2d ctf or ctf-multiplied array.
    ///                     If no remapping is done, it can be equal to \p input.
    /// \param shape        Logical BDHW shape.
    /// \param ctf          Anisotropic ctf. A contiguous vector of ctfs can be passed. In this case, there
    ///                     should be one ctf per output batch. If a single value is passed, it is applied
    ///                     to every batch.
    /// \param options      Spectrum and CTF options. Note that the frequency range options are ignored,
    ///                     if the input is not empty.
    template<noa::fft::RemapInterface REMAP, typename Output, typename CTF,
             typename Input = View<nt::value_type_t<Output>>>
    requires guts::CTFConcept<REMAP.remap, Input, Output, CTF, false>::value
    void ctf_anisotropic(
            const Input& input,
            const Output& output,
            Shape4<i64> shape,
            const CTF& ctf,
            const CTFSpectrumOptions& options
    ) {
        if (input.is_empty())
            return ctf_anisotropic<guts::remove_input_layout<REMAP.remap>()>(output, shape, ctf, options);

        guts::ctf_check_parameters<REMAP>(input, output, shape, ctf);

        auto input_strides = input.strides();
        if (not input.is_empty() and not ni::broadcast(input.shape(), input_strides, output.shape())) {
            panic("Cannot broadcast an array of shape {} into an array of shape {}",
                  input.shape(), output.shape());
        }

        using noa::fft::Remap;
        using noa::fft::Layout;
        using input_value_t = nt::mutable_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;
        using coord_t = nt::value_type_twice_t<CTF>;
        using ctf_t = decltype(guts::extract_ctf(ctf));
        constexpr bool is_rfft = static_cast<u8>(REMAP.remap) & Layout::SRC_HALF;

        // Reorder HW dimensions to rightmost.
        auto output_strides = output.strides();
        const auto order = ni::order(output_strides.filter(2, 3), shape.filter(2, 3));
        if (any(order != Vec2<i64>{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
        }
        using intput_accessor_t = Accessor<const input_value_t, 3, i64>;
        using output_accessor_t = Accessor<output_value_t, 3, i64>;
        using op_t = CTFGenerator<REMAP.remap, 2, coord_t, i64, intput_accessor_t, output_accessor_t, ctf_t>;
        auto op = op_t(intput_accessor_t(input.get(), input_strides.filter(0, 2, 3)),
                       output_accessor_t(output.get(), output_strides.filter(0, 2, 3)),
                       shape.filter(2, 3), guts::extract_ctf(ctf),
                       options.ctf_abs, options.ctf_squared);

        auto iwise_shape = shape.filter(0, 2, 3);
        if constexpr (is_rfft)
            iwise_shape = iwise_shape.rfft();
        iwise(iwise_shape, output.device(), op, input, output, ctf);
    }
}
