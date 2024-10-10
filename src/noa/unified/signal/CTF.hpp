#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/utils/BatchedParameter.hpp"
#include "noa/core/signal/CTF.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Iwise.hpp"

namespace noa::signal::guts {
    template<typename T, typename U = nt::value_type_t<T>, typename V = std::decay_t<T>>
    concept varray_decay_or_ctf_isotropic = (nt::ctf_isotropic<V> or (nt::varray<V> and nt::ctf_isotropic<U>));

    template<typename T, typename U = nt::value_type_t<T>, typename V = std::decay_t<T>>
    concept varray_decay_or_ctf_anisotropic = (nt::ctf_anisotropic<V> or (nt::varray<V> and nt::ctf_anisotropic<U>));

    template<Remap REMAP, typename Input, typename Output, typename CTF, bool ISOTROPIC>
    concept ctfable =
        ((ISOTROPIC and varray_decay_or_ctf_isotropic<CTF>) or (not ISOTROPIC and varray_decay_or_ctf_anisotropic<CTF>)) and
        (REMAP.is_hx2hx() or REMAP.is_fx2fx()) and
        nt::varray_decay_with_spectrum_types<Input, Output>;

    template<Remap REMAP, typename Input, typename Output, typename CTF>
    void ctf_check_parameters(
            const Input& input,
            const Output& output,
            const Shape4<i64>& shape,
            const CTF& ctf
    ) {
        check(not output.is_empty(), "Empty array detected");

        const auto expected_shape = REMAP.is_xx2hx() ? shape.rfft() : shape;
        check(vall(Equal{}, output.shape(), expected_shape),
              "The output shape doesn't match the expected shape. Got output:shape={} and expected:shape={}",
              output.shape(), expected_shape);

        if constexpr (not nt::empty<Input>) {
            if (not input.is_empty()) {
                check(input.device() == output.device(),
                      "The input and output arrays must be on the same device, "
                      "but got input:device={} and output:device={}",
                      input.device(), output.device());
                check(not REMAP.has_layout_change() or not ni::are_overlapped(input, output),
                      "This function cannot execute an in-place multiplication and a remapping");
            }
        }

        if constexpr (nt::varray<CTF>) {
            check(not ctf.is_empty() and ni::is_contiguous_vector(ctf) and ctf.n_elements() == shape[0],
                  "The CTFs should be specified as a contiguous vector with {} elements, "
                  "but got ctf:shape={} and ctf:strides={}",
                  shape[0], ctf.shape(), ctf.strides());
            check(ctf.device() == output.device(),
                  "The ctf and output arrays must be on the same device, but got ctf:device={} and output:device={}",
                  ctf.device(), output.device());
        }
        if constexpr (nt::ctf_anisotropic<CTF> or nt::ctf_anisotropic<nt::value_type_t<CTF>>) {
            check(shape.ndim() == 2,
                  "Only (batched) 2d arrays are supported with anisotropic CTFs, but got ctf:shape={}",
                  shape);
        }
    }

    template<typename CTF>
    constexpr auto extract_ctf(const CTF& ctf) {
        if constexpr (nt::varray<CTF>) {
            using accessor_t = AccessorRestrictContiguousI64<nt::const_value_type_t<CTF>, 1>;
            return BatchedParameter{accessor_t(ctf.get())};
        } else {
            return BatchedParameter{ctf};
        }
    }
}

namespace noa::signal {
    struct CTFOptions {
        /// Frequency [start, end] range of the output, in cycle/pixels, from the zero, along the cartesian axes.
        /// If the end is negative (default), the library sets it to the highest frequencies for the given dimensions,
        /// i.e. the entire rfft/fft range is selected. For even dimensions, this is equivalent to {0, 0.5}.
        Vec2<f64> fftfreq_range{0, -1};

        /// Whether the \p frequency_range 's end should be included in the range.
        bool fftfreq_range_endpoint{true};

        /// Whether the absolute of the ctf should be computed.
        bool ctf_abs{};

        /// Whether the square of the ctf should be computed.
        /// If ctf_abs=true, computes the abs square (aka the power spectrum).
        bool ctf_squared{};
    };

    /// Computes isotropic CTF(s) over entire FFT or rFFT spectrum or over a specific frequency range (see \p options).
    /// \tparam REMAP       Output layout. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[out] output  1d, 2d, or 3d CTF(s).
    /// \param shape        Logical BDHW shape.
    /// \param[in] ctf      Isotropic CTF(s). A contiguous vector of CTFs can be passed. In this case, there should be
    ///                     one CTF per output batch. If a single value is passed, it is applied to every batch.
    /// \param options      Spectrum and CTF options.
    template<Remap REMAP, typename Output, typename CTF>
    requires guts::ctfable<REMAP, View<nt::value_type_t<Output>>, Output, CTF, true>
    void ctf_isotropic(
        Output&& output,
        Shape4<i64> shape,
        CTF&& ctf,
        const CTFOptions& options = {}
    ) {
        guts::ctf_check_parameters<REMAP>(Empty{}, output, shape, ctf);
        const Device device = output.device();

        using value_t = nt::value_type_t<Output>;
        using coord_t = nt::value_type_twice_t<CTF>;
        using ctf_t = decltype(guts::extract_ctf(ctf));

        switch (shape.ndim()) {
            case 1: {
                using output_accessor_t = Accessor<value_t, 2, i64>;
                using op_t = guts::CTF<REMAP, 1, coord_t, i64, Empty, output_accessor_t, ctf_t>;
                auto index = ni::non_empty_dhw_dimension(shape);
                auto output_accessor = output_accessor_t(output.get(), output.strides().filter(0, index));
                auto op = op_t(output_accessor, shape.filter(index), guts::extract_ctf(ctf),
                               options.ctf_abs, options.ctf_squared,
                               options.fftfreq_range, options.fftfreq_range_endpoint);

                auto iwise_shape = shape.filter(0, index);
                if constexpr (REMAP.is_hx2hx())
                    iwise_shape = iwise_shape.rfft();
                return iwise(iwise_shape, device, op, std::forward<Output>(output), std::forward<CTF>(ctf));
            }
            case 2: {
                auto output_strides = output.strides();
                const auto order = ni::order(output_strides.filter(2, 3), shape.filter(2, 3));
                if (vany(NotEqual{}, order, Vec{0, 1})) {
                    std::swap(output_strides[2], output_strides[3]);
                    std::swap(shape[2], shape[3]);
                }

                using output_accessor_t = Accessor<value_t, 3, i64>;
                using op_t = guts::CTF<REMAP, 2, coord_t, i64, Empty, output_accessor_t, ctf_t>;
                auto output_accessor = output_accessor_t(output.get(), output_strides.filter(0, 2, 3));
                auto op = op_t(output_accessor, shape.filter(2, 3), guts::extract_ctf(ctf),
                               options.ctf_abs, options.ctf_squared,
                               options.fftfreq_range, options.fftfreq_range_endpoint);

                auto iwise_shape = shape.filter(0, 2, 3);
                if constexpr (REMAP.is_hx2hx())
                    iwise_shape = iwise_shape.rfft();
                return iwise(iwise_shape, device, op, std::forward<Output>(output), std::forward<CTF>(ctf));
            }
            case 3: {
                auto output_strides = output.strides();
                const auto order = ni::order(output_strides.pop_front(), shape.pop_front());
                if (vany(NotEqual{}, order, Vec{0, 1, 2})) {
                    const auto order_3d = (order + 1).push_front(0);
                    output_strides = ni::reorder(output_strides, order_3d);
                    shape = ni::reorder(shape, order_3d);
                }

                using output_accessor_t = Accessor<value_t, 4, i64>;
                using op_t = guts::CTF<REMAP, 3, coord_t, i64, Empty, output_accessor_t, ctf_t>;
                auto output_accessor = output_accessor_t(output.get(), output_strides);
                auto op = op_t(output_accessor, shape.pop_front(), guts::extract_ctf(ctf),
                               options.ctf_abs, options.ctf_squared,
                               options.fftfreq_range, options.fftfreq_range_endpoint);

                auto iwise_shape = shape;
                if constexpr (REMAP.is_hx2hx())
                    iwise_shape = iwise_shape.rfft();
                return iwise(iwise_shape, device, op, std::forward<Output>(output), std::forward<CTF>(ctf));
            }
        }
    }

    /// Computes isotropic CTF(s).
    /// \tparam REMAP       Remapping operation. Should be H2H, HC2H, H2HC, HC2HC, F2F, FC2F, F2FC or FC2FC.
    /// \param[in] input    1d, 2d, or 3d array(s) to multiply with the CTF.
    ///                     If empty, the CTF is directly written to \p output.
    ///                     If complex and \p output is real, the power spectrum is first computed.
    /// \param[out] output  1d, 2d, or 3d CTF or CTF-multiplied array, depending on \p input.
    ///                     If no remapping is done, it can be equal to \p input.
    /// \param shape        Logical BDHW shape.
    /// \param[in] ctf      Isotropic CTF. A contiguous vector of CTFs can be passed. In this case, there
    ///                     should be one CTF per output batch. If a single value is passed, it is applied
    ///                     to every batch.
    /// \param options      Spectrum and CTF options. The frequency range is ignored if the input is not empty.
    template<Remap REMAP,
             typename Output, typename CTF,
             typename Input = View<nt::value_type_t<Output>>>
    requires guts::ctfable<REMAP, Input, Output, CTF, true>
    void ctf_isotropic(
        Input&& input,
        Output&& output,
        Shape4<i64> shape,
        CTF&& ctf,
        const CTFOptions& options = {}
    ) {
        if (input.is_empty()) {
            return ctf_isotropic<REMAP.erase_input()>(
                std::forward<Output>(output), shape, std::forward<CTF>(ctf), options);
        }

        guts::ctf_check_parameters<REMAP>(input, output, shape, ctf);

        auto input_strides = ng::broadcast_strides_optional(input, output);
        const Device device = output.device();

        using input_value_t = nt::const_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;
        using coord_t = nt::value_type_twice_t<CTF>;
        using ctf_t = decltype(guts::extract_ctf(ctf));

        switch (shape.ndim()) {
            case 1: {
                auto index = ni::non_empty_dhw_dimension(shape);
                auto iwise_shape = shape.filter(0, index);
                if constexpr (REMAP.is_xx2hx())
                    iwise_shape = iwise_shape.rfft();

                using input_accessor_t = Accessor<input_value_t, 2, i64>;
                using output_accessor_t = Accessor<output_value_t, 2, i64>;
                using op_t = guts::CTF<REMAP, 1, coord_t, i64, input_accessor_t, output_accessor_t, ctf_t>;
                auto op = op_t(input_accessor_t(input.get(), input_strides.filter(0, index)),
                               output_accessor_t(output.get(), output.strides().filter(0, index)),
                               shape.filter(index), guts::extract_ctf(ctf),
                               options.ctf_abs, options.ctf_squared);
                return iwise(iwise_shape, device, op,
                             std::forward<Input>(input),
                             std::forward<Output>(output),
                             std::forward<CTF>(ctf));
            }
            case 2: {
                auto output_strides = output.strides();
                const auto order = ni::order(output_strides.filter(2, 3), shape.filter(2, 3));
                if (vany(NotEqual{}, order, Vec{0, 1})) {
                    std::swap(input_strides[2], input_strides[3]);
                    std::swap(output_strides[2], output_strides[3]);
                    std::swap(shape[2], shape[3]);
                }
                using input_accessor_t = Accessor<input_value_t, 3, i64>;
                using output_accessor_t = Accessor<output_value_t, 3, i64>;
                using op_t = guts::CTF<REMAP, 2, coord_t, i64, input_accessor_t, output_accessor_t, ctf_t>;
                auto op = op_t(input_accessor_t(input.get(), input_strides.filter(0, 2, 3)),
                               output_accessor_t(output.get(), output_strides.filter(0, 2, 3)),
                               shape.filter(2, 3), guts::extract_ctf(ctf),
                               options.ctf_abs, options.ctf_squared);

                auto iwise_shape = shape.filter(0, 2, 3);
                if constexpr (REMAP.is_xx2hx())
                    iwise_shape = iwise_shape.rfft();
                return iwise(iwise_shape, device, op,
                             std::forward<Input>(input),
                             std::forward<Output>(output),
                             std::forward<CTF>(ctf));
            }
            case 3: {
                auto output_strides = output.strides();
                const auto order = ni::order(output_strides.pop_front(), shape.pop_front());
                if (vany(NotEqual{}, order, Vec{0, 1, 2})) {
                    const auto order_3d = (order + 1).push_front(0);
                    input_strides = ni::reorder(input_strides, order_3d);
                    output_strides = ni::reorder(output_strides, order_3d);
                    shape = ni::reorder(shape, order_3d);
                }

                using input_accessor_t = Accessor<input_value_t, 4, i64>;
                using output_accessor_t = Accessor<output_value_t, 4, i64>;
                using op_t = guts::CTF<REMAP, 3, coord_t, i64, input_accessor_t, output_accessor_t, ctf_t>;
                auto op = op_t(input_accessor_t(input.get(), input_strides),
                               output_accessor_t(output.get(), output_strides),
                               shape.pop_front(), guts::extract_ctf(ctf),
                               options.ctf_abs, options.ctf_squared);

                auto iwise_shape = shape;
                if constexpr (REMAP.is_xx2hx())
                    iwise_shape = iwise_shape.rfft();
                return iwise(iwise_shape, device, op,
                             std::forward<Input>(input),
                             std::forward<Output>(output),
                             std::forward<CTF>(ctf));
            }
        }
    }

    /// Computes anisotropic CTF(s) over entire FFT or rFFT spectrum or over a specific frequency range (see \p options).
    /// \tparam REMAP       Output layout. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[out] output  2d CTF(s).
    /// \param shape        Logical BDHW shape.
    /// \param[in] ctf      Anisotropic CTF(s). A contiguous vector of CTFs can be passed, in which case there
    ///                     should be one CTF per output batch. If a single value is passed, it is applied
    ///                     to every batch.
    /// \param options      Spectrum and CTF options.
    template<Remap REMAP, typename Output, typename CTF>
    requires guts::ctfable<REMAP, View<nt::value_type_t<Output>>, Output, CTF, false>
    void ctf_anisotropic(
        Output&& output,
        Shape4<i64> shape,
        CTF&& ctf,
        const CTFOptions& options = {}
    ) {
        guts::ctf_check_parameters<REMAP>(Empty{}, output, shape, ctf);

        using value_t = nt::value_type_t<Output>;
        using coord_t = nt::value_type_twice_t<CTF>;
        using ctf_t = decltype(guts::extract_ctf(ctf));

        auto output_strides = output.strides();
        const auto order = ni::order(output_strides.filter(2, 3), shape.filter(2, 3));
        if (vany(NotEqual{}, order, Vec{0, 1})) {
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
        }

        using output_accessor_t = Accessor<value_t, 3, i64>;
        using op_t = guts::CTF<REMAP, 2, coord_t, i64, Empty, output_accessor_t, ctf_t>;
        auto output_accessor = output_accessor_t(output.get(), output_strides.filter(0, 2, 3));
        auto op = op_t(output_accessor, shape.filter(2, 3), guts::extract_ctf(ctf),
                       options.ctf_abs, options.ctf_squared,
                       options.fftfreq_range, options.fftfreq_range_endpoint);

        auto iwise_shape = shape.filter(0, 2, 3);
        if constexpr (REMAP.is_xx2hx())
            iwise_shape = iwise_shape.rfft();
        iwise(iwise_shape, output.device(), op, std::forward<Output>(output), std::forward<CTF>(ctf));
    }

    /// Computes the anisotropic CTF(s).
    /// \tparam REMAP       Remapping operation. Should be H2H, HC2H, H2HC, HC2HC, F2F, FC2F, F2FC or FC2FC.
    /// \param[in] input    2d array to multiply with the CTF.
    ///                     If empty, the CTF is directly written to \p output.
    ///                     If complex and \p output is real, \c output=abs(input)^2*CTF is computed.
    /// \param[out] output  2d CTF or CTF-multiplied array.
    ///                     If no remapping is done, it can be equal to \p input.
    /// \param shape        Logical BDHW shape.
    /// \param[in] ctf      Anisotropic CTF(s). A contiguous vector of CTFs can be passed. In this case, there
    ///                     should be one CTF per output batch. If a single value is passed, it is applied
    ///                     to every batch.
    /// \param options      Spectrum and CTF options. frequency range is ignored if the input is not empty.
    template<Remap REMAP, typename Output, typename CTF,
             typename Input = View<nt::value_type_t<Output>>>
    requires guts::ctfable<REMAP, Input, Output, CTF, false>
    void ctf_anisotropic(
        Input&& input,
        Output&& output,
        Shape4<i64> shape,
        CTF&& ctf,
        const CTFOptions& options = {}
    ) {
        if (input.is_empty()) {
            return ctf_anisotropic<REMAP.erase_input()>(
                std::forward<Output>(output), shape, std::forward<CTF>(ctf), options);
        }

        guts::ctf_check_parameters<REMAP>(input, output, shape, ctf);

        auto input_strides = ng::broadcast_strides_optional(input, output);
        auto output_strides = output.strides();
        const auto order = ni::order(output_strides.filter(2, 3), shape.filter(2, 3));
        if (vany(NotEqual{}, order, Vec{0, 1})) {
            std::swap(input_strides[2], input_strides[3]);
            std::swap(output_strides[2], output_strides[3]);
            std::swap(shape[2], shape[3]);
        }
        using input_accessor_t = Accessor<nt::const_value_type_t<Input>, 3, i64>;
        using output_accessor_t = Accessor<nt::value_type_t<Output>, 3, i64>;
        using coord_t = nt::value_type_twice_t<CTF>;
        using ctf_t = decltype(guts::extract_ctf(ctf));
        using op_t = guts::CTF<REMAP, 2, coord_t, i64, input_accessor_t, output_accessor_t, ctf_t>;
        auto op = op_t(input_accessor_t(input.get(), input_strides.filter(0, 2, 3)),
                       output_accessor_t(output.get(), output_strides.filter(0, 2, 3)),
                       shape.filter(2, 3), guts::extract_ctf(ctf),
                       options.ctf_abs, options.ctf_squared);

        auto iwise_shape = shape.filter(0, 2, 3);
        if constexpr (REMAP.is_xx2hx())
            iwise_shape = iwise_shape.rfft();

        iwise(iwise_shape, output.device(), op,
              std::forward<Input>(input),
              std::forward<Output>(output),
              std::forward<CTF>(ctf));
    }
}
