#pragma once

#include "noa/core/Enums.hpp"
#include "noa/core/Iwise.hpp"
#include "noa/core/utils/BatchedParameter.hpp"
#include "noa/core/signal/CTF.hpp"
#include "noa/core/fft/Frequency.hpp"
#include "noa/unified/Array.hpp"
#include "noa/unified/Iwise.hpp"

namespace noa::signal::guts {
    /// Index-wise operator, to compute/apply CTFs to {1|2|3}d DFTs.
    /// \details If the input is valid, input*ctf->output is computed.
    ///          If the input is complex and the output is real, the power spectrum is saved.
    ///          If the input is empty, ctf->output is computed.
    template<Remap REMAP, size_t N,
             nt::any_of<f32, f64> Coord,
             nt::sinteger Index,
             nt::readable_nd_or_empty<N + 1> Input,
             nt::writable_nd<N + 1> Output,
             nt::batched_parameter CTFParameter>
    class CTF {
    public:
        static_assert(REMAP.is_any(Remap::H2H, Remap::HC2HC, Remap::HC2H, Remap::H2HC,
                                   Remap::F2F, Remap::FC2FC, Remap::FC2F, Remap::F2FC));

        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::value_type_t<input_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static_assert(nt::spectrum_types<input_value_type, output_value_type> or
                      (nt::empty<input_type> and nt::real_or_complex<output_value_type>));

        using ctf_parameter_type = CTFParameter;
        using ctf_type = nt::value_type_t<ctf_parameter_type>;
        static_assert(nt::ctf<ctf_type>);

        static constexpr bool HAS_INPUT = not nt::empty<input_type>;
        static constexpr bool IS_RFFT = REMAP.is_hx2hx();
        static constexpr bool IS_DST_CENTERED = REMAP.is_xx2xc();
        static constexpr bool IS_ISOTROPIC = nt::ctf_isotropic<ctf_type>;

        using index_type = Index;
        using coord_type = Coord;
        using shape_nd_type = Shape<index_type, N>;
        using shape_type = Shape<index_type, N - IS_RFFT>;
        using coord_nd_type = Vec<coord_type, N>;
        using coord2_type = Vec2<coord_type>;
        using coord_or_empty_type = std::conditional_t<N == 1, coord_type, Empty>;

    public:
        constexpr CTF(
            const input_type& input,
            const output_type& output,
            const shape_nd_type& shape,
            const ctf_parameter_type& ctf,
            bool ctf_abs,
            bool ctf_squared,
            const Linspace<coord_type>& fftfreq_range
        ) :
            m_ctf(ctf),
            m_output(output),
            m_shape(shape.template pop_back<IS_RFFT>()),
            m_input(input),
            m_ctf_abs(ctf_abs),
            m_ctf_squared(ctf_squared)
        {
            // If frequency.end is negative, defaults to the highest frequency.
            // In this case, and if the frequency.start is 0, this results in the full frequency range.
            for (size_t i{}; i < N; ++i) {
                const auto max_sample_size = shape[i] / 2 + 1;
                const auto frequency_stop =
                    fftfreq_range.stop <= 0 ?
                    noa::fft::highest_fftfreq<coord_type>(shape[i]) :
                    fftfreq_range.stop;
                m_fftfreq_step[i] = Linspace{
                    .start = fftfreq_range.start,
                    .stop = frequency_stop,
                    .endpoint = fftfreq_range.endpoint
                }.for_size(max_sample_size).step;
            }
            if constexpr (N == 1)
                m_fftfreq_start = fftfreq_range.start;
        }

    public:
        template<nt::same_as<index_type>... I> requires (sizeof...(I) == N)
        NOA_HD void operator()(
            index_type batch,
            I... output_indices
        ) const {
            const auto frequency = noa::fft::index2frequency<IS_DST_CENTERED, IS_RFFT>(Vec{output_indices...}, m_shape);
            auto fftfreq = coord_nd_type::from_vec(frequency) * m_fftfreq_step;
            if constexpr (N == 1)
                fftfreq += m_fftfreq_start;

            // TODO Add frequency cutoff?

            auto ctf = m_ctf[batch].value_at([&] {
                if constexpr (N == 1)
                    return fftfreq[0];
                else if constexpr ((N == 2 and IS_ISOTROPIC) or N == 3)
                    return norm(fftfreq);
                else // N == 2 anisotropic
                    return fftfreq;
            }());
            if (m_ctf_squared)
                ctf *= ctf;
            else if (m_ctf_abs)
                ctf = abs(ctf);

            if constexpr (HAS_INPUT) {
                const auto input_indices = noa::fft::remap_indices<REMAP, true>(Vec{output_indices...}, m_shape);
                m_output(batch, output_indices...) = cast_or_abs_squared<output_value_type>(
                    m_input(input_indices.push_front(batch)) * static_cast<input_real_type>(ctf));
            } else {
                m_output(batch, output_indices...) = static_cast<output_value_type>(ctf);
            }
        }

    private:
        ctf_parameter_type m_ctf;
        output_type m_output;
        shape_type m_shape;
        NOA_NO_UNIQUE_ADDRESS input_type m_input{};
        coord_nd_type m_fftfreq_step;
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_fftfreq_start;
        bool m_ctf_abs;
        bool m_ctf_squared;
    };
}

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
        const CTF& ctf,
        const noa::Linspace<f64>& fftfreq_range
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

        check(expected_shape.ndim() == 1 or allclose(fftfreq_range.start, 0.),
              "For multidimensional cases, the starting fftfreq should be 0, but got {}", fftfreq_range.start);
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
        /// Frequency range of the input and output, from the zero, along the cartesian axes.
        /// If the end is negative or zero, it is set to the highest frequencies for the given dimensions,
        /// i.e. the entire rfft/fft range is selected. For even dimensions, this is equivalent to {0, 0.5}.
        /// For 2d and 3d cases, the start should be 0, otherwise, an error will be thrown.
        Linspace<f64> fftfreq_range{.start = 0, .stop = -1, .endpoint = true};

        /// Whether the absolute of the ctf should be computed.
        bool ctf_abs{};

        /// Whether the square of the ctf should be computed.
        bool ctf_squared{};
    };

    /// Computes isotropic CTF(s) over entire FFT or rFFT spectrum or over a specific frequency range (see options).
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
        guts::ctf_check_parameters<REMAP>(Empty{}, output, shape, ctf, options.fftfreq_range);
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
                auto op = op_t(
                    {}, output_accessor, shape.filter(index), guts::extract_ctf(ctf),
                    options.ctf_abs, options.ctf_squared,
                    options.fftfreq_range.as<coord_t>()
                );
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
                auto op = op_t(
                    {}, output_accessor, shape.filter(2, 3), guts::extract_ctf(ctf),
                    options.ctf_abs, options.ctf_squared,
                    options.fftfreq_range.as<coord_t>()
                );
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
                auto op = op_t(
                    {}, output_accessor, shape.pop_front(), guts::extract_ctf(ctf),
                    options.ctf_abs, options.ctf_squared,
                    options.fftfreq_range.as<coord_t>()
                );
                auto iwise_shape = shape;
                if constexpr (REMAP.is_hx2hx())
                    iwise_shape = iwise_shape.rfft();
                return iwise(iwise_shape, device, op, std::forward<Output>(output), std::forward<CTF>(ctf));
            }
        }
    }

    /// Computes isotropic CTF(s).
    /// \tparam REMAP       Remapping operation. Should be H2H, HC2H, H2HC, HC2HC, F2F, FC2F, F2FC or FC2FC.
    /// \param[in] input    {1|2|3}d array(s) to multiply with the CTF.
    ///                     If empty, the CTF is directly written to the output.
    ///                     If complex and the output is real, the power spectrum is saved.
    /// \param[out] output  {1|2|3}d CTF or CTF-multiplied array, depending on the input.
    ///                     If no remapping is done, it can be equal to the input.
    /// \param shape        Logical BDHW shape.
    /// \param[in] ctf      Isotropic CTF. A contiguous vector of CTFs can be passed. In this case, there
    ///                     should be one CTF per output batch. If a single value is passed, it is applied
    ///                     to every batch.
    /// \param options      Spectrum and CTF options.
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

        guts::ctf_check_parameters<REMAP>(input, output, shape, ctf, options.fftfreq_range);

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
                auto op = op_t(
                    input_accessor_t(input.get(), input_strides.filter(0, index)),
                    output_accessor_t(output.get(), output.strides().filter(0, index)),
                    shape.filter(index), guts::extract_ctf(ctf),
                    options.ctf_abs, options.ctf_squared,
                    options.fftfreq_range.as<coord_t>()
                );
                return iwise(
                    iwise_shape, device, op,
                    std::forward<Input>(input),
                    std::forward<Output>(output),
                    std::forward<CTF>(ctf)
                );
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
                auto op = op_t(
                    input_accessor_t(input.get(), input_strides.filter(0, 2, 3)),
                    output_accessor_t(output.get(), output_strides.filter(0, 2, 3)),
                    shape.filter(2, 3), guts::extract_ctf(ctf),
                    options.ctf_abs, options.ctf_squared,
                    options.fftfreq_range.as<coord_t>()
                );
                auto iwise_shape = shape.filter(0, 2, 3);
                if constexpr (REMAP.is_xx2hx())
                    iwise_shape = iwise_shape.rfft();
                return iwise(
                    iwise_shape, device, op,
                    std::forward<Input>(input),
                    std::forward<Output>(output),
                    std::forward<CTF>(ctf)
                );
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
                auto op = op_t(
                    input_accessor_t(input.get(), input_strides),
                    output_accessor_t(output.get(), output_strides),
                    shape.pop_front(), guts::extract_ctf(ctf),
                    options.ctf_abs, options.ctf_squared,
                    options.fftfreq_range.as<coord_t>()
                );
                auto iwise_shape = shape;
                if constexpr (REMAP.is_xx2hx())
                    iwise_shape = iwise_shape.rfft();
                return iwise(
                    iwise_shape, device, op,
                    std::forward<Input>(input),
                    std::forward<Output>(output),
                    std::forward<CTF>(ctf)
                );
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
        guts::ctf_check_parameters<REMAP>(Empty{}, output, shape, ctf, options.fftfreq_range);

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
        auto op = op_t(
            {}, output_accessor, shape.filter(2, 3), guts::extract_ctf(ctf),
            options.ctf_abs, options.ctf_squared,
            options.fftfreq_range.as<coord_t>()
        );

        auto iwise_shape = shape.filter(0, 2, 3);
        if constexpr (REMAP.is_xx2hx())
            iwise_shape = iwise_shape.rfft();
        iwise(iwise_shape, output.device(), op, std::forward<Output>(output), std::forward<CTF>(ctf));
    }

    /// Computes the anisotropic CTF(s).
    /// \tparam REMAP       Remapping operation. Should be H2H, HC2H, H2HC, HC2HC, F2F, FC2F, F2FC or FC2FC.
    /// \param[in] input    2d array to multiply with the CTF.
    ///                     If empty, the CTF is directly written to the output.
    ///                     If complex and the output is real, the power spectrum is saved.
    /// \param[out] output  2d CTF or CTF-multiplied array.
    ///                     If no remapping is done, it can be equal to the input.
    /// \param shape        Logical BDHW shape.
    /// \param[in] ctf      Anisotropic CTF(s). A contiguous vector of CTFs can be passed. In this case, there
    ///                     should be one CTF per output batch. If a single value is passed, it is applied
    ///                     to every batch.
    /// \param options      Spectrum and CTF options.
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

        guts::ctf_check_parameters<REMAP>(input, output, shape, ctf, options.fftfreq_range);

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
        auto op = op_t(
            input_accessor_t(input.get(), input_strides.filter(0, 2, 3)),
            output_accessor_t(output.get(), output_strides.filter(0, 2, 3)),
            shape.filter(2, 3), guts::extract_ctf(ctf),
            options.ctf_abs, options.ctf_squared,
            options.fftfreq_range.as<coord_t>()
        );

        auto iwise_shape = shape.filter(0, 2, 3);
        if constexpr (REMAP.is_xx2hx())
            iwise_shape = iwise_shape.rfft();
        iwise(iwise_shape, output.device(), op,
              std::forward<Input>(input),
              std::forward<Output>(output),
              std::forward<CTF>(ctf));
    }
}
