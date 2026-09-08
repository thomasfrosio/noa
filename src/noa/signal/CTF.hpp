#pragma once

#include "noa/fft/core/Frequency.hpp"
#include "noa/fft/core/Transform.hpp"
#include "noa/runtime/Array.hpp"
#include "noa/runtime/Iwise.hpp"
#include "noa/runtime/core/Iwise.hpp"
#include "noa/signal/core/CTF.hpp"

namespace noa::signal::details {
    // Index-wise operator, to compute/apply CTFs to spectra with rank 1, 2 or 3.
    // If the input is valid, input*ctf->output is computed.
    // If the input is complex and the output is real, the power spectrum is saved.
    // If the input is empty, ctf->output is computed.
    template<nf::Layout REMAP, usize B, usize R,
             nt::any_of<f32, f64> Coord,
             nt::sinteger Index,
             nt::readable_nd_or_empty<B + R> Input,
             nt::writable_nd<B + R> Output,
             nt::readable_nd<B> BatchedCTF>
    class CTF {
    public:
        static_assert(REMAP.is_any(nf::Layout::H2H, nf::Layout::HC2HC, nf::Layout::HC2H, nf::Layout::H2HC,
                                   nf::Layout::F2F, nf::Layout::FC2FC, nf::Layout::FC2F, nf::Layout::F2FC));

        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::value_type_t<input_type>;
        using input_real_type = nt::value_type_t<input_value_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static_assert(nt::spectrum_types<input_value_type, output_value_type> or
                      (nt::empty<input_type> and nt::real_or_complex<output_value_type>));

        using ctf_batched_type = BatchedCTF;
        using ctf_type = nt::value_type_t<ctf_batched_type>;
        static_assert(nt::ctf<ctf_type>);

        static constexpr bool HAS_INPUT = not nt::empty<input_type>;
        static constexpr bool IS_RFFT = REMAP.is_hx2hx();
        static constexpr bool IS_DST_CENTERED = REMAP.is_xx2xc();
        static constexpr bool IS_ISOTROPIC = nt::ctf_isotropic<ctf_type>;

        using index_type = Index;
        using coord_type = Coord;
        using shape_nd_type = Shape<index_type, R>;
        using shape_type = Shape<index_type, R - IS_RFFT>;
        using coord_nd_type = Vec<coord_type, R>;
        using coord2_type = Vec<coord_type, 2>;
        using coord_or_empty_type = std::conditional_t<R == 1, coord_type, Empty>;

    public:
        constexpr CTF(
            const input_type& input,
            const output_type& output,
            const shape_nd_type& shape,
            const ctf_batched_type& ctf,
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
            for (usize i{}; i < R; ++i) {
                const auto max_sample_size = shape[i] / 2 + 1;
                const auto frequency_stop =
                    fftfreq_range.stop <= 0 ?
                    nf::highest_fftfreq<coord_type>(shape[i]) :
                    fftfreq_range.stop;
                m_fftfreq_step[i] = Linspace{
                    .start = fftfreq_range.start,
                    .stop = frequency_stop,
                    .endpoint = fftfreq_range.endpoint
                }.for_size(max_sample_size).step;
            }
            if constexpr (R == 1)
                m_fftfreq_start = fftfreq_range.start;
        }

    public:
        NOA_HD void operator()(const Vec<index_type, B + R>& output_batched_indices) const {
            const auto& [batches, output_indices] = output_batched_indices.template split<B>();
            const auto frequency = nf::index2frequency<IS_DST_CENTERED, IS_RFFT>(output_indices, m_shape);
            auto fftfreq = coord_nd_type::from_vec(frequency) * m_fftfreq_step;
            if constexpr (R == 1)
                fftfreq += m_fftfreq_start;

            // TODO Add frequency cutoff?

            auto ctf = m_ctf(batches).value_at([&] {
                if constexpr (R == 1)
                    return fftfreq[0];
                else if constexpr ((R == 2 and IS_ISOTROPIC) or R == 3)
                    return norm(fftfreq);
                else // N == 2 anisotropic
                    return fftfreq;
            }());
            if (m_ctf_squared)
                ctf *= ctf;
            else if (m_ctf_abs)
                ctf = abs(ctf);

            if constexpr (HAS_INPUT) {
                const auto input_indices = nf::remap_indices<REMAP, true>(output_indices, m_shape);
                m_output[output_batched_indices] = cast_or_abs_squared<output_value_type>(
                    m_input[input_indices.push_front(batches)] * static_cast<input_real_type>(ctf));
            } else {
                m_output[output_batched_indices] = static_cast<output_value_type>(ctf);
            }
        }

    private:
        ctf_batched_type m_ctf;
        output_type m_output;
        shape_type m_shape;
        NOA_NO_UNIQUE_ADDRESS input_type m_input{};
        coord_nd_type m_fftfreq_step;
        NOA_NO_UNIQUE_ADDRESS coord_or_empty_type m_fftfreq_start;
        bool m_ctf_abs;
        bool m_ctf_squared;
    };

    template<typename CTF, usize N, usize R, usize B = nt::array_size_v<CTF>>
    concept array_or_ctf_isotropic =
        nt::ctf_isotropic<CTF> or
        (nt::array<CTF> and nt::ctf_isotropic<nt::value_type_t<CTF>> and B >= 1 and B < N and (R == 0 or N - B == R));

    template<typename CTF, usize N, usize R, usize B = nt::array_size_v<CTF>>
    concept array_or_ctf_anisotropic =
        nt::ctf_anisotropic<CTF> or
        (nt::array<CTF> and nt::ctf_anisotropic<nt::value_type_t<CTF>> and B >= 1 and B < N and (R == 0 or N - B == R));

    template<nf::Layout REMAP, typename Input, typename Output, typename CTF, bool ISOTROPIC, usize R>
    concept ctfable =
        (REMAP.is_hx2hx() or REMAP.is_fx2fx()) and
        ((nt::empty<std::decay_t<Input>> and nt::array_decay_with_spectrum_types<Output, Output>) or
         (nt::array_decay_with_spectrum_types<Input, Output> and nt::array_decay_with_same_nd<Input, Output>)) and
        ((ISOTROPIC and array_or_ctf_isotropic<std::decay_t<CTF>, nt::array_size_v<Output>, R>) or
         (not ISOTROPIC and array_or_ctf_anisotropic<std::decay_t<CTF>, nt::array_size_v<Output>, R>));

    template<typename CTF>
    constexpr auto extract_ctf(const CTF& ctf) {
        if constexpr (nt::array<CTF>) {
            using accessor_t = AccessorRestrictContiguous<nt::const_value_type_t<CTF>, nt::array_size_v<CTF>, isize>;
            return accessor_t(ctf.get(), ctf.strides());
        } else {
            return AccessorValue(ctf);
        }
    }

    // Use a function and not a lambda because of bug with the if-constexpr in templated lambda, which tries to
    // instantiate the branch even if it's false, and in this case not compiling because ctf.shape() doesn't exist...
    template<nf::Layout REMAP, usize R, typename Input, typename Output, usize N, typename CTF, typename Options>
    void launch_ctf_iwise(
        Input&& input,
        Output&& output,
        Shape<isize, N> shape,
        CTF&& ctf,
        const Options& options
    ) {
        if constexpr (nt::ctf_anisotropic<std::decay_t<CTF>> and R != 2) {
            panic("Only arrays with rank=2 are supported with anisotropic CTFs, but got rank={} (shape={})", R, shape);
        } else if constexpr (nt::array_decay<CTF> and nt::ctf_anisotropic<nt::value_type_t<CTF>> and R != 2) {
            panic("Only arrays with rank=2 are supported with anisotropic CTFs, but got rank={} (shape={}, ctf:shape={})",
                  R, shape, ctf.shape());
        } else {
            check(R == 1 or allclose(options.fftfreq_range.start, 0.),
                  "For rank > 1, the starting fftfreq should be 0, but got {}",
                  options.fftfreq_range.start);

            // B can be 0 for the case with a single CTF.
            // If an array of CTFs is passed, B >= 1.
            static_assert(N >= R);
            constexpr auto B = N - R;

            using ivalue_t = nt::const_value_type_t<Input>;
            using ovalue_t = nt::value_type_t<Output>;
            using coord_t = nt::value_type_twice_t<CTF>;
            using ctf_t = decltype(details::extract_ctf(ctf));

            using iaccessor_t = std::conditional_t<nt::array_decay<Input>, Accessor<ivalue_t, N, isize>, Empty>;
            using oaccessor_t = Accessor<ovalue_t, N, isize>;
            using op_t = details::CTF<REMAP, B, R, coord_t, isize, iaccessor_t, oaccessor_t, ctf_t>;
            auto iaccessor_n = iaccessor_t{};
            if constexpr (nt::array_decay<Input>)
                iaccessor_n = input.span().accessor();
            auto accessor_n = output.span();
            auto shape_r = shape.template pop_front<B>();
            auto op = op_t(
                iaccessor_n, accessor_n.accessor(), shape_r, details::extract_ctf(ctf),
                options.ctf_abs, options.ctf_squared,
                options.fftfreq_range.template as<coord_t>()
            );
            return iwise(accessor_n.shape(), output.device(), op, NOA_FWD(output), NOA_FWD(ctf));
        }
    }

    template<nf::Layout REMAP, usize RANK, bool ISOTROPIC,
             typename Input, typename Output, usize N, typename CTF, typename Options>
        requires details::ctfable<REMAP, Input, Output, CTF, ISOTROPIC, RANK>
    void launch_ctf(
        Input&& input,
        Output&& output,
        Shape<isize, N> shape,
        CTF&& ctf,
        const Options& options
    ) {
        if constexpr (nt::array_decay<Input>)
            if (input.is_empty())
                return launch_ctf<REMAP.erase_input(), RANK, ISOTROPIC, Empty>(
                    {}, NOA_FWD(output), shape, NOA_FWD(ctf), options);

        check(not output.is_empty(), "Empty array detected");
        const auto expected_output_shape = REMAP.is_xx2hx() ? shape.rfft() : shape;
        check(output.shape() == expected_output_shape,
              "The output shape doesn't match the expected shape. Got output:shape={} and expected_shape={}",
              output.shape(), expected_output_shape);

        if constexpr (nt::array_decay<Input>) {
            const auto expected_input_shape = REMAP.is_hx2xx() ? shape.rfft() : shape;
            check(input.shape() == expected_input_shape,
                  "The input shape doesn't match the expected shape. Got input:shape={} and expected_shape={}",
                  input.shape(), expected_input_shape);
            check(input.device() == output.device(),
                  "The input and output arrays must be on the same device, but got input:device={} and output:device={}",
                  input.device(), output.device());
            check(not REMAP.has_layout_change() or not noa::are_overlapped(input, output),
                  "This function cannot execute in-place with a remapping");
        }

        if constexpr (nt::array_decay<CTF>) {
            constexpr usize B = nt::array_size_v<CTF>;
            const auto expected_ctf_shape = shape.template pop_back<N - B>();
            check(not ctf.is_empty() and ctf.is_contiguous() and ctf.shape() == expected_ctf_shape,
                  "The CTFs should be specified as a contiguous array matching the batch axes of the output array, but got ctf:shape={}, ctf:strides={} and output:shape:batches={}",
                  ctf.shape(), ctf.strides(), expected_ctf_shape);
            check(ctf.device() == output.device(),
                  "The ctf and output arrays must be on the same device, but got ctf:device={} and output:device={}",
                  ctf.device(), output.device());
        }

        if constexpr (nt::array_decay<CTF>) {
            constexpr auto R = N - nt::array_size_v<CTF>;
            static_assert(RANK == 0 or RANK == R);
            check(options.rank == 0 or options.rank == R);
            return launch_ctf_iwise<REMAP, R>(NOA_FWD(input), NOA_FWD(output), shape, NOA_FWD(ctf), options);
        } else {
            if constexpr (RANK == 0) {
                const auto rank = shape.rank_checked(options.rank);
                if (rank == 1)
                    return launch_ctf_iwise<REMAP, 1>(NOA_FWD(input), NOA_FWD(output), shape, NOA_FWD(ctf), options);
                if constexpr (N >= 2)
                    if (rank == 2)
                        return launch_ctf_iwise<REMAP, 2>(NOA_FWD(input), NOA_FWD(output), shape, NOA_FWD(ctf), options);
                if constexpr (N >= 3)
                    if (rank == 3)
                        return launch_ctf_iwise<REMAP, 3>(NOA_FWD(input), NOA_FWD(output), shape, NOA_FWD(ctf), options);
                panic();
            } else {
                return launch_ctf_iwise<REMAP, RANK>(NOA_FWD(input), NOA_FWD(output), shape, NOA_FWD(ctf), options);
            }
        }
    }
}

namespace noa::signal {
    struct CTFOptions {
        /// Rank of the transform.
        /// Only used if a single CTF value is passed.
        /// See Shape::rank_checked for more details.
        usize rank{};

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

    /// Computes isotropic CTF(s) over entire FFT or rFFT spectra, or over a specific frequency range (see options).
    /// \tparam RANK:
    ///     Compile time rank of the spectra.
    ///     If zero, then code for rank 1, 2, and 3, are all generated, and options.rank is used at runtime.
    /// \tparam REMAP:
    ///     Output layout.
    ///     Should be H(C)2H(C) or F(C)2F(C).
    /// \param[in] input:
    ///     Input spectra.
    ///     If Empty or if empty array, the CTF is directly written to the output.
    ///     Otherwise, it should be an array with the same shape as the output (accounting for REMAP).
    ///     If complex and the output is real, the power spectrum is saved.
    /// \param[out] output:
    ///     Simulated CTF(s) of shape ((B..)R..), where (R..) is DHW, HW or W.
    /// \param shape:
    ///     Logical shape of the output.
    /// \param[in] ctf:
    ///     Isotropic CTF(s).
    ///   - If a single CTF value is passed, the spectrum rank is set by RANK or options.rank.
    ///   - If a contiguous array is passed, it should be of shape (B..), matching the batch dimensions of the output.
    ///     In this case, the rank is set by the number of remaining dimensions (R..) (matching RANK, if non zero) and
    ///     options.rank is ignored. This also implies that if an array is passed, the input and output must be batched.
    /// \param options
    ///     Spectrum and CTF options.
    template<nf::Layout REMAP, usize RANK = 0, typename Input = Empty, typename Output, usize N, typename CTF>
    void ctf_isotropic(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        CTF&& ctf,
        const CTFOptions& options = {}
    ) {
        details::launch_ctf<REMAP, RANK, true>(NOA_FWD(input), NOA_FWD(output), shape, NOA_FWD(ctf), options);
    }

    template<nf::Layout REMAP, typename Input = Empty, typename Output, usize N, typename CTF>
    void ctf_isotropic_1d(
        Input&& input, Output&& output, const Shape<isize, N>& shape, CTF&& ctf, CTFOptions options = {}
    ) {
        options.rank = 1;
        details::launch_ctf<REMAP, 1, true>(NOA_FWD(input), NOA_FWD(output), shape, NOA_FWD(ctf), options);
    }
    template<nf::Layout REMAP, typename Input = Empty, typename Output, usize N, typename CTF>
    void ctf_isotropic_2d(
        Input&& input, Output&& output, const Shape<isize, N>& shape, CTF&& ctf, CTFOptions options = {}
    ) {
        options.rank = 2;
        details::launch_ctf<REMAP, 2, true>(NOA_FWD(input), NOA_FWD(output), shape, NOA_FWD(ctf), options);
    }
    template<nf::Layout REMAP, typename Input = Empty, typename Output, usize N, typename CTF>
    void ctf_isotropic_3d(
        Input&& input, Output&& output, const Shape<isize, N>& shape, CTF&& ctf, CTFOptions options = {}
    ) {
        options.rank = 3;
        details::launch_ctf<REMAP, 3, true>(NOA_FWD(input), NOA_FWD(output), shape, NOA_FWD(ctf), options);
    }

    /// Computes isotropic CTF(s) over entire FFT or rFFT spectra, or over a specific frequency range (see options).
    /// Same as the overload above, but with an empty input.
    template<nf::Layout REMAP, usize RANK = 0, typename Output, usize N, typename CTF>
    void ctf_isotropic(
        Output&& output,
        const Shape<isize, N>& shape,
        CTF&& ctf,
        const CTFOptions& options = {}
    ) {
        details::launch_ctf<REMAP, RANK, true, Empty>({}, NOA_FWD(output), shape, NOA_FWD(ctf), options);
    }

    template<nf::Layout REMAP, typename Output, usize N, typename CTF>
    void ctf_isotropic_1d(
        Output&& output, const Shape<isize, N>& shape, CTF&& ctf, CTFOptions options = {}
    ) {
        options.rank = 1;
        details::launch_ctf<REMAP, 1, true, Empty>({}, NOA_FWD(output), shape, NOA_FWD(ctf), options);
    }
    template<nf::Layout REMAP, typename Output, usize N, typename CTF>
    void ctf_isotropic_2d(
        Output&& output, const Shape<isize, N>& shape, CTF&& ctf, CTFOptions options = {}
    ) {
        options.rank = 2;
        details::launch_ctf<REMAP, 2, true, Empty>({}, NOA_FWD(output), shape, NOA_FWD(ctf), options);
    }
    template<nf::Layout REMAP, typename Output, usize N, typename CTF>
    void ctf_isotropic_3d(
        Output&& output, const Shape<isize, N>& shape, CTF&& ctf, CTFOptions options = {}
    ) {
        options.rank = 3;
        details::launch_ctf<REMAP, 3, true, Empty>({}, NOA_FWD(output), shape, NOA_FWD(ctf), options);
    }

    /// Computes anisotropic CTF(s) over entire FFT or rFFT spectra, or over a specific frequency range (see options).
    /// Same as the isotropic overload, except that the rank of the transforms is restrict to 2.
    template<nf::Layout REMAP, usize RANK = 0, typename Input = Empty, typename Output, usize N, typename CTF>
    void ctf_anisotropic(
        Input&& input,
        Output&& output,
        const Shape<isize, N>& shape,
        CTF&& ctf,
        const CTFOptions& options = {}
    ) {
        details::launch_ctf<REMAP, RANK, false>(NOA_FWD(input), NOA_FWD(output), shape, NOA_FWD(ctf), options);
    }

    template<nf::Layout REMAP, typename Input = Empty, typename Output, usize N, typename CTF>
    void ctf_anisotropic_2d(
        Input&& input, Output&& output, const Shape<isize, N>& shape, CTF&& ctf, CTFOptions options = {}
    ) {
        options.rank = 2;
        details::launch_ctf<REMAP, 2, false>(NOA_FWD(input), NOA_FWD(output), shape, NOA_FWD(ctf), options);
    }

    /// Computes anisotropic CTF(s) over entire FFT or rFFT spectra, or over a specific frequency range (see options).
    /// Same as the overload above, but with an empty input.
    template<nf::Layout REMAP, usize RANK = 0, typename Output, usize N, typename CTF>
    void ctf_anisotropic(
        Output&& output,
        const Shape<isize, N>& shape,
        CTF&& ctf,
        const CTFOptions& options = {}
    ) {
        details::launch_ctf<REMAP, RANK, false, Empty>({}, NOA_FWD(output), shape, NOA_FWD(ctf), options);
    }

    template<nf::Layout REMAP, typename Output, usize N, typename CTF>
    void ctf_anisotropic_2d(
        Output&& output,
        const Shape<isize, N>& shape,
        CTF&& ctf,
        CTFOptions options = {}
    ) {
        options.rank = 2;
        details::launch_ctf<REMAP, 2, false, Empty>({}, NOA_FWD(output), shape, NOA_FWD(ctf), options);
    }
}
