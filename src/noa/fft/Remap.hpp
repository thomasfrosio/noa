#pragma once

#include "noa/runtime/core/Traits.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/Array.hpp"
#include "noa/runtime/Iwise.hpp"

#include "noa/fft/core/Frequency.hpp"
#include "noa/fft/core/Transform.hpp"
#include "noa/fft/core/Layout.hpp"

namespace noa::fft::details {
    template<bool INVERSE, typename T, usize R, usize N>
        requires (N >= R - 1)
    NOA_FHD constexpr auto remap_rfftshift(const Vec<T, R>& output_indices, const Shape<T, N>& shape) -> Vec<T, R> {
        Vec<T, R> input_indices;
        if constexpr (R >= 2) {
            for (usize i = 0; i < R - 1; ++i) {
                if constexpr (INVERSE)
                    input_indices[i] = ifftshift(output_indices[i], shape[i]);
                else
                    input_indices[i] = fftshift(output_indices[i], shape[i]);
            }
        }
        input_indices[R - 1] = output_indices[R - 1];
        return input_indices;
    }

    template<Layout REMAP, usize B, usize R,
             nt::sinteger Index,
             nt::writable_nd<B + R> Output>
    requires nt::numeric<nt::value_type_t<Output>>
    class FourierRemapInplace {
    public:
        using index_type = Index;
        using shape_type = Shape<index_type, R>;
        using shape_no_width_type = Shape<index_type, R - 1>;
        using output_type = Output;
        using value_type = nt::mutable_value_type_t<output_type>;

        constexpr FourierRemapInplace(
            const output_type& array,
            const shape_type& shape
        ) : m_array(array),
            m_shape(shape.pop_back()) {}

        constexpr void operator()(const Vec<index_type, B + R>& output_batched_indices) const {
            const auto [batches, output_indices] = output_batched_indices.template split<B>();
            Vec<index_type, R> input_indices;
            if constexpr (REMAP == Layout::H2HC) {
                input_indices = remap_rfftshift<true>(output_indices, m_shape);
            } else if constexpr (REMAP == Layout::HC2H) {
                input_indices = remap_rfftshift<false>(output_indices, m_shape);
            } else {
                static_assert(nt::always_false<output_type>);
            }

            auto& o = m_array(batches.push_back(output_indices));
            auto& i = m_array(batches.push_back(input_indices));

            // Swap
            auto tmp = o;
            o = i;
            i = tmp;
        }

    private:
        output_type m_array;
        shape_no_width_type m_shape;
    };

    template<Layout REMAP, usize B, usize R,
             nt::sinteger Index,
             nt::readable_nd<B + R> Input,
             nt::writable_nd<B + R> Output>
    class FourierRemap {
    public:
        using index_type = Index;
        using shape_type = Shape<index_type, R>;
        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;
        static_assert(nt::compatible_or_spectrum_types<input_value_type, output_value_type>);

        constexpr FourierRemap(
            const input_type& input,
            const output_type& output,
            const shape_type& shape
        ) : m_input(input),
            m_output(output),
            m_shape(shape) {}

        constexpr void operator()(const Vec<index_type, B + R>& output_batched_indices) const {
            const auto [batches, output_indices] = output_batched_indices.template split<B>();
            auto input = m_input[batches];
            auto output = m_output[batches];

            if constexpr (REMAP == Layout::HC2H) {
                const auto input_indices = remap_rfftshift<false>(output_indices, m_shape);
                output(output_indices) = cast_or_abs_squared<output_value_type>(input(input_indices));

            } else if constexpr (REMAP == Layout::H2HC) {
                const auto input_indices = remap_rfftshift<true>(output_indices, m_shape);
                output(output_indices) = cast_or_abs_squared<output_value_type>(input(input_indices));

            } else if constexpr (REMAP == Layout::FC2F) {
                const auto input_indices = fftshift(output_indices, m_shape);
                output(output_indices) = cast_or_abs_squared<output_value_type>(input(input_indices));

            } else if constexpr (REMAP == Layout::F2FC) {
                const auto input_indices = ifftshift(output_indices, m_shape);
                output(output_indices) = cast_or_abs_squared<output_value_type>(input(input_indices));

            } else if constexpr (REMAP == Layout::F2H) {
                output(output_indices) = cast_or_abs_squared<output_value_type>(input(output_indices)); // copy

            } else if constexpr (REMAP == Layout::F2HC) {
                const auto input_indices = remap_rfftshift<true>(output_indices, m_shape);
                output(output_indices) = cast_or_abs_squared<output_value_type>(input(input_indices));

            } else if constexpr (REMAP == Layout::FC2H) {
                const auto input_indices = fftshift(output_indices, m_shape);
                output(output_indices) = cast_or_abs_squared<output_value_type>(input(input_indices));

            } else if constexpr (REMAP == Layout::FC2HC) {
                const auto input_indices = output_indices.template set<R - 1>(fftshift(output_indices[R - 1], m_shape[R - 1]));
                output(output_indices) = cast_or_abs_squared<output_value_type>(input(input_indices));

            } else if constexpr (REMAP == Layout::HC2F or REMAP == Layout::HC2FC) {
                input_value_type value;
                if (output_indices[R - 1] < m_shape[R - 1] / 2 + 1) {
                    // Copy first non-redundant half.
                    const auto input_indices = remap_rfftshift<false>(output_indices, m_shape);
                    value = input(input_indices);
                } else {
                    // Rebase to the symmetric row in the non-redundant input.
                    // Then copy in reverse order.
                    Vec<index_type, R> input_indices;
                    for (usize i = 0; i < R - 1; ++i)
                        input_indices[i] = fftshift(output_indices[i] != 0 ? m_shape[i] - output_indices[i] : output_indices[i], m_shape[i]);
                    input_indices[R - 1] = m_shape[R - 1] - output_indices[R - 1];
                    value = input(input_indices);
                    if constexpr (nt::complex<output_value_type>)
                        value = conj(value);
                }

                if constexpr (REMAP == Layout::HC2F) {
                    output(output_indices) = cast_or_abs_squared<output_value_type>(value);
                } else { // HC2FC: HC2F -> F2FC
                    const auto o = fftshift(output_indices, m_shape);
                    output(o) = cast_or_abs_squared<output_value_type>(value);
                }

            } else if constexpr (REMAP == Layout::H2F or REMAP == Layout::H2FC) {
                input_value_type value;
                if (output_indices[R - 1] < m_shape[R - 1] / 2 + 1) {
                    // Copy first non-redundant half.
                    value = input(output_indices);
                } else {
                    // Rebase to the symmetric row in the non-redundant input.
                    // Then copy in reverse order.
                    Vec<index_type, R> input_indices;
                    for (usize i = 0; i < R - 1; ++i)
                        input_indices[i] = output_indices[i] != 0 ? m_shape[i] - output_indices[i] : output_indices[i];
                    input_indices[R - 1] = m_shape[R - 1] - output_indices[R - 1];
                    value = input(input_indices);
                    if constexpr (nt::complex<output_value_type>)
                        value = conj(value);
                }

                if constexpr (REMAP == Layout::H2F) {
                    output(output_indices) = cast_or_abs_squared<output_value_type>(value);
                } else { // H2FC: H2F -> F2FC
                    const auto o = fftshift(output_indices, m_shape);
                    output(o) = cast_or_abs_squared<output_value_type>(value);
                }
            } else {
                static_assert(nt::always_false<input_type>);
            }
        }

    private:
        input_type m_input;
        output_type m_output;
        shape_type m_shape;
    };
}

namespace noa::fft {
    struct RemapOptions {
        /// Rank of the transform.
        /// This determines which dimensions are considered batch dimensions.
        /// Batch dimensions are collapsed to a single dimensions and should therefore be collapsible.
        /// See Shape::rank_checked and ranked_shape for more details.
        usize rank{};
    };

    /// Remaps FFT(s).
    /// \tparam RANK:
    ///     Rank of the transform.
    ///     This determines which dimensions are considered batch dimensions.
    ///     Should be 1, 2, 3, or 0. If 0, fallback to runtime options.rank, in which case the batch dimensions
    ///     are collapsed to a single dimension and should therefore be collapsible.
    /// \param remap:
    ///     Remapping operation.
    /// \param[in] input:
    ///     Input FFT to remap.
    ///     ((B..,)R), where R is W (RANK=1), HW (RANK=2), or DHW (RANK=3).
    /// \param[out] output:
    ///     Remapped FFT.
    ///     ((B..,)R), where R is W (RANK=1), HW (RANK=2), or DHW (RANK=3).
    /// \param shape:
    ///     Logical shape.
    /// \param options:
    ///     Remap options.
    ///
    /// \note If \p remap is \c h2hc, \p input can be equal to \p output, iff the height and depth are even or 1.
    /// \note This function can also perform a cast or compute the power spectrum of the input, depending on the
    ///       input and output types.
    ///
    /// \bug See noa/docs/040_fft_layouts.md.
    ///      The rfft<->fft remapping should be used with care. If \p input has an anisotropic field, with an
    ///      anisotropic angle that is not a multiple of \c pi/2, and has even-sized dimensions, the remapping will
    ///      not be correct for the real-Nyquist frequencies, except the ones on the DHW axes. This situation is rare
    ///      but can happen if a geometric scaling factor is applied to the \p input, or with CTFs with astigmatic
    ///      defocus. While this could be fixed for the fft->rfft remapping, it cannot be fixed for the opposite
    ///      rfft->fft. This shouldn't be a big issue since, 1) these remapping should only be done for debugging and
    ///      visualization anyway, 2) if the remap is done directly after a dft, it will work since the field is
    ///      isotropic in this case, and 3) the problematic frequencies are past the Nyquist frequency, so lowpass
    ///      filtering to Nyquist (fftfreq=0.5) fixes this issue.
    template<usize RANK = 0, nt::readable_array_decay Input, nt::writable_array_decay Output, usize N>
        requires (nt::array_decay_with_compatible_or_spectrum_types<Input, Output> and
                  nt::array_size_v<Input> == N and nt::array_size_v<Output> == N)
    void remap(Layout remap, Input&& input, Output&& output, Shape<isize, N> shape, RemapOptions options = {}) {
        using input_value_t = nt::mutable_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;

        // Check shape.
        check(not input.is_empty() and not output.is_empty(), "Empty array detected");
        check(input.shape() == (remap.is_fx2xx() ? shape : shape.rfft()),
              "Given the {} remap, the input fft is expected to have a physical shape of {}, but got {}",
              remap, remap.is_fx2xx() ? shape : shape.rfft(), input.shape());
        check(output.shape() == (remap.is_xx2fx() ? shape : shape.rfft()),
              "Given the {} remap, the output fft is expected to have a physical shape of {}, but got {}",
              remap, remap.is_xx2fx() ? shape : shape.rfft(), output.shape());

        // Shortcut if there's no remapping.
        const bool is_inplace = are_overlapped(input, output);
        if (not remap.has_layout_change()) {
            if constexpr (nt::same_as<input_value_t, output_value_t>) {
                if (not is_inplace)
                    copy(std::forward<Input>(input), std::forward<Output>(output));
            } else {
                cast(std::forward<Input>(input), std::forward<Output>(output));
            }
            return;
        }

        // Transform to BR.
        constexpr bool RUNTIME_RANK = RANK == 0;
        constexpr usize R = RUNTIME_RANK ? 3 : RANK;
        constexpr usize B = RUNTIME_RANK ? 1 : N <= RANK ? 0 : N - RANK;
        constexpr usize BR = B + R;
        auto input_br = input.span().template as_nd<BR>();
        auto output_br = output.span().template as_nd<BR>();
        auto shape_br = shape.template as_nd<BR>();
        if constexpr (RUNTIME_RANK) {
            static_assert(BR == 4);
            shape_br = shape_br.ranked(shape_br.rank_checked(options.rank));
            input_br = input_br.reshape(shape_br.template set<3>(input_br.shape()[3]));
            output_br = output_br.reshape(shape_br.template set<3>(output_br.shape()[3]));
        }

        // Special in-place rfft case:
        if (is_inplace) {
            check(remap.is_any(Layout::H2HC, Layout::HC2H),
                  "In-place remapping is not supported with {}", remap);
            check(static_cast<const void*>(input.get()) == static_cast<const void*>(output.get()) and
                  input.strides() == output.strides(),
                  "Arrays are overlapping (which triggers in-place remapping), but do not point to the same elements. Got input:data={}, input:strides={}, output:data={} and output:strides={}",
                  static_cast<const void*>(input.get()), input.strides(),
                  static_cast<const void*>(output.get()), output.strides());

            const auto shape_hw = shape_br.template pop_front<B + (R == 3 ? 1 : 0)>(); // FIXME why removing depth? runtime requires all DHW to be even
            for (auto e: shape_hw)
                if (e != 1 and noa::is_odd(e))
                    panic("In-place remapping requires the depth and height dimensions to have an even number of elements, but got shape_br={}", shape_br);
        }

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        using input_t = AccessorRestrict<const input_value_t, BR, isize>;
        using output_t = AccessorRestrict<output_value_t, BR, isize>;
        const auto iaccessor = input_t(input_br.get(), input_br.strides());
        const auto oaccessor = output_t(output_br.get(), output_br.strides());
        const auto shape_n = shape_br.template pop_front<B>();

        auto iwise_shape = remap.is_xx2fx() ? shape_br : shape_br.rfft();
        if constexpr (R >= 2) {
            if (is_inplace)
                iwise_shape[BR - 2] = max(iwise_shape[BR - 2] / 2, isize{1}); // iterate only through half of height
        }

        switch (remap) {
            using namespace details;
            case Layout::H2H:
            case Layout::HC2HC:
            case Layout::F2F:
            case Layout::FC2FC:
                break;
            case Layout::H2HC: {
                if (is_inplace) {
                    auto op = FourierRemapInplace<Layout::H2HC, B, R, isize, output_t>(oaccessor, shape_n);
                    return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
                }
                auto op = FourierRemap<Layout::H2HC, B, R, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::HC2H: {
                if (is_inplace) {
                    auto op = FourierRemapInplace<Layout::HC2H, B, R, isize, output_t>(oaccessor, shape_n);
                    return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
                }
                auto op = FourierRemap<Layout::HC2H, B, R, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::H2F: {
                auto op = FourierRemap<Layout::H2F, B, R, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::F2H: {
                auto op = FourierRemap<Layout::F2H, B, R, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::F2FC: {
                auto op = FourierRemap<Layout::F2FC, B, R, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::FC2F: {
                auto op = FourierRemap<Layout::FC2F, B, R, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::HC2F: {
                auto op = FourierRemap<Layout::HC2F, B, R, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::F2HC: {
                auto op = FourierRemap<Layout::F2HC, B, R, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::FC2H: {
                auto op = FourierRemap<Layout::FC2H, B, R, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::FC2HC: {
                auto op = FourierRemap<Layout::FC2HC, B, R, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::HC2FC: {
                auto op = FourierRemap<Layout::HC2FC, B, R, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
            case Layout::H2FC: {
                auto op = FourierRemap<Layout::H2FC, B, R, isize, input_t, output_t>(iaccessor, oaccessor, shape_n);
                return iwise(iwise_shape, device, op, std::forward<Input>(input), std::forward<Output>(output));
            }
        }
    }
    template<typename Input, typename Output, usize N>
    void remap_1d(Layout remap, Input&& input, Output&& output, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<1>(remap, std::forward<Input>(input), std::forward<Output>(output), shape, options);
    }
    template<typename Input, typename Output, usize N>
    void remap_2d(Layout remap, Input&& input, Output&& output, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<2>(remap, std::forward<Input>(input), std::forward<Output>(output), shape, options);
    }
    template<typename Input, typename Output, usize N>
    void remap_3d(Layout remap, Input&& input, Output&& output, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<3>(remap, std::forward<Input>(input), std::forward<Output>(output), shape, options);
    }
    template<typename Input, typename Output, usize N>
    void fftshift_1d(Input&& input, Output&& output, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<1>(Layout::F2FC, std::forward<Input>(input), std::forward<Output>(output), shape, options);
    }
    template<typename Input, typename Output, usize N>
    void fftshift_2d(Input&& input, Output&& output, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<2>(Layout::F2FC, std::forward<Input>(input), std::forward<Output>(output), shape, options);
    }
    template<typename Input, typename Output, usize N>
    void fftshift_3d(Input&& input, Output&& output, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<3>(Layout::F2FC, std::forward<Input>(input), std::forward<Output>(output), shape, options);
    }
    template<typename Input, typename Output, usize N>
    void rfftshift_1d(Input&& input, Output&& output, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<1>(Layout::H2HC, std::forward<Input>(input), std::forward<Output>(output), shape, options);
    }
    template<typename Input, typename Output, usize N>
    void rfftshift_2d(Input&& input, Output&& output, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<2>(Layout::H2HC, std::forward<Input>(input), std::forward<Output>(output), shape, options);
    }
    template<typename Input, typename Output, usize N>
    void rfftshift_3d(Input&& input, Output&& output, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<3>(Layout::H2HC, std::forward<Input>(input), std::forward<Output>(output), shape, options);
    }
    template<typename Input, typename Output, usize N>
    void ifftshift_1d(Input&& input, Output&& output, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<1>(Layout::FC2F, std::forward<Input>(input), std::forward<Output>(output), shape, options);
    }
    template<typename Input, typename Output, usize N>
    void ifftshift_2d(Input&& input, Output&& output, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<2>(Layout::FC2F, std::forward<Input>(input), std::forward<Output>(output), shape, options);
    }
    template<typename Input, typename Output, usize N>
    void ifftshift_3d(Input&& input, Output&& output, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<3>(Layout::FC2F, std::forward<Input>(input), std::forward<Output>(output), shape, options);
    }
    template<typename Input, typename Output, usize N>
    void irfftshift_1d(Input&& input, Output&& output, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<1>(Layout::HC2H, std::forward<Input>(input), std::forward<Output>(output), shape, options);
    }
    template<typename Input, typename Output, usize N>
    void irfftshift_2d(Input&& input, Output&& output, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<2>(Layout::HC2H, std::forward<Input>(input), std::forward<Output>(output), shape, options);
    }
    template<typename Input, typename Output, usize N>
    void irfftshift_3d(Input&& input, Output&& output, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<3>(Layout::HC2H, std::forward<Input>(input), std::forward<Output>(output), shape, options);
    }

    /// Remaps fft(s).
    template<usize RANK = 0, nt::readable_array_decay_of_numeric Input, usize N>
        requires nt::array_decay_nd<Input, N>
    [[nodiscard]] auto remap(Layout remap, Input&& input, const Shape<isize, N>& shape, RemapOptions options = {}) {
        using value_t = nt::mutable_value_type_t<Input>;
        auto output = Array<value_t, N>(remap.is_xx2fx() ? shape : shape.rfft(), input.options());
        nf::remap<RANK>(remap, std::forward<Input>(input), output, shape, options);
        return output;
    }
    template<typename Input, usize N>
    void remap_1d(Layout remap, Input&& input, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<1>(remap, std::forward<Input>(input), shape, options);
    }
    template<typename Input, usize N>
    void remap_2d(Layout remap, Input&& input, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<2>(remap, std::forward<Input>(input), shape, options);
    }
    template<typename Input, usize N>
    void remap_3d(Layout remap, Input&& input, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<3>(remap, std::forward<Input>(input), shape, options);
    }
    template<typename Input, usize N>
    void fftshift_1d(Input&& input, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<1>(Layout::F2FC, std::forward<Input>(input), shape, options);
    }
    template<typename Input, usize N>
    void fftshift_2d(Input&& input, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<2>(Layout::F2FC, std::forward<Input>(input), shape, options);
    }
    template<typename Input, usize N>
    void fftshift_3d(Input&& input, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<3>(Layout::F2FC, std::forward<Input>(input), shape, options);
    }
    template<typename Input, usize N>
    void rfftshift_1d(Input&& input, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<1>(Layout::H2HC, std::forward<Input>(input), shape, options);
    }
    template<typename Input, usize N>
    void rfftshift_2d(Input&& input, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<2>(Layout::H2HC, std::forward<Input>(input), shape, options);
    }
    template<typename Input, usize N>
    void rfftshift_3d(Input&& input, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<3>(Layout::H2HC, std::forward<Input>(input), shape, options);
    }
    template<typename Input, usize N>
    void ifftshift_1d(Input&& input, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<1>(Layout::FC2F, std::forward<Input>(input), shape, options);
    }
    template<typename Input, usize N>
    void ifftshift_2d(Input&& input, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<2>(Layout::FC2F, std::forward<Input>(input), shape, options);
    }
    template<typename Input, usize N>
    void ifftshift_3d(Input&& input, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<3>(Layout::FC2F, std::forward<Input>(input), shape, options);
    }
    template<typename Input, usize N>
    void irfftshift_1d(Input&& input, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<1>(Layout::HC2H, std::forward<Input>(input), shape, options);
    }
    template<typename Input, usize N>
    void irfftshift_2d(Input&& input, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<2>(Layout::HC2H, std::forward<Input>(input), shape, options);
    }
    template<typename Input, usize N>
    void irfftshift_3d(Input&& input, Shape<isize, N> shape, RemapOptions options = {}) {
        noa::fft::remap<3>(Layout::HC2H, std::forward<Input>(input), shape, options);
    }
}
