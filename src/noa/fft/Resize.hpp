#pragma once

#include "noa/runtime/core/Traits.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/Resize.hpp"
#include "noa/runtime/Factory.hpp"

#include "noa/fft/core/Layout.hpp"
#include "noa/fft/core/Transform.hpp"
#include "noa/fft/core/Frequency.hpp"

// TODO Add rescale like in IMOD
// TODO Add compile-time rank (B+RANK dimensions), to not have to collapse
//      batch dimensions and keep problem to RANK-D instead of 3D.

namespace noa::fft::details {
    template<Layout LAYOUT, bool CROP,
             nt::sinteger Index, usize B, usize R,
             nt::readable_nd<B + R> Input,
             nt::writable_nd<B + R> Output>
    class FourierResize {
    public:
        using index_type = Index;
        using input_type = Input;
        using output_type = Output;
        using input_value_type = nt::mutable_value_type_t<input_type>;
        using output_value_type = nt::value_type_t<output_type>;

        static_assert(nt::compatible_or_spectrum_types<input_value_type, output_value_type>);
        static constexpr bool IS_CENTERED = LAYOUT.is_xc2xx();
        static constexpr bool IS_HALF = LAYOUT.is_hx2xx();

        using shape_no_width_type = std::conditional_t<IS_HALF, Shape<index_type, R - 1>, Empty>;
        using vec_type = std::conditional_t<not IS_HALF or IS_CENTERED, Shape<index_type, R - (IS_CENTERED and IS_HALF)>, Empty>;

        constexpr FourierResize(
            const input_type& input,
            const output_type& output,
            const Shape<index_type, R>& input_shape,
            const Shape<index_type, R>& output_shape
        ) : m_input(input),
            m_output(output)
        {
            if constexpr (not IS_CENTERED) {
                if constexpr (IS_HALF) {
                    m_input_shape = input_shape.pop_back();
                    m_output_shape = output_shape.pop_back();
                } else if constexpr (not IS_HALF and CROP) {
                    m_offset = input_shape - output_shape;
                    m_limit = (output_shape + 1) / 2;
                } else if constexpr (not IS_HALF and not CROP) {
                    m_offset = output_shape - input_shape;
                    m_limit = (input_shape + 1) / 2;
                }
            } else {
                if constexpr (not IS_HALF or R >= 2) {
                    const auto border_left = output_shape / 2 - input_shape / 2;
                    m_offset = abs(border_left.template pop_back<IS_HALF>());
                }
            }
        }

        constexpr void operator()(const Vec<index_type, B + R>& batched_indices) const {
            const auto [batches, indices] = batched_indices.template split<B>();

            if constexpr (IS_CENTERED) {
                Vec<index_type, R> new_indices;
                if constexpr (R >= 2) {
                    for (usize i{}; i < R - 1; ++i)
                        new_indices[i] = indices[i] + m_offset[i];
                }
                new_indices[R - 1] = indices[R - 1];
                if constexpr (not IS_HALF)
                    new_indices[R - 1] += m_offset[R - 1];
                if constexpr (not CROP) {
                    m_output(batches.push_back(new_indices)) =
                        cast_or_abs_squared<output_value_type>(m_input(batched_indices));
                } else {
                    m_output(batched_indices) =
                        cast_or_abs_squared<output_value_type>(m_input(batches.push_back(new_indices)));
                }
            } else {
                if constexpr (CROP and IS_HALF) {
                    Vec<index_type, R> input_indices;
                    if constexpr (R >= 2) {
                        for (usize i{}; i < R - 1; ++i)
                            input_indices[i] = indices[i] < (m_output_shape[i] + 1) / 2 ?
                                indices[i] : indices[i] + m_input_shape[i] - m_output_shape[i];
                    }
                    input_indices[R - 1] = indices[R - 1];
                    m_output(batched_indices) =
                        cast_or_abs_squared<output_value_type>(m_input(batches.push_back(input_indices)));

                } else if constexpr (not CROP and IS_HALF) {
                    Vec<index_type, R> output_indices;
                    if constexpr (R >= 2) {
                        for (usize i{}; i < R - 1; ++i)
                            output_indices[i] = indices[i] < (m_input_shape[i] + 1) / 2 ?
                                indices[i] : indices[i] + m_output_shape[i] - m_input_shape[i];
                    }
                    output_indices[R - 1] = indices[R - 1];
                    m_output(batches.push_back(output_indices)) =
                        cast_or_abs_squared<output_value_type>(m_input(batched_indices));

                } else if constexpr (CROP and not IS_HALF) {
                    Vec<index_type, R> input_indices;
                    for (usize i{}; i < R; ++i)
                        input_indices[i] = indices[i] < m_limit[i] ? indices[i] : indices[i] + m_offset[i];
                    m_output(batched_indices) =
                        cast_or_abs_squared<output_value_type>(m_input(batches.push_back(input_indices)));

                } else if constexpr (not CROP and not IS_HALF) {
                    Vec<index_type, R> output_indices;
                    for (usize i{}; i < R; ++i)
                        output_indices[i] = indices[i] < m_limit[i] ? indices[i] : indices[i] + m_offset[i];
                    m_output(batches.push_back(output_indices)) =
                        cast_or_abs_squared<output_value_type>(m_input(batched_indices));
                }
            }
        }

    private:
        input_type m_input;
        output_type m_output;
        shape_no_width_type m_input_shape{};
        shape_no_width_type m_output_shape{};
        vec_type m_offset{};
        vec_type m_limit{};
    };

    template<bool IS_CENTERED, bool IS_HALF,
             nt::sinteger Index, usize B, usize R,
             nt::writable_nd<B + R - 1> Output>
        requires (R == 2 or R == 3)
    struct FourierResizeCorrect {
        using index_type = Index;
        using output_type = Output;
        using value_type = nt::value_type_t<output_type>;

        output_type output;
        Shape<index_type, R - 1 - IS_HALF> shape{};

        static constexpr void average_pairs(const auto& span, const auto& indices, const auto& indices_pair) {
            auto& a = span.at(indices);
            auto& b = span.at(indices_pair);
            auto b_conj = b;
            if constexpr (nt::complex<value_type>)
                b_conj.imag = -b_conj.imag;
            auto avg = (a + b_conj) / 2;
            a = avg;
            if constexpr (nt::complex<value_type>)
                avg.imag = -avg.imag;
            b = avg;
        }

        constexpr void symmetrize_plane(const auto& span, const Vec<index_type, R - 1>& indices) const {
            const auto frequency = index2frequency<IS_CENTERED, IS_HALF>(indices, shape);

            // Get the Hermitian pair.
            // N = 10: [-5,-4,-3,-2,-1,+0,+1,+2,+3,+4], e.g. pair=[-5,-5], pair=[-4, 4], .., pair[0, 0]
            // N = 9:     [-4,-3,-2,-1,+0,+1,+2,+3,+4], e.g. pair=[-4, 4], ..., pair[0, 0]
            auto frequency_pair = frequency;
            if constexpr (R - 1 - IS_HALF >= 1) {
                for (usize l{}; l < R - 1 - IS_HALF; ++l) {
                    // If nyquist or zero, conjugate-average with itself, aka set imaginary to 0.
                    const auto nyquist = -(shape[l] + 1) / 2;
                    if (frequency[l] != nyquist)
                        frequency_pair[l] = -frequency[l];
                }
            }

            const auto indices_pair = frequency2index<IS_CENTERED, IS_HALF>(frequency_pair, shape);
            average_pairs(span, indices, indices_pair);
        }

        constexpr void operator()(const Vec<index_type, B + R - 1>& batched_indices) const {
            const auto [batches, indices] = batched_indices.template split<B>();
            symmetrize_plane(output[batches], indices);
        }
    };
}

namespace noa::fft {
    struct ResizeOptions {
        /// Rank of the transform. Unused if the rank is specified at compile time.
        /// This determines which dimensions are considered batch dimensions.
        /// Batch dimensions are collapsed to a single dimensions and should therefore be collapsible.
        /// See Shape::rank_checked and ranked_shape for more details.
        usize rank{};

        /// For rank 2 or 3 transforms, when cropping to a new logical even-size, the new nyquist frequencies need
        /// to be corrected to restore the Hermitian symmetry. The correction applied here consists in conjugate-
        /// averaging the Hermitian pairs (the imaginary value of frequencies without pairs is set to zero).
        /// This correction can be skipped if a lowpass filter removing Nyquist is applied before computing the c2r
        /// transform. Otherwise, without this correction, the c2r transform is implementation defined (cuFFT will
        /// likely generate a different output compared to FFTW).
        bool correct_nyquist{false};
    };

    /// Crops or zero-pads (r)FFT(s).
    /// \tparam LAYOUT
    ///     FFT layout.
    ///     Should be H2H, HC2HC, F2F or FC2FC.
    /// \tparam RANK:
    ///     Rank of the transform.
    ///     This determines which dimensions are considered batch dimensions.
    ///     Should be 1, 2, 3, or 0. If 0, fallback to runtime options.rank, in which case the batch dimensions
    ///     are collapsed to a single dimension and should therefore be collapsible.
    /// \param[in] input, input_shape:
    ///     The FFT to resize and its logical shape.
    ///     ((B..,)R), where R is W (RANK=1), HW (RANK=2), or DHW (RANK=3).
    /// \param[out] output, output_shape:
    ///     The resized FFT and its logical shape.
    ///     ((B..,)R), where R is W (RANK=1), HW (RANK=2), or DHW (RANK=3).
    ///     If real and the input is complex, the power-spectrum is computed.
    /// \param options:
    ///     Resizing options.
    template<Layout LAYOUT, usize RANK = 0, nt::readable_array_decay Input, nt::writable_array_decay Output, usize N>
        requires (nt::array_decay_with_compatible_or_spectrum_types<Input, Output> and
                  nt::array_size_v<Input> == N and
                  nt::array_size_v<Output> == N and
                  not LAYOUT.has_layout_change() and RANK <= 3)
    void resize(
        Input&& input, Shape<isize, N> input_shape,
        Output&& output, Shape<isize, N> output_shape,
        ResizeOptions options = {}
    ) {
        using input_value_t = nt::mutable_value_type_t<Input>;
        using output_value_t = nt::value_type_t<Output>;

        if (input_shape == output_shape) {
            if constexpr (nt::same_as<input_value_t, output_value_t>) {
                return copy(std::forward<Input>(input), std::forward<Output>(output));
            } else {
                return cast(std::forward<Input>(input), std::forward<Output>(output));
            }
        }

        constexpr bool IS_FULL = LAYOUT.is_fx2xx();
        constexpr bool IS_CENTERED = LAYOUT.is_xc2xx();
        check(nd::are_arrays_valid(input, output), "Empty array detected");
        check(not are_overlapped(input, output), "Input and output arrays should not overlap");
        check(input.shape() == (IS_FULL ? input_shape : input_shape.rfft()),
              "Given the {} remap, the input fft is expected to have a physical shape of {}, but got {}",
              LAYOUT, IS_FULL ? input_shape : input_shape.rfft(), input.shape());
        check(output.shape() == (IS_FULL ? output_shape : output_shape.rfft()),
              "Given the {} remap, the output fft is expected to have a physical shape of {}, but got {}",
              LAYOUT, IS_FULL ? output_shape : output_shape.rfft(), output.shape());

        const Device device = output.device();
        check(device == input.device(),
              "The input and output arrays must be on the same device, but got input:device={}, output:device={}",
              input.device(), device);

        bool crop;
        if (input_shape >= output_shape) {
            crop = true;
        } else if (input_shape <= output_shape) {
            crop = false;
            fill(output, output_value_t{});
        } else {
            panic("This function cannot crop and pad at the same time, but got input:shape={} and output:shape={}",
                  input_shape, output_shape);
        }

        // Transform to BR.
        constexpr bool RUNTIME_RANK = RANK == 0;
        constexpr usize R = RUNTIME_RANK ? 3 : RANK;
        constexpr usize B = RUNTIME_RANK ? 1 : N <= RANK ? 0 : N - RANK;
        constexpr usize BR = B + R;
        const usize rank = RUNTIME_RANK ? output_shape.rank_checked(options.rank) : R;

        auto get_span = [&](const auto& a, const auto& a_shape) {
            if constexpr (RUNTIME_RANK) {
                // Convert to BDHW, fusing the batch dimensions into B, according to the rank.
                auto a_ = a.span().template as_nd<BR>();
                auto a_shape_ = a_.shape().ranked(rank);
                a_ = a_.reshape(a_shape_);
                a_shape_[BR - 1] = a_shape[N - 1]; // if rfft, reset to logical width
                return Pair{a_, a_shape_};
            } else {
                return Pair{a.span(), a_shape};
            }
        };
        auto [input_br, input_shape_br] = get_span(input, input_shape);
        auto [output_br, output_shape_br] = get_span(output, output_shape);
        const auto [input_shape_b, input_shape_r] = input_shape_br.template split<B>();
        const auto [output_shape_b, output_shape_r] = output_shape_br.template split<B>();

        if constexpr (RUNTIME_RANK) {
            Vec<isize, N> mask{};
            for (usize i{}; i < N - static_cast<usize>(rank); ++i)
                mask[i] = 1;
            check(input_shape.vec * mask == output_shape.vec * mask, "The batch dimensions cannot be resized, {}, {}, {}",mask, input_shape, output_shape);
        } else if constexpr (B > 0) {
            check(input_shape_b == output_shape_b, "The batch dimensions cannot be resized");
        }

        using input_accessor_t = AccessorRestrict<const input_value_t, BR, isize>;
        using output_accessor_t = AccessorRestrict<output_value_t, BR, isize>;
        const auto input_accessor = input_accessor_t(input_br.get(), input_br.strides());
        const auto output_accessor = output_accessor_t(output_br.get(), output_br.strides());

        // Loop through the smallest shape.
        if (crop) {
            auto op = details::FourierResize<LAYOUT, true, isize, B, R, input_accessor_t, output_accessor_t>(
                input_accessor, output_accessor, input_shape_r, output_shape_r);
            noa::iwise(IS_FULL ? output_shape_br : output_shape_br.rfft(), device, op, std::forward<Input>(input), output);
        } else {
            auto op = details::FourierResize<LAYOUT, false, isize, B, R, input_accessor_t, output_accessor_t>(
                input_accessor, output_accessor, input_shape_r, output_shape_r);
            noa::iwise(IS_FULL ? input_shape_br : input_shape_br.rfft(), device, op, std::forward<Input>(input), output);
        }

        if constexpr (R >= 2) {
            if (options.correct_nyquist) {
                auto correct_redundant_plane = [&]<bool IS_NEW_RIGHTMOST_FULL_LENGTH>(const auto& plane, usize axis_to_exclude) {
                    // Remove the axis to exclude.
                    // For R=2, the are dealing with lines (1D), and for R=3 with planes (2D).
                    auto rank_axes = Vec<i32, R>::arange().exclude(axis_to_exclude);

                    // Add the batches.
                    const auto batch_axes = Vec<i32, B>::arange();
                    rank_axes += B;

                    // Construct the operator.
                    auto span = plane.span().filter(batch_axes.push_back(rank_axes));
                    auto iwise_shape = output_shape_br.filter(rank_axes);
                    auto op = details::FourierResizeCorrect<IS_CENTERED, not IS_NEW_RIGHTMOST_FULL_LENGTH, isize, B, R, decltype(span)>{
                        .output = span,
                        .shape = iwise_shape.template pop_back<not IS_NEW_RIGHTMOST_FULL_LENGTH>(),
                    };

                    // For RFFT, things can be simplified:
                    //  - The first correction fixes the ZY plane (or Y column for R=2) at X=nyquist.
                    //  - The second correction fixes the ZX plane (or X row for R=2) at Y=nyquist.
                    //      Because ZY is already corrected, we don't need to iterate through X, only through Z.
                    //      For R=3, this is a line along Z. For 2D, this is a single element.
                    //  - The third correction fixes the YX plane at Z=nyquist.
                    //      Because ZY and ZX are already corrected, we don't need to iterate through X, only through Y.
                    iwise_shape[R - 2] = IS_NEW_RIGHTMOST_FULL_LENGTH ? iwise_shape[R - 2] / 2 + 1 : 1;
                    noa::iwise(iwise_shape.push_front(output_shape_b.vec), device, op, output);
                };

                if (is_even(output_shape_r[R - 1]) and input_shape_r[R - 1] > output_shape_r[R - 1]) {
                    // For 2D transforms, this fixes the column (y=all,x=nyquist).
                    const auto nyquist_x = output_shape_r[R - 1] / 2;
                    const auto plane = output_br.subregion(Ellipsis{}, nyquist_x); // ((b..,)h,1)
                    correct_redundant_plane.template operator()<true>(plane, R - 1);
                }
                if (output_shape_r[R - 2] > 1 and is_even(output_shape_r[R - 2]) and input_shape_r[R - 2] > output_shape_r[R - 2]) {
                    // For 2D transforms, this only fixes one element at (y=nyquist,x=0) making sure its imag=0.
                    // The element (y=nyquist,x=nyquist) was already made real by the previous correction.
                    const auto nyquist_y = nf::frequency2index<IS_CENTERED>(-output_shape_r[R - 2] / 2, output_shape_r[R - 2]);
                    auto plane = output_br.subregion(Ellipsis{}, nyquist_y, Full{}); // ((b..,)1,w)
                    correct_redundant_plane.template operator()<IS_FULL>(plane, R - 2);
                }
                if constexpr (R == 3) {
                    if (output_shape_r[R - 3] > 1 and is_even(output_shape_r[R - 3]) and input_shape_r[R - 3] > output_shape_r[R - 3]) {
                        const auto nyquist_z = nf::frequency2index<IS_CENTERED>(-output_shape_r[R - 3] / 2, output_shape_r[R - 3]);
                        auto plane = output_br.subregion(Ellipsis{}, nyquist_z, Full{}, Full{}); // ((b..,)1,h,w)
                        correct_redundant_plane.template operator()<IS_FULL>(plane, R - 3);
                    }
                }
            }
        }
    }
    template<Layout LAYOUT, typename Input, typename Output, usize N>
    void resize_1d(
        Input&& input, Shape<isize, N> input_shape,
        Output&& output, Shape<isize, N> output_shape,
        ResizeOptions options = {}
    ) {
        resize<LAYOUT, 1>(std::forward<Input>(input), input_shape, std::forward<Output>(output), output_shape, options);
    }
    template<Layout LAYOUT, typename Input, typename Output, usize N>
    void resize_2d(
        Input&& input, Shape<isize, N> input_shape,
        Output&& output, Shape<isize, N> output_shape,
        ResizeOptions options = {}
    ) {
        resize<LAYOUT, 2>(std::forward<Input>(input), input_shape, std::forward<Output>(output), output_shape, options);
    }
    template<Layout LAYOUT, typename Input, typename Output, usize N>
    void resize_3d(
        Input&& input, Shape<isize, N> input_shape,
        Output&& output, Shape<isize, N> output_shape,
        ResizeOptions options = {}
    ) {
        resize<LAYOUT, 3>(std::forward<Input>(input), input_shape, std::forward<Output>(output), output_shape, options);
    }

    /// Returns cropped or zero-padded (r)FFT(s).
    /// \tparam LAYOUT
    ///     FFT layout.
    ///     Should be H2H, HC2HC, F2F or FC2FC.
    /// \tparam RANK:
    ///     Rank of the transform.
    ///     This determines which dimensions are considered batch dimensions.
    ///     Should be 1, 2, 3, or 0. If 0, fallback to runtime options.rank, in which case the batch dimensions
    ///     are collapsed to a single dimension and should therefore be collapsible.
    /// \param[in] input, input_shape:
    ///     The FFT to resize and its logical shape.
    ///     ((B..,)R), where R is W (RANK=1), HW (RANK=2), or DHW (RANK=3).
    /// \param[out] output_shape:
    ///     The logical shape of the output.
    ///     ((B..,)R), where R is W (RANK=1), HW (RANK=2), or DHW (RANK=3).
    /// \param options:
    ///     Resizing options.
    template<Layout LAYOUT, usize RANK = 0, nt::readable_array_decay_of_numeric Input, usize N>
        requires (nt::array_decay_nd<Input, N> and not LAYOUT.has_layout_change())
    [[nodiscard]] auto resize(
        Input&& input,
        const Shape<isize, N>& input_shape,
        const Shape<isize, N>& output_shape,
        const ResizeOptions& options = {}
    ) {
        using value_t = nt::mutable_value_type_t<Input>;
        auto output = Array<value_t, N>(LAYOUT.is_fx2fx() ? output_shape : output_shape.rfft(), input.options());
        resize<LAYOUT, RANK>(std::forward<Input>(input), input_shape, output, output_shape, options);
        return output;
    }
    template<Layout LAYOUT, typename Input, usize N>
    [[nodiscard]] auto resize_1d(
        Input&& input,
        const Shape<isize, N>& input_shape,
        const Shape<isize, N>& output_shape,
        const ResizeOptions& options = {}
    ) {
        return resize<LAYOUT, 1>(std::forward<Input>(input), input_shape, output_shape, options);
    }
    template<Layout LAYOUT, typename Input, usize N>
    [[nodiscard]] auto resize_2d(
        Input&& input,
        const Shape<isize, N>& input_shape,
        const Shape<isize, N>& output_shape,
        const ResizeOptions& options = {}
    ) {
        return resize<LAYOUT, 2>(std::forward<Input>(input), input_shape, output_shape, options);
    }
    template<Layout LAYOUT, typename Input, usize N>
    [[nodiscard]] auto resize_3d(
        Input&& input,
        const Shape<isize, N>& input_shape,
        const Shape<isize, N>& output_shape,
        const ResizeOptions& options = {}
    ) {
        return resize<LAYOUT, 3>(std::forward<Input>(input), input_shape, output_shape, options);
    }
}
