#pragma once

#include "noa/runtime/core/Traits.hpp"
#include "noa/runtime/core/Shape.hpp"
#include "noa/runtime/Resize.hpp"
#include "noa/runtime/Factory.hpp"

#include "noa/fft/core/Layout.hpp"
#include "noa/fft/core/Transform.hpp"
#include "noa/fft/core/Frequency.hpp"

namespace noa::fft::details {
    template<Layout LAYOUT, bool CROP,
             nt::sinteger Index,
             nt::readable_nd<4> Input,
             nt::writable_nd<4> Output>
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

        using dh_shape_type = std::conditional_t<IS_HALF, Shape<index_type, 2>, Empty>;
        using dhw_vec_type = std::conditional_t<not IS_HALF or IS_CENTERED, Shape<index_type, 3 - (IS_CENTERED and IS_HALF)>, Empty>;

        constexpr FourierResize(
            const input_type& input,
            const output_type& output,
            const Shape<index_type, 3>& input_shape,
            const Shape<index_type, 3>& output_shape
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
                const auto border_left = output_shape / 2 - input_shape / 2;
                m_offset = abs(border_left.template pop_back<IS_HALF>());
            }
        }

        constexpr void operator()(index_type i, index_type j, index_type k, index_type l) const {
            if constexpr (IS_CENTERED) {
                const auto j_ = j + m_offset[0];
                const auto k_ = k + m_offset[1];
                auto l_ = l;
                if constexpr (not IS_HALF)
                    l_ += m_offset[2];
                if constexpr (not CROP) {
                    m_output(i, j_, k_, l_) = cast_or_abs_squared<output_value_type>(m_input(i, j, k, l));
                } else {
                    m_output(i, j, k, l) = cast_or_abs_squared<output_value_type>(m_input(i, j_, k_, l_));
                }
            } else {
                if constexpr (CROP and IS_HALF) {
                    const auto ij = j < (m_output_shape[0] + 1) / 2 ? j : j + m_input_shape[0] - m_output_shape[0];
                    const auto ik = k < (m_output_shape[1] + 1) / 2 ? k : k + m_input_shape[1] - m_output_shape[1];
                    m_output(i, j, k, l) = cast_or_abs_squared<output_value_type>(m_input(i, ij, ik, l));

                } else if constexpr (not CROP and IS_HALF) {
                    const auto oj = j < (m_input_shape[0] + 1) / 2 ? j : j + m_output_shape[0] - m_input_shape[0];
                    const auto ok = k < (m_input_shape[1] + 1) / 2 ? k : k + m_output_shape[1] - m_input_shape[1];
                    m_output(i, oj, ok, l) = cast_or_abs_squared<output_value_type>(m_input(i, j, k, l));

                } else if constexpr (CROP and not IS_HALF) {
                    const auto ij = j < m_limit[0] ? j : j + m_offset[0];
                    const auto ik = k < m_limit[1] ? k : k + m_offset[1];
                    const auto il = l < m_limit[2] ? l : l + m_offset[2];
                    m_output(i, j, k, l) = cast_or_abs_squared<output_value_type>(m_input(i, ij, ik, il));

                } else if constexpr (not CROP and not IS_HALF) {
                    const auto oj = j < m_limit[0] ? j : j + m_offset[0];
                    const auto ok = k < m_limit[1] ? k : k + m_offset[1];
                    const auto ol = l < m_limit[2] ? l : l + m_offset[2];
                    m_output(i, oj, ok, ol) = cast_or_abs_squared<output_value_type>(m_input(i, j, k, l));
                }
            }
        }

    private:
        input_type m_input;
        output_type m_output;
        dh_shape_type m_input_shape{};
        dh_shape_type m_output_shape{};
        dhw_vec_type m_offset{};
        dhw_vec_type m_limit{};
    };

    template<bool IS_CENTERED, bool IS_HALF,
             nt::sinteger Index,
             nt::writable_nd<3> Output>
    struct FourierResizeCorrect {
        using index_type = Index;
        using output_type = Output;
        using value_type = nt::value_type_t<output_type>;

        output_type output;
        Shape<index_type, 2 - IS_HALF> shape{};

        static constexpr void average_pairs(const auto& array, const auto& indices, const auto& indices_pair) {
            auto& a = array.at(indices);
            auto& b = array.at(indices_pair);
            auto b_conj = b;
            if constexpr (nt::complex<value_type>)
                b_conj.imag = -b_conj.imag;
            auto avg = (a + b_conj) / 2;
            a = avg;
            if constexpr (nt::complex<value_type>)
                avg.imag = -avg.imag;
            b = avg;
        }

        constexpr void symmetrize_plane(const auto& array, const Vec<index_type, 2>& indices) const {
            const auto frequency = index2frequency<IS_CENTERED, IS_HALF>(indices, shape);

            // Get the Hermitian pair.
            // N = 10: [-5,-4,-3,-2,-1,+0,+1,+2,+3,+4], e.g. pair=[-5,-5], pair=[-4, 4], .., pair[0, 0]
            // N = 9:     [-4,-3,-2,-1,+0,+1,+2,+3,+4], e.g. pair=[-4, 4], ..., pair[0, 0]
            auto frequency_pair = frequency;
            for (i32 l{}; l < 2 - IS_HALF; ++l) {
                // If nyquist or zero, conjugate-average with itself, aka set imaginary to 0.
                const auto nyquist = -(shape[l] + 1) / 2;
                if (frequency[l] != nyquist)
                    frequency_pair[l] = -frequency[l];
            }

            const auto indices_pair = frequency2index<IS_CENTERED, IS_HALF>(frequency_pair, shape);
            average_pairs(array, indices, indices_pair);
        }

        constexpr void operator()(index_type b, index_type i, index_type j) const {
            symmetrize_plane(output[b], Vec{i, j});
        }
    };

    struct FourierResizeCorrectImagZero {
        constexpr void operator()(nt::complex auto& value) {
            value.imag = 0;
        }
    };
}

namespace noa::fft {
    struct ResizeOptions {
        // TODO rescale

        /// Rank of the transform.
        /// See transform_shape for more details.
        i32 rank{-1};

        /// When cropping to a new logical even-size, the new nyquist frequencies need to be corrected to restore
        /// the Hermitian symmetry. The correction applied here consists in conjugate-averaging the Hermitian pairs
        /// (the imaginary value of frequencies without pairs is set to zero). This correction can be skipped if a
        /// lowpass filter removing Nyquist is applied before computing the c2r transform. Otherwise, without this
        /// correction, the c2r transform is implementation defined (cuFFT will likely generate a different output
        /// compared to FFTW).
        bool correct_nyquist{false};
    };

    /// Crops or zero-pads (r)FFT(s).
    /// \tparam REMAP       FFT layouts. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[in] input    FFT to resize.
    /// \param input_shape  Logical shape of the input.
    /// \param[out] output  Resized FFT. If real and the input is complex, the power-spectrum is computed.
    /// \param output_shape Logical shape of the output.
    /// \param options      Resizing options.
    template<Layout REMAP, nt::readable_array_decay Input, nt::writable_array_decay Output, usize N>
    requires (nt::array_decay_with_compatible_or_spectrum_types<Input, Output> and
              nt::array_decay_nd<Input, N> and
              nt::array_decay_nd<Output, N> and
              not REMAP.has_layout_change())
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

        constexpr bool IS_FULL = REMAP.is_fx2xx();
        constexpr bool IS_CENTERED = REMAP.is_xc2xx();
        check(nd::are_arrays_valid(input, output), "Empty array detected");
        check(not are_overlapped(input, output), "Input and output arrays should not overlap");

        check(input.shape() == (IS_FULL ? input_shape : input_shape.rfft()),
              "Given the {} remap, the input fft is expected to have a physical shape of {}, but got {}",
              REMAP, IS_FULL ? input_shape : input_shape.rfft(), input.shape());
        check(output.shape() == (IS_FULL ? output_shape : output_shape.rfft()),
              "Given the {} remap, the output fft is expected to have a physical shape of {}, but got {}",
              REMAP, IS_FULL ? output_shape : output_shape.rfft(), output.shape());

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

        // Transform to BDHW.
        auto transform_bdhw = [&]<typename T>(const auto& a, const T& a_shape) {
            auto a_ = a.span().template as_nd<4>();
            auto a_shape_ = transform_shape(a_.shape(), options.rank);
            a_ = a_.reshape(a_shape_);
            a_shape_[T::SIZE - 1] = a_shape[T::SIZE - 1];
            return Pair{a_, a_shape_};
        };
        auto [input_4d, input_shape_4d] = transform_bdhw(input, input_shape);
        auto [output_4d, output_shape_4d] = transform_bdhw(output, output_shape);
        check(input_shape_4d[0] == output_shape_4d[0], "The batch dimension cannot be resized");

        using input_accessor_t = AccessorRestrict<const input_value_t, 4, isize>;
        using output_accessor_t = AccessorRestrict<output_value_t, 4, isize>;
        const auto input_accessor = input_accessor_t(input_4d.get(), input_4d.strides());
        const auto output_accessor = output_accessor_t(output_4d.get(), output_4d.strides());
        const auto input_shape_3d = input_shape_4d.pop_front();
        const auto output_shape_3d = output_shape_4d.pop_front();

        // Loop through the smallest shape.
        if (crop) {
            auto op = details::FourierResize<REMAP, true, isize, input_accessor_t, output_accessor_t>(
                input_accessor, output_accessor, input_shape_3d, output_shape_3d);
            noa::iwise(IS_FULL ? output_shape_4d : output_shape_4d.rfft(), device, op, std::forward<Input>(input), output);
        } else {
            auto op = details::FourierResize<REMAP, false, isize, input_accessor_t, output_accessor_t>(
                input_accessor, output_accessor, input_shape_3d, output_shape_3d);
            noa::iwise(IS_FULL ? input_shape_4d : input_shape_4d.rfft(), device, op, std::forward<Input>(input), output);
        }

        if (options.correct_nyquist) {
            auto correct_redundant_plane = [&]<bool IS_X_FULL>(const auto& plane, i32 i, i32 j) {
                auto span = plane.span().filter(0, i, j);
                auto iwise_shape = output_shape_4d.filter(i, j);
                auto op = details::FourierResizeCorrect<IS_CENTERED, not IS_X_FULL, isize, decltype(span)>{
                    .output = span,
                    .shape = iwise_shape.template pop_back<not IS_X_FULL>(),
                };
                // Loop through the first half + Nyquist.
                // For ZX and YX plane, only the lines x=0 and x=nyquist need to be corrected.
                // But since the ZY plane corrects x=nyquist, only x=0 needs to be corrected.
                iwise_shape[1] = IS_X_FULL ? iwise_shape[1] / 2 + 1 : 1;
                noa::iwise(iwise_shape.push_front(output_shape_4d[0]), device, op, output);
            };

            if (is_even(output_shape_4d[3]) and input_shape_4d[3] > output_shape_4d[3]) {
                const auto nyquist_x = output_shape_4d[3] / 2;
                const auto plane = output_4d.subregion(Full{}, Full{}, Full{}, nyquist_x); // (b,z,y,1)
                correct_redundant_plane.template operator()<true>(plane, 1, 2); // ZY is always full
            }
            if (output_shape_4d[2] > 1 and is_even(output_shape_4d[2]) and input_shape_4d[2] > output_shape_4d[2]) {
                const auto nyquist_y = nf::frequency2index<IS_CENTERED>(-output_shape_4d[2] / 2, output_shape_4d[2]);
                auto plane = output_4d.subregion(Full{}, Full{}, nyquist_y, Full{}); // (b,z,1,x)
                correct_redundant_plane.template operator()<IS_FULL>(plane, 1, 3);
            }
            if (output_shape_4d[1] > 1 and is_even(output_shape_4d[1]) and input_shape_4d[1] > output_shape_4d[1]) {
                const auto nyquist_z = nf::frequency2index<IS_CENTERED>(-output_shape_4d[1] / 2, output_shape_4d[1]);
                auto plane = output_4d.subregion(Full{}, nyquist_z, Full{}, Full{}); // (b,1,y,x)
                correct_redundant_plane.template operator()<IS_FULL>(plane, 2, 3);
            }
        }
    }

    /// Returns a cropped or zero-padded FFT.
    /// \tparam REMAP       FFT layouts. Should be H2H, HC2HC, F2F or FC2FC.
    /// \param[in] input    FFT to resize.
    /// \param input_shape  Logical shape of the input.
    /// \param output_shape Logical shape of the output.
    /// \param options      Resizing options.
    template<Layout REMAP, nt::readable_array_decay_of_numeric Input, usize N>
        requires (nt::array_decay_nd<Input, N> and not REMAP.has_layout_change())
    [[nodiscard]] auto resize(
        Input&& input,
        const Shape<isize, N>& input_shape,
        const Shape<isize, N>& output_shape,
        const ResizeOptions& options = {}
    ) {
        using value_t = nt::mutable_value_type_t<Input>;
        auto output = Array<value_t, N>(REMAP.is_fx2fx() ? output_shape : output_shape.rfft(), input.options());
        resize<REMAP>(std::forward<Input>(input), input_shape, output, output_shape, options);
        return output;
    }
}
