#pragma once

#include "noa/core/Types.hpp"
#include "noa/algorithms/Utilities.hpp"

// Note: To support rectangular shapes, the kernels compute the transformation using normalized frequencies.
//       One other solution could have been to use an affine transform encoding the appropriate scaling to effectively
//       normalize the frequencies. Both options are fine and probably equivalent performance-wise.

// FIXME    For even sizes, there's an asymmetry due to the fact that there's only one Nyquist. After fftshift,
//          this extra frequency is on the left side of the axis. However, since we work with non-redundant
//          transforms, there's a slight error that can be introduced. For the elements close (+-1 element) to x=0
//          and close (+-1 element) to y=z=0.5, if during the rotation x becomes negative and we have to flip it, the
//          interpolator will incorrectly weight towards 0 the output value. This is simply due to the fact that on
//          the right side of the axis, there's no Nyquist (the axis stops and n+1 element is OOB, i.e. =0).
//          Anyway, the fix should be done in the interpolator, not here. Maybe create a InterpolatorRFFT?

namespace noa::algorithm::geometry {
    // Linear transformations for 2D or 3D non-redundant centered FFTs:
    //  * While the other transformations are OK from complex arrays,
    //    they do not work for non-redundant FFTs. This class adds support that.
    //    Note that while the output layout can be non-centered, the input
    //    (where the interpolation is done) must be centered.
    template<size_t N, noa::fft::Remap REMAP,
            typename Index, typename Value, typename Matrix,
            typename ShiftOrEmpty, typename Interpolator, typename Offset>
    class TransformRFFT {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_DST_CENTERED = u8_REMAP & Layout::DST_CENTERED;
        static_assert(u8_REMAP & Layout::SRC_CENTERED &&
                      u8_REMAP & Layout::SRC_HALF &&
                      u8_REMAP & Layout::DST_HALF);

        static_assert(traits::is_real_or_complex_v<Value>);
        static_assert(traits::is_sint_v<Index>);

        using value_type = Value;
        using matrix_type = Matrix;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;
        using shift_or_empty_type = ShiftOrEmpty;

        using coord_type = typename Interpolator::coord_type;
        using vec_type = Vec<coord_type, N>;
        using preshift_or_empty_type = std::conditional_t<
                std::is_pointer_v<shift_or_empty_type>, vec_type, Empty>;
        using real_value_type = traits::value_type_t<value_type>;
        using shape_type = Shape<index_type, N>;
        using accessor_type = AccessorRestrict<value_type, N + 1, offset_type>;

        static_assert(traits::is_any_v<ShiftOrEmpty, Empty, vec_type, const vec_type*>);
        static_assert(
                (N == 2 && noa::traits::is_any_v<Matrix, Mat22<coord_type>, const Mat22<coord_type>*>) ||
                (N == 3 && noa::traits::is_any_v<Matrix, Mat33<coord_type>, const Mat33<coord_type>*>));

    public:
        TransformRFFT(const interpolator_type& input,
                      const accessor_type& output,
                      const Shape4<index_type>& shape,
                      const matrix_type& matrix,
                      const shift_or_empty_type& shift,
                      coord_type cutoff) noexcept
                : m_input(input), m_output(output),
                  m_matrix(matrix), m_shift(shift) {

            if constexpr (std::is_pointer_v<matrix_type>) {
                NOA_ASSERT(matrix != nullptr);
            } else if constexpr (std::is_pointer_v<shift_or_empty_type>) {
                NOA_ASSERT(shift != nullptr);
            }
            if constexpr (N == 2) {
                NOA_ASSERT(shape[1] == 1);
            }

            m_shape = shape_type(shape.data() + 4 - N);
            m_f_shape = vec_type((m_shape / 2 * 2 + shape_type(m_shape == 1)).vec()); // if odd, n-1

            m_cutoff_sqd = noa::math::clamp(cutoff, coord_type{0}, coord_type{0.5});
            m_cutoff_sqd *= m_cutoff_sqd;

            if constexpr (traits::is_realN_v<shift_or_empty_type, N>)
                m_shift *= 2 * noa::math::Constant<coord_type>::PI / vec_type(m_shape.vec());
            else if constexpr (traits::is_realN_v<preshift_or_empty_type, N>)
                m_preshift = 2 * noa::math::Constant<coord_type>::PI / vec_type(m_shape.vec());
        }

        template<typename Void = void, typename = std::enable_if_t<(N == 2) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type y, index_type x) const noexcept { // x == u
            // Compute the frequency corresponding to the gid and inverse transform.
            const index_type v = index2frequency<IS_DST_CENTERED>(y, m_shape[0]);
            vec_type freq{v, x};
            freq /= m_f_shape; // [-0.5, 0.5]
            if (noa::math::dot(freq, freq) > m_cutoff_sqd) {
                m_output(batch, y, x) = 0;
                return;
            }

            if constexpr (std::is_pointer_v<matrix_type>)
                freq = m_matrix[batch] * freq;
            else
                freq = m_matrix * freq;

            // Non-redundant transform, so flip to the valid Hermitian pair, if necessary.
            real_value_type conj = 1;
            if (freq[1] < coord_type{0}) {
                freq = -freq;
                if constexpr (noa::traits::is_complex_v<value_type>)
                    conj = -1;
            }

            // Convert back to index and fetch value.
            freq[0] += coord_type{0.5}; // [0, 1]
            freq *= m_f_shape; // [0, N-1]
            auto value = static_cast<value_type>(m_input(freq, batch));
            if constexpr (noa::traits::is_complex_v<value_type>)
                value.imag *= conj;
            else
                (void) conj;

            // Phase shift the interpolated value.
            if constexpr (noa::traits::is_complex_v<value_type> && !std::is_empty_v<shift_or_empty_type>) {
                if constexpr (std::is_pointer_v<shift_or_empty_type>) {
                    const vec_type shift = m_shift[batch] * m_preshift;
                    value *= phase_shift<value_type>(shift, vec_type{v, x});
                } else {
                    value *= phase_shift<value_type>(m_shift, vec_type{v, x});
                }
            }
            m_output(batch, y, x) = value;
        }

        template<typename Void = void, typename = std::enable_if_t<(N == 3) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type z, index_type y, index_type x) const noexcept { // x == u
            const index_type w = index2frequency<IS_DST_CENTERED>(z, m_shape[0]);
            const index_type v = index2frequency<IS_DST_CENTERED>(y, m_shape[1]);
            vec_type freq{w, v, x};
            freq /= m_f_shape; // [-0.5, 0.5]
            if (noa::math::dot(freq, freq) > m_cutoff_sqd) {
                m_output(batch, z, y, x) = 0;
                return;
            }

            if constexpr (std::is_pointer_v<matrix_type>)
                freq = m_matrix[batch] * freq;
            else
                freq = m_matrix * freq;

            real_value_type conj = 1;
            if (freq[2] < coord_type{0}) {
                freq = -freq;
                if constexpr (noa::traits::is_complex_v<value_type>)
                    conj = -1;
            }

            freq[0] += coord_type{0.5};
            freq[1] += coord_type{0.5};
            freq *= m_f_shape;
            auto value = static_cast<value_type>(m_input(freq, batch));
            if constexpr (noa::traits::is_complex_v<value_type>)
                value.imag *= conj;
            else
                (void) conj;

            if constexpr (noa::traits::is_complex_v<value_type> && !std::is_empty_v<shift_or_empty_type>) {
                if constexpr (std::is_pointer_v<shift_or_empty_type>) {
                    const vec_type shift = m_shift[batch] * m_preshift;
                    value *= phase_shift<value_type>(shift, vec_type{w, v, x});
                } else {
                    value *= phase_shift<value_type>(m_shift, vec_type{w, v, x});
                }
            }
            m_output(batch, z, y, x) = value;
        }

    private:
        interpolator_type m_input;
        accessor_type m_output;
        matrix_type m_matrix;
        shape_type m_shape;
        vec_type m_f_shape;
        coord_type m_cutoff_sqd;
        NOA_NO_UNIQUE_ADDRESS shift_or_empty_type m_shift;
        NOA_NO_UNIQUE_ADDRESS preshift_or_empty_type m_preshift;
    };

    template<size_t N, noa::fft::Remap REMAP,
             typename Index, typename Value, typename MatrixOrEmpty,
             typename ShiftOrEmpty, typename Interpolator, typename Offset>
    class TransformSymmetryRFFT {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_DST_CENTERED = u8_REMAP & Layout::DST_CENTERED;
        static_assert(u8_REMAP & Layout::SRC_CENTERED &&
                      u8_REMAP & Layout::SRC_HALF &&
                      u8_REMAP & Layout::DST_HALF);

        static_assert(noa::traits::is_real_or_complex_v<Value>);
        static_assert(noa::traits::is_sint_v<Index>);

        using value_type = Value;
        using matrix_or_empty_type = MatrixOrEmpty;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;
        using shift_or_empty_type = ShiftOrEmpty;
        using real_value_type = noa::traits::value_type_t<value_type>;

        using coord_type = typename Interpolator::coord_type;
        using vec_type = Vec<coord_type, N>;
        using shape_type = Shape<index_type, N - 1>;
        using accessor_type = AccessorRestrict<value_type, N + 1, offset_type>;

        using matrix_type = std::conditional_t<N == 2, Mat22<coord_type>, Mat33<coord_type>>;
        static_assert(noa::traits::is_any_v<MatrixOrEmpty, Empty, matrix_type>);
        static_assert(noa::traits::is_any_v<ShiftOrEmpty, Empty, vec_type>);

    public:
        TransformSymmetryRFFT(
                const interpolator_type& input,
                const accessor_type& output,
                const Shape4<index_type>& shape,
                const matrix_or_empty_type& matrix,
                const Float33* symmetry_matrices,
                index_type symmetry_count,
                real_value_type scaling,
                const shift_or_empty_type& shift,
                coord_type cutoff) noexcept
                : m_input(input), m_output(output),
                  m_matrix(matrix), m_shift(shift),
                  m_symmetry_matrices(symmetry_matrices),
                  m_symmetry_count(symmetry_count),
                  m_scaling(scaling) {

            if constexpr (std::is_pointer_v<matrix_or_empty_type>) {
                NOA_ASSERT(matrix != nullptr);
            } else if constexpr (std::is_pointer_v<shift_or_empty_type>) {
                NOA_ASSERT(shift != nullptr);
            }
            NOA_ASSERT(symmetry_matrices != nullptr || symmetry_count == 0);
            if constexpr (N == 2) {
                NOA_ASSERT(shape[1] == 1);
            }

            using shape_nd_t = Shape<index_type, N>;
            const auto i_shape = shape_nd_t(shape.data() + 4 - N); // {y, x} or {z, y, x}
            m_shape = i_shape.pop_back(); // {y} or {z, y}
            m_f_shape = vec_type((i_shape / 2 * 2 + shape_nd_t(i_shape == 1)).vec()); // if odd, n-1

            m_cutoff_sqd = noa::math::clamp(cutoff, coord_type{0}, coord_type{0.5});
            m_cutoff_sqd *= m_cutoff_sqd;

            if constexpr (traits::is_realN_v<shift_or_empty_type, N>)
                m_shift *= 2 * noa::math::Constant<coord_type>::PI / vec_type(i_shape.vec());
        }

        NOA_HD void operator()(index_type batch, index_type y, index_type x) const noexcept { // x == u
            const index_type v = index2frequency<IS_DST_CENTERED>(y, m_shape[0]);
            vec_type freq{v, x};
            freq /= m_f_shape; // [-0.5, 0.5]
            if (noa::math::dot(freq, freq) > m_cutoff_sqd) {
                m_output(batch, y, x) = 0;
                return;
            }

            value_type value;
            if constexpr (!std::is_empty_v<matrix_or_empty_type>) {
                freq = m_matrix * freq;
                value = interpolate_rfft_(freq, batch);
            } else {
                const index_type iy = output2input_index_(y, m_shape[0]);
                value = m_input.at(batch, iy, x); // bypass interpolation when possible
            }

            for (index_type i = 0; i < m_symmetry_count; ++i) {
                const Float33& m = m_symmetry_matrices[i];
                const matrix_type sym_matrix{m[1][1], m[1][2],
                                             m[2][1], m[2][2]};
                const vec_type i_freq = sym_matrix * freq;
                value += interpolate_rfft_(i_freq, batch);
            }

            value *= m_scaling;
            if constexpr (noa::traits::is_complex_v<value_type> && !std::is_empty_v<shift_or_empty_type>)
                value *= phase_shift<value_type>(m_shift, vec_type{v, x});

            m_output(batch, y, x) = value;
        }

        NOA_HD void operator()(index_type batch, index_type z, index_type y, index_type x) const noexcept { // x == u
            const index_type w = index2frequency<IS_DST_CENTERED>(z, m_shape[0]);
            const index_type v = index2frequency<IS_DST_CENTERED>(y, m_shape[1]);
            vec_type frequency{w, v, x};
            frequency /= m_f_shape; // [-0.5, 0.5]
            if (noa::math::dot(frequency, frequency) > m_cutoff_sqd) {
                m_output(batch, z, y, x) = 0;
                return;
            }

            if constexpr (!std::is_empty_v<matrix_or_empty_type>)
                frequency = m_matrix * frequency;

            value_type value;
            if constexpr (!std::is_empty_v<matrix_or_empty_type>) {
                frequency = m_matrix * frequency;
                value = interpolate_rfft_(frequency, batch);
            } else {
                const index_type iz = output2input_index_(z, m_shape[0]);
                const index_type iy = output2input_index_(y, m_shape[1]);
                value = static_cast<value_type>(m_input.at(batch, iz, iy, x)); // bypass interpolation when possible
            }

            for (index_type i = 0; i < m_symmetry_count; ++i) {
                const vec_type i_frequency(m_symmetry_matrices[i] * frequency);
                value += interpolate_rfft_(i_frequency, batch);
            }

            value *= m_scaling;
            if constexpr (traits::is_complex_v<value_type> && !std::is_empty_v<shift_or_empty_type>)
                value *= phase_shift<value_type>(m_shift, vec_type{w, v, x});

            m_output(batch, z, y, x) = value;
        }

    private:
        // Interpolates the (complex) value at the normalized frequency "freq".
        NOA_HD value_type interpolate_rfft_(vec_type frequency, index_type batch) const noexcept {
            real_value_type conj = 1;
            if (frequency[N - 1] < coord_type{0}) {
                frequency = -frequency;
                if constexpr (noa::traits::is_complex_v<value_type>)
                    conj = -1;
            }

            frequency[0] += coord_type{0.5};
            if constexpr (N == 3)
                frequency[1] += coord_type{0.5};
            frequency *= m_f_shape;
            auto value = static_cast<value_type>(m_input(frequency, batch));
            if constexpr (noa::traits::is_complex_v<value_type>)
                value.imag *= conj;
            else
                (void) conj;
            return value;
        }

        NOA_HD static constexpr index_type output2input_index_(index_type i, index_type size) noexcept {
            if constexpr (IS_DST_CENTERED) {
                return i; // output is centered, so do nothing
            } else {
                // FIXME The output is center and the output isn't?
                return noa::math::fft_shift(i, size);
            }
        }

    private:
        interpolator_type m_input;
        accessor_type m_output;
        NOA_NO_UNIQUE_ADDRESS matrix_or_empty_type m_matrix;
        NOA_NO_UNIQUE_ADDRESS shift_or_empty_type m_shift;
        const Float33* m_symmetry_matrices;
        index_type m_symmetry_count;
        shape_type m_shape;
        vec_type m_f_shape;
        coord_type m_cutoff_sqd;
        real_value_type m_scaling;
    };
}

namespace noa::algorithm::geometry {
    template<noa::fft::Remap REMAP, typename Index, typename Value, typename Matrix,
             typename Coord, typename ShiftOrEmpty, typename Interpolator, typename Offset>
    auto transform_rfft_2d(
            const Interpolator& input,
            const AccessorRestrict<Value, 3, Offset>& output,
            const Shape4<Index>& shape,
            const Matrix& matrix,
            const ShiftOrEmpty& shift,
            Coord cutoff) noexcept {
        return TransformRFFT<2, REMAP, Index, Value, Matrix, ShiftOrEmpty, Interpolator, Offset>(
                input, output, shape, matrix, shift, cutoff);
    }

    template<noa::fft::Remap REMAP, typename Index, typename Data, typename Matrix,
             typename Coord, typename ShiftOrEmpty, typename Interpolator, typename Offset>
    auto transform_rfft_3d(
            const Interpolator& input,
            const AccessorRestrict<Data, 4, Offset>& output,
            const Shape4<Index>& shape,
            const Matrix& matrix,
            const ShiftOrEmpty& shift,
            Coord cutoff) noexcept {
        return TransformRFFT<3, REMAP, Index, Data, Matrix, ShiftOrEmpty, Interpolator, Offset>(
                input, output, shape, matrix, shift, cutoff);
    }

    template<noa::fft::Remap REMAP,
             typename Index, typename Value, typename Coord, typename MatrixOrEmpty,
             typename ShiftOrEmpty, typename Interpolator, typename Offset,
             typename Real = traits::value_type_t<Value>>
    auto transform_symmetry_rfft_2d(
            const Interpolator& input,
            const AccessorRestrict<Value, 3, Offset>& output,
            const Shape4<Index>& shape,
            const MatrixOrEmpty& matrix,
            const Float33* symmetry_matrices,
            Index symmetry_count, Real scaling,
            const ShiftOrEmpty& shift,
            Coord cutoff) noexcept {
        return TransformSymmetryRFFT<
                2, REMAP, Index, Value, MatrixOrEmpty, ShiftOrEmpty, Interpolator, Offset>(
                input, output, shape, matrix, symmetry_matrices, symmetry_count, scaling, shift, cutoff);
    }

    template<noa::fft::Remap REMAP,
             typename Index, typename Data, typename Coord, typename MatrixOrEmpty,
             typename ShiftOrEmpty, typename Interpolator, typename Offset,
             typename Real = traits::value_type_t<Data>>
    auto transform_symmetry_rfft_3d(
            const Interpolator& input,
            const AccessorRestrict<Data, 4, Offset>& output,
            const Shape4<Index>& shape,
            const MatrixOrEmpty& matrix,
            const Float33* symmetry_matrices,
            Index symmetry_count, Real scaling,
            const ShiftOrEmpty& shift,
            Coord cutoff) noexcept {
        return TransformSymmetryRFFT<
                3, REMAP, Index, Data, MatrixOrEmpty, ShiftOrEmpty, Interpolator, Offset>(
                input, output, shape, matrix, symmetry_matrices, symmetry_count, scaling, shift, cutoff);
    }
}
