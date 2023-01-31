#pragma once

#include "noa/common/Types.h"
#include "noa/common/geometry/details/Utilities.h"

// Note: To support rectangular shapes, the kernels compute the transformation using normalized frequencies.
//       One other solution could have been to use an affine transform encoding the appropriate scaling to effectively
//       normalize the frequencies. Both options are fine and probably equivalent performance-wise.

// FIXME    For even sizes, there's an asymmetry due to the fact that there's only one Nyquist. After fftshift,
//          this extra frequency is on the left side of the axis. However, since we work with non-redundant
//          transforms, there's a slight error that can be introduced. For the elements close (+-1 element) to x=0
//          and close (+-1 element) to y=z=0.5, if during the rotation x becomes negative and we have to flip it, the
//          interpolator will incorrectly weight towards 0 the output value. This is simply due to the fact that on
//          the right side of the axis, there's no Nyquist (the axis stops and n+1 element is OOB, i.e. =0).

namespace noa::geometry::fft::details {
    // Linear transformations for 2D non-redundant centered FFTs:
    //  * While the other transformations are OK from complex arrays,
    //    they do not work for non-redundant FFTs. This class add supports that.
    //    Note that while the output layout can be non-centered, the input
    //    (where the interpolation is done) must be centered.
    template<Remap REMAP, typename Index, typename Data, typename Matrix,
             typename ShiftOrEmpty, typename Interpolator, typename Offset>
    class Transform2D {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        static_assert(REMAP_ & Layout::SRC_CENTERED &&
                      !(REMAP_ & Layout::SRC_FULL) &&
                      !(REMAP_ & Layout::DST_FULL));

        static_assert(traits::is_any_v<Matrix, float22_t, const float22_t*>);
        static_assert(traits::is_any_v<Data, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_any_v<ShiftOrEmpty, empty_t, float2_t, const float2_t*>);
        static_assert(traits::is_sint_v<Index>);

        using data_type = Data;
        using matrix_type = Matrix;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;
        using shift_or_empty_type = ShiftOrEmpty;
        using preshift_or_empty_type = std::conditional_t<std::is_pointer_v<shift_or_empty_type>, float2_t, empty_t>;

        using real_type = traits::value_type_t<data_type>;
        using index2_type = Int2<index_type>;
        using accessor_type = AccessorRestrict<data_type, 3, offset_type>;

    public:
        Transform2D(interpolator_type input, accessor_type output, dim4_t shape,
                    matrix_type matrix, shift_or_empty_type shift, float cutoff) noexcept
                : m_input(input), m_output(output),
                  m_matrix(matrix), m_shift(shift) {

            if constexpr (std::is_pointer_v<matrix_type>) {
                NOA_ASSERT(matrix != nullptr);
            } else if constexpr (std::is_pointer_v<shift_or_empty_type>) {
                NOA_ASSERT(shift != nullptr);
            }
            NOA_ASSERT(shape[1] == 1);

            m_shape = safe_cast<index2_type>(dim2_t(shape.get(2)));
            m_f_shape = float2_t(m_shape / 2 * 2 + index2_type(m_shape == 1)); // if odd, n-1

            m_cutoff_sqd = noa::math::clamp(cutoff, 0.f, 0.5f);
            m_cutoff_sqd *= m_cutoff_sqd;

            if constexpr (traits::is_float2_v<shift_or_empty_type>)
                m_shift *= math::Constants<float>::PI2 / float2_t(m_shape);
            else if constexpr (traits::is_float2_v<preshift_or_empty_type>)
                m_preshift = math::Constants<float>::PI2 / float2_t(m_shape);
        }

        NOA_IHD void operator()(index_type batch, index_type y, index_type x) const noexcept { // x == u
            // Compute the frequency corresponding to the gid and inverse transform.
            const index_type v = index2frequency<IS_DST_CENTERED>(y, m_shape[0]);
            float2_t freq{v, x};
            freq /= m_f_shape; // [-0.5, 0.5]
            if (math::dot(freq, freq) > m_cutoff_sqd) {
                m_output(batch, y, x) = 0;
                return;
            }

            if constexpr (std::is_pointer_v<matrix_type>)
                freq = m_matrix[batch] * freq;
            else
                freq = m_matrix * freq;

            // Non-redundant transform, so flip to the valid Hermitian pair, if necessary.
            real_type conj = 1;
            if (freq[1] < 0.f) {
                freq = -freq;
                if constexpr (traits::is_complex_v<data_type>)
                    conj = -1;
            }

            // Convert back to index and fetch value.
            freq[0] += 0.5f; // [0, 1]
            freq *= m_f_shape; // [0, N-1]
            data_type value = m_input(freq, batch);
            if constexpr (traits::is_complex_v<data_type>)
                value.imag *= conj;
            else
                (void) conj;

            // Phase shift the interpolated value.
            if constexpr (traits::is_complex_v<data_type> && !std::is_empty_v<shift_or_empty_type>) {
                if constexpr (std::is_pointer_v<shift_or_empty_type>) {
                    const float2_t shift = m_shift[batch] * m_preshift;
                    value *= phaseShift<data_type>(shift, float2_t{v, x});
                } else {
                    value *= phaseShift<data_type>(m_shift, float2_t{v, x});
                }
            }

            m_output(batch, y, x) = value;
        }

    private:
        interpolator_type m_input;
        accessor_type m_output;
        matrix_type m_matrix;
        shift_or_empty_type m_shift;
        preshift_or_empty_type m_preshift;
        index2_type m_shape;
        float2_t m_f_shape;
        float m_cutoff_sqd;
    };

    template<Remap REMAP, typename Index, typename Data, typename Matrix,
             typename ShiftOrEmpty, typename Interpolator, typename Offset>
    auto transform2D(const Interpolator& input,
                     const AccessorRestrict<Data, 3, Offset>& output,
                     dim4_t shape, Matrix matrix, ShiftOrEmpty shift, float cutoff) noexcept {
        return Transform2D<REMAP, Index, Data, Matrix, ShiftOrEmpty, Interpolator, Offset>(
                input, output, shape, matrix, shift, cutoff);
    }
}

namespace noa::geometry::fft::details {
    template<Remap REMAP, typename Index, typename Data, typename MatrixOrEmpty,
            typename ShiftOrEmpty, typename Interpolator, typename Offset>
    class TransformSymmetry2D {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        static_assert(REMAP_ & Layout::SRC_CENTERED &&
                      !(REMAP_ & Layout::SRC_FULL) &&
                      !(REMAP_ & Layout::DST_FULL));

        static_assert(traits::is_any_v<Data, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_any_v<MatrixOrEmpty, empty_t, float22_t>);
        static_assert(traits::is_any_v<ShiftOrEmpty, empty_t, float2_t>);
        static_assert(traits::is_sint_v<Index>);

        using data_type = Data;
        using matrix_or_empty_type = MatrixOrEmpty;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;
        using shift_or_empty_type = ShiftOrEmpty;

        using real_type = traits::value_type_t<data_type>;
        using index2_type = Int2<index_type>;
        using accessor_type = AccessorRestrict<data_type, 3, offset_type>;

    public:
        TransformSymmetry2D(interpolator_type input, accessor_type output, dim4_t shape,
                            matrix_or_empty_type matrix, const float33_t* symmetry_matrices,
                            index_type symmetry_count, real_type scaling,
                            shift_or_empty_type shift, float cutoff) noexcept
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
            NOA_ASSERT(shape[1] == 1);
            NOA_ASSERT(symmetry_matrices != nullptr || symmetry_count == 0);

            const auto i_shape = safe_cast<index2_type>(dim2_t(shape.get(2)));
            m_size_y = i_shape[0];
            m_f_shape = float2_t(i_shape / 2 * 2 + index2_type(i_shape == 1)); // if odd, n-1

            m_cutoff_sqd = noa::math::clamp(cutoff, 0.f, 0.5f);
            m_cutoff_sqd *= m_cutoff_sqd;

            if constexpr (traits::is_float2_v<shift_or_empty_type>)
                m_shift *= math::Constants<float>::PI2 / float2_t(i_shape);
        }

        NOA_IHD void operator()(index_type batch, index_type y, index_type x) const noexcept { // x == u
            const index_type v = index2frequency<IS_DST_CENTERED>(y, m_size_y);
            float2_t freq{v, x};
            freq /= m_f_shape; // [-0.5, 0.5]
            if (math::dot(freq, freq) > m_cutoff_sqd) {
                m_output(batch, y, x) = 0;
                return;
            }

            data_type value;
            if constexpr (!std::is_empty_v<matrix_or_empty_type>) {
                freq = m_matrix * freq;
                value = interpolateFFT_(freq, batch);
            } else {
                const index_type iy = output2inputIndex_(y);
                value = m_input.at(batch, iy, x); // bypass interpolation when possible
            }

            for (index_type i = 0; i < m_symmetry_count; ++i) {
                const float33_t& m = m_symmetry_matrices[i];
                const float22_t sym_matrix{m[1][1], m[1][2],
                                           m[2][1], m[2][2]};
                const float2_t i_freq = sym_matrix * freq;
                value += interpolateFFT_(i_freq, batch);
            }

            value *= m_scaling;
            if constexpr (traits::is_complex_v<data_type> && !std::is_empty_v<shift_or_empty_type>)
                value *= phaseShift<data_type>(m_shift, float2_t{v, x});

            m_output(batch, y, x) = value;
        }

    private:
        // Interpolates the (complex) value at the normalized frequency "freq".
        NOA_IHD data_type interpolateFFT_(float2_t frequency, index_type batch) const noexcept {
            real_type conj = 1;
            if (frequency[1] < 0.f) {
                frequency = -frequency;
                if constexpr (traits::is_complex_v<data_type>)
                    conj = -1;
            }

            frequency[0] += 0.5f;
            frequency *= m_f_shape;
            data_type value = m_input(frequency, batch);
            if constexpr (traits::is_complex_v<data_type>)
                value.imag *= conj;
            else
                (void) conj;
            return value;
        }

        NOA_FHD index_type output2inputIndex_(index_type y) const noexcept {
            if constexpr (IS_DST_CENTERED) {
                return y; // output is centered, so do nothing
            } else {
                // The output is center and the output isn't.
                // FIXME
                return noa::math::FFTShift(y, m_size_y);
            }
        }

    private:
        interpolator_type m_input;
        accessor_type m_output;
        matrix_or_empty_type m_matrix;
        shift_or_empty_type m_shift;
        const float33_t* m_symmetry_matrices;
        index_type m_symmetry_count;
        index_type m_size_y;
        float2_t m_f_shape;
        float m_cutoff_sqd;
        real_type m_scaling;
    };

    template<Remap REMAP, typename Index, typename Data, typename MatrixOrEmpty,
             typename ShiftOrEmpty, typename Interpolator, typename Offset,
             typename Real = traits::value_type_t<Data>>
    auto transformSymmetry2D(const Interpolator& input,
                             const AccessorRestrict<Data, 3, Offset>& output,
                             dim4_t shape, MatrixOrEmpty matrix,
                             const float33_t* symmetry_matrices, Index symmetry_count, Real scaling,
                             ShiftOrEmpty shift, float cutoff) noexcept {
        return TransformSymmetry2D<
                REMAP, Index, Data, MatrixOrEmpty, ShiftOrEmpty, Interpolator, Offset>(
                input, output, shape, matrix, symmetry_matrices, symmetry_count, scaling, shift, cutoff);
    }
}
