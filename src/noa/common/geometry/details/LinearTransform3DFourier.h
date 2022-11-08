#pragma once
#include "noa/common/Types.h"

namespace noa::geometry::fft::details {
    using Remap = ::noa::fft::Remap;

    template<bool IS_CENTERED, typename Index>
    [[nodiscard]] NOA_FHD Index index2frequency(Index index, Index size) {
        if constexpr (IS_CENTERED)
            return index - size / 2;
        else
            return index < (size + 1) / 2 ? index : index - size;
        return Index{}; // unreachable
    }

    [[nodiscard]] NOA_FHD cfloat_t phaseShift(float3_t shift, float3_t freq) {
        const float factor = -math::dot(shift, freq);
        cfloat_t phase_shift;
        noa::math::sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }
}

namespace noa::geometry::fft::details {
    template<Remap REMAP, typename Index, typename Data, typename Matrix,
             typename ShiftOrEmpty, typename Interpolator, typename Offset>
    class Transform3D {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        static_assert(REMAP_ & Layout::SRC_CENTERED &&
                      !(REMAP_ & Layout::SRC_FULL) &&
                      !(REMAP_ & Layout::DST_FULL));

        static_assert(traits::is_any_v<Matrix, float33_t, const float33_t*>);
        static_assert(traits::is_any_v<Data, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_any_v<ShiftOrEmpty, empty_t, float3_t, const float3_t*>);
        static_assert(traits::is_int_v<Index>);

        using data_type = Data;
        using matrix_type = Matrix;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;
        using shift_or_empty_type = ShiftOrEmpty;

        using real_type = traits::value_type_t<data_type>;
        using index3_type = Int3<index_type>;
        using accessor_type = AccessorRestrict<data_type, 4, offset_type>;

    public:
        Transform3D(interpolator_type input, accessor_type output, dim4_t shape,
                    matrix_type matrix, shift_or_empty_type shift, float cutoff) noexcept
                : m_input(input), m_output(output),
                  m_matrix(matrix), m_shift(shift) {

            if constexpr (std::is_pointer_v<matrix_type>) {
                NOA_ASSERT(matrix != nullptr);
            } else if constexpr (std::is_pointer_v<shift_or_empty_type>) {
                NOA_ASSERT(shift != nullptr);
            }
            NOA_ASSERT(shape[1] == 1);

            m_shape = safe_cast<index3_type>(dim3_t(shape.get(1)));
            m_f_shape = float3_t(m_shape / 2 * 2 + int3_t(m_shape == 1)); // if odd, n-1

            m_cutoff_sqd = noa::math::clamp(cutoff, 0.f, 0.5f);
            m_cutoff_sqd *= m_cutoff_sqd;

            if constexpr (traits::is_float3_v<shift_or_empty_type>)
                m_shift *= math::Constants<float>::PI2 / float3_t(m_shape);
        }

        NOA_IHD void operator()(index_type batch, index_type z, index_type y, index_type x) const noexcept { // x == u
            const index_type w = index2frequency<IS_DST_CENTERED>(z, m_shape[0]);
            const index_type v = index2frequency<IS_DST_CENTERED>(y, m_shape[1]);
            float3_t freq{w, v, x};
            freq /= m_f_shape; // [-0.5, 0.5]
            if (math::dot(freq, freq) > m_cutoff_sqd) {
                m_output(batch, z, y, x) = 0;
                return;
            }

            if constexpr (std::is_pointer_v<matrix_type>)
                freq = m_matrix[batch] * freq;
            else
                freq = m_matrix * freq;

            real_type conj = 1;
            if (freq[2] < 0.f) {
                freq = -freq;
                if constexpr (traits::is_complex_v<data_type>)
                    conj = -1;
            }

            freq[0] += 0.5f;
            freq[1] += 0.5f;
            freq *= m_f_shape;
            data_type value = m_input(freq, batch);
            if constexpr (traits::is_complex_v<data_type>)
                value.imag *= conj;
            else
                (void) conj;

            if constexpr (traits::is_complex_v<data_type> && !std::is_empty_v<shift_or_empty_type>) {
                if constexpr (std::is_pointer_v<shift_or_empty_type>) {
                    const float3_t shift = m_shift[batch] * math::Constants<float>::PI2 / float3_t(m_shape);
                    value *= phaseShift(shift, float3_t{w, v, x});
                } else {
                    value *= phaseShift(m_shift, float3_t{w, v, x});
                }
            }

            m_output(batch, z, y, x) = value;
        }

    private:
        interpolator_type m_input;
        accessor_type m_output;
        matrix_type m_matrix;
        shift_or_empty_type m_shift;
        index3_type m_shape;
        float3_t m_f_shape;
        float m_cutoff_sqd;
    };

    template<Remap REMAP, typename Index, typename Data, typename Matrix,
             typename ShiftOrEmpty, typename Interpolator, typename Offset>
    auto transform3D(const Interpolator& input,
                     const AccessorRestrict<Data, 4, Offset>& output, dim4_t shape,
                     Matrix matrix, ShiftOrEmpty shift, float cutoff) noexcept {
        return Transform3D<REMAP, Index, Data, Matrix, ShiftOrEmpty, Interpolator, Offset>(
                input, output, shape, matrix, shift, cutoff);
    }

    template<Remap REMAP, typename Index, typename Data, typename MatrixOrEmpty,
             typename ShiftOrEmpty, typename Interpolator, typename Offset>
    class TransformSymmetry3D {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        static_assert(REMAP_ & Layout::SRC_CENTERED &&
                      !(REMAP_ & Layout::SRC_FULL) &&
                      !(REMAP_ & Layout::DST_FULL));

        static_assert(traits::is_any_v<Data, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_any_v<MatrixOrEmpty, empty_t, float33_t>);
        static_assert(traits::is_any_v<ShiftOrEmpty, empty_t, float3_t>);
        static_assert(traits::is_int_v<Index>);

        using data_type = Data;
        using matrix_or_empty_type = MatrixOrEmpty;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;
        using shift_or_empty_type = ShiftOrEmpty;

        using real_type = traits::value_type_t<data_type>;
        using index2_type = Int2<index_type>;
        using index3_type = Int3<index_type>;
        using accessor_type = AccessorRestrict<data_type, 4, offset_type>;

    public:
        TransformSymmetry3D(interpolator_type input, accessor_type output, dim4_t shape,
                            matrix_or_empty_type matrix, const float33_t* symmetry_matrices,
                            index_type symmetry_count, float scaling,
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

            const auto i_shape = safe_cast<index3_type>(dim3_t(shape.get(1)));
            m_size_zy = {i_shape[0], i_shape[1]};
            m_f_shape = float3_t(i_shape / 2 * 2 + int3_t(i_shape == 1)); // if odd, n-1

            m_cutoff_sqd = noa::math::clamp(cutoff, 0.f, 0.5f);
            m_cutoff_sqd *= m_cutoff_sqd;

            if constexpr (traits::is_float3_v<shift_or_empty_type>)
                m_shift *= math::Constants<float>::PI2 / float3_t(i_shape);
        }

        NOA_IHD void operator()(index_type batch, index_type z, index_type y, index_type x) const noexcept { // x == u
            const index_type w = index2frequency<IS_DST_CENTERED>(z, m_size_zy[0]);
            const index_type v = index2frequency<IS_DST_CENTERED>(y, m_size_zy[1]);
            float3_t freq{w, v, x};
            freq /= m_f_shape; // [-0.5, 0.5]
            if (math::dot(freq, freq) > m_cutoff_sqd) {
                m_output(batch, z, y, x) = 0;
                return;
            }

            if constexpr (!std::is_empty_v<matrix_or_empty_type>)
                freq = m_matrix * freq;

            data_type value = interpolateFFT_(freq, batch);
            for (index_type ii = 0; ii < m_symmetry_count; ++ii) {
                const float3_t i_freq(m_symmetry_matrices[ii] * freq);
                value += interpolateFFT_(i_freq, batch);
            }

            value *= m_scaling;
            if constexpr (traits::is_complex_v<data_type> && !std::is_empty_v<shift_or_empty_type>)
                value *= phaseShift(m_shift, float3_t{w, v, x});

            m_output(batch, z, y, x) = value;
        }

    private:
        // Interpolates the (complex) value at the normalized frequency "freq".
        NOA_IHD data_type interpolateFFT_(float3_t freq, index_type i) const noexcept{
            real_type conj = 1;
            if (freq[2] < 0.f) {
                freq = -freq;
                if constexpr (traits::is_complex_v<data_type>)
                    conj = -1;
            }

            freq[0] += 0.5f;
            freq[1] += 0.5f;
            freq *= m_f_shape;
            data_type value = m_input(freq, i);
            if constexpr (traits::is_complex_v<data_type>)
                value.imag *= conj;
            else
                (void) conj;
            return value;
        }

    private:
        interpolator_type m_input;
        accessor_type m_output;
        matrix_or_empty_type m_matrix;
        shift_or_empty_type m_shift;
        const float33_t* m_symmetry_matrices;
        index_type m_symmetry_count;
        index2_type m_size_zy;
        float3_t m_f_shape;
        float m_cutoff_sqd;
        float m_scaling;
    };

    template<Remap REMAP, typename Index, typename Data, typename MatrixOrEmpty,
             typename ShiftOrEmpty, typename Interpolator, typename Offset>
    auto transformSymmetry3D(const Interpolator& input,
                             const AccessorRestrict<Data, 4, Offset>& output,
                             dim4_t shape, MatrixOrEmpty matrix,
                             const float33_t* symmetry_matrices, Index symmetry_count, float scaling,
                             ShiftOrEmpty shift, float cutoff) noexcept {
        return TransformSymmetry3D<
                REMAP, Index, Data, MatrixOrEmpty, ShiftOrEmpty, Interpolator, Offset>(
                input, output, shape, matrix, symmetry_matrices, symmetry_count, scaling, shift, cutoff);
    }
}
