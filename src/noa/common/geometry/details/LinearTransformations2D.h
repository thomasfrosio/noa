#pragma once
#include "noa/common/Types.h"

namespace noa::geometry::details {
    template<bool LAYERED,
             typename data_t, typename matrix_t, typename interpolator_t,
             typename index_t, typename offset_t>
    class Transform2D {
    public:
        static_assert(traits::is_any_v<matrix_t, float23_t, const float23_t*, const float33_t*>);
        static_assert(traits::is_any_v<data_t, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_int_v<index_t>);
        using output_accessor = AccessorRestrict<data_t, 3, offset_t>;

    public:
        Transform2D(interpolator_t input, output_accessor output, matrix_t matrix) noexcept
                : m_input(input), m_output(output), m_matrix(matrix) {
            if constexpr (std::is_pointer_v<matrix_t>) {
                NOA_ASSERT(matrix != nullptr);
            }
        }

        NOA_IHD void operator()(index_t i, index_t y, index_t x) const noexcept {
            const float3_t pos{y, x, 1.f};
            if constexpr (LAYERED)
                m_output(i, y, x) = m_input(this->transform_(i, pos), i);
            else
                m_output(i, y, x) = m_input(this->transform_(i, pos));
        }

    private:
        NOA_IHD float2_t transform_(index_t i, float3_t pos) const noexcept {
            if constexpr (traits::is_any_v<matrix_t, const float23_t*, const float33_t*>) {
                const float23_t matrix(m_matrix[i]);
                const float2_t coordinates = matrix * pos;
                return coordinates;
            } else if constexpr (std::is_same_v<matrix_t, float23_t>) {
                const float2_t coordinates = m_matrix * pos;
                return coordinates;
            }
        }

    private:
        interpolator_t m_input;
        output_accessor m_output;
        matrix_t m_matrix;
    };

    template<bool LAYERED, typename index_t,
             typename data_t, typename matrix_t,
             typename interpolator_t, typename offset_t>
    auto transform2D(const interpolator_t& input,
                     const AccessorRestrict<data_t, 3, offset_t>& output,
                     matrix_t matrix) noexcept {
        return Transform2D<LAYERED, data_t, matrix_t, interpolator_t, index_t, offset_t>(input, output, matrix);
    }
}

namespace noa::geometry::details {
    template<bool LAYERED,
             typename data_t, typename interpolator_t,
             typename index_t, typename offset_t>
    class TransformSymmetry2D {
    public:
        static_assert(traits::is_any_v<data_t, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_int_v<index_t>);
        using output_accessor = AccessorRestrict<data_t, 3, offset_t>;

    public:
        TransformSymmetry2D(interpolator_t input, output_accessor output,
                            float2_t shift, float22_t matrix, float2_t center,
                            const float33_t* symmetry_matrices, int32_t symmetry_count,
                            float scaling) noexcept
                : m_input(input), m_output(output), m_matrix(matrix), m_shift(shift),
                  m_center(center), m_symmetry_matrices(symmetry_matrices),
                  m_symmetry_count(symmetry_count), m_scaling(scaling) {
            NOA_ASSERT(symmetry_matrices != nullptr);
        }

        NOA_IHD void operator()(index_t i, index_t y, index_t x) const noexcept {
            float2_t coordinates{y, x};
            coordinates -= m_center;
            coordinates = m_matrix * coordinates;
            data_t value = interpolate_(coordinates + m_center + m_shift, i);
            for (index_t ii = 0; ii < m_symmetry_count; ++ii) {
                const float33_t& m = m_symmetry_matrices[ii];
                const float22_t sym_matrix{m[1][1], m[1][2],
                                           m[2][1], m[2][2]};
                const float2_t i_coordinates = sym_matrix * coordinates;
                value += interpolate_(i_coordinates + m_center + m_shift, i);
            }

            m_output(i, y, x) = value * m_scaling;
        }

    private:
        NOA_IHD data_t interpolate_(float2_t pos, index_t i) const noexcept {
            if constexpr (LAYERED)
                return m_input(pos, i);
            else
                return m_input(pos);
        }

    private:
        interpolator_t m_input;
        output_accessor m_output;
        float22_t m_matrix;
        float2_t m_shift;
        float2_t m_center;

        const float33_t* m_symmetry_matrices;
        index_t m_symmetry_count;
        float m_scaling;
    };

    template<bool LAYERED, typename index_t,
             typename data_t, typename interpolator_t, typename offset_t>
    auto transformSymmetry2D(interpolator_t input, const AccessorRestrict<data_t, 3, offset_t>& output,
                             float2_t shift, float22_t matrix, float2_t center,
                             const float33_t* symmetry_matrices, int32_t symmetry_count,
                             float scaling) noexcept {
        return TransformSymmetry2D<LAYERED, data_t, interpolator_t, index_t, offset_t>(
                input, output, shift, matrix, center, symmetry_matrices, symmetry_count, scaling);
    }
}

namespace noa::geometry::details {
    template<bool LAYERED,
             typename data_t, typename interpolator_t,
             typename index_t, typename offset_t>
    class Symmetry2D {
    public:
        static_assert(traits::is_any_v<data_t, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_int_v<index_t>);
        using output_accessor = AccessorRestrict<data_t, 3, offset_t>;

    public:
        Symmetry2D(interpolator_t input, output_accessor output,
                   float2_t center, const float33_t* symmetry_matrices,
                   int32_t symmetry_count, float scaling) noexcept
                : m_input(input), m_output(output),
                  m_center(center), m_symmetry_matrices(symmetry_matrices),
                  m_symmetry_count(symmetry_count), m_scaling(scaling) {
            NOA_ASSERT(symmetry_matrices != nullptr);
        }

        NOA_IHD void operator()(index_t i, index_t y, index_t x) const noexcept {
            float2_t coordinates{y, x};
            data_t value = interpolate_(coordinates, i);
            coordinates -= m_center;
            for (index_t ii = 0; ii < m_symmetry_count; ++ii) {
                const float33_t& m = m_symmetry_matrices[ii];
                float22_t sym_matrix{m[1][1], m[1][2],
                                     m[2][1], m[2][2]};
                float2_t i_coordinates = sym_matrix * coordinates;
                value += interpolate_(i_coordinates + m_center, i);
            }

            m_output(i, y, x) = value * m_scaling;
        }

    private:
        NOA_FHD data_t interpolate_(float2_t pos, index_t i) const noexcept {
            if constexpr (LAYERED)
                return m_input(pos, i);
            else
                return m_input(pos);
        }

    private:
        interpolator_t m_input;
        output_accessor m_output;
        float2_t m_center;

        const float33_t* m_symmetry_matrices;
        index_t m_symmetry_count;
        float m_scaling;
    };

    template<bool LAYERED, typename index_t,
             typename data_t, typename interpolator_t, typename offset_t>
    auto symmetry2D(interpolator_t input, const AccessorRestrict<data_t, 3, offset_t>& output,
                    float2_t center, const float33_t* symmetry_matrices, int32_t symmetry_count,
                    float scaling) noexcept {
        return Symmetry2D<LAYERED, data_t, interpolator_t, index_t, offset_t>(
                input, output, center, symmetry_matrices, symmetry_count, scaling);
    }
}

namespace noa::geometry::fft::details {
    using Remap = ::noa::fft::Remap;

    template<bool IS_CENTERED, typename index_t>
    [[nodiscard]] NOA_FHD index_t index2frequency(index_t index, index_t size) {
        if constexpr (IS_CENTERED)
            return index - size / 2;
        else
            return index < (size + 1) / 2 ? index : index - size;
        return index_t{}; // unreachable
    }

    [[nodiscard]] NOA_FHD cfloat_t phaseShift(float2_t shift, float2_t freq) {
        const float factor = -math::dot(shift, freq);
        cfloat_t phase_shift;
        noa::math::sincos(factor, &phase_shift.imag, &phase_shift.real);
        return phase_shift;
    }

    template<Remap REMAP, bool LAYERED, typename index_t,
             typename data_t, typename matrix_t,
             typename shift_or_empty_t, typename interpolator_t,
             typename offset_t>
    class Transform2D {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        static_assert(REMAP_ & Layout::SRC_CENTERED &&
                      !(REMAP_ & Layout::SRC_FULL) &&
                      !(REMAP_ & Layout::DST_FULL));

        static_assert(traits::is_any_v<matrix_t, float22_t, const float22_t*>);
        static_assert(traits::is_any_v<data_t, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_any_v<shift_or_empty_t, empty_t, float2_t, const float2_t*>);
        static_assert(traits::is_int_v<index_t>);

        using real_t = traits::value_type_t<data_t>;
        using index2_t = Int2<index_t>;
        using output_accessor = AccessorRestrict<data_t, 3, offset_t>;

    public:
        Transform2D(interpolator_t input, output_accessor output, dim4_t shape,
                    matrix_t matrix, shift_or_empty_t shift, float cutoff) noexcept
                : m_input(input), m_output(output),
                  m_matrix(matrix), m_shift(shift) {

            if constexpr (std::is_pointer_v<matrix_t>) {
                NOA_ASSERT(matrix != nullptr);
            } else if constexpr (std::is_pointer_v<shift_or_empty_t>) {
                NOA_ASSERT(shift != nullptr);
            }
            NOA_ASSERT(shape[1] == 1);

            m_shape = safe_cast<index2_t>(dim2_t(shape.get(2)));
            m_f_shape = float2_t(m_shape / 2 * 2 + int2_t(m_shape == 1)); // if odd, n-1

            m_cutoff_sqd = noa::math::clamp(cutoff, 0.f, 0.5f);
            m_cutoff_sqd *= m_cutoff_sqd;

            if constexpr (traits::is_float2_v<shift_or_empty_t>)
                m_shift *= math::Constants<float>::PI2 / float2_t(m_shape);
        }

        NOA_IHD void operator()(index_t i, index_t y, index_t x) const noexcept { // x == u
            // Compute the frequency corresponding to the gid and inverse transform.
            const int32_t v = index2frequency<IS_DST_CENTERED>(y, m_shape[0]);
            float2_t freq{v, x};
            freq /= m_f_shape; // [-0.5, 0.5]
            if (math::dot(freq, freq) > m_cutoff_sqd) {
                m_output(i, y, x) = 0;
                return;
            }

            if constexpr (std::is_pointer_v<matrix_t>)
                freq = m_matrix[i] * freq;
            else
                freq = m_matrix * freq;

            // Non-redundant transform, so flip to the valid Hermitian pair, if necessary.
            real_t conj = 1;
            if (freq[1] < 0.f) {
                freq = -freq;
                if constexpr (traits::is_complex_v<data_t>)
                    conj = -1;
            }

            // Convert back to index and fetch value from input texture.
            freq[0] += 0.5f; // [0, 1]
            freq *= m_f_shape; // [0, N-1]
            data_t value = interpolate_(freq, i);
            if constexpr (traits::is_complex_v<data_t>)
                value.imag *= conj;
            else
                (void) conj;

            // Phase shift the interpolated value.
            if constexpr (traits::is_complex_v<data_t> && !std::is_empty_v<shift_or_empty_t>) {
                if constexpr (std::is_pointer_v<shift_or_empty_t>) {
                    const float2_t shift = m_shift[i] * math::Constants<float>::PI2 / float2_t(m_shape);
                    value *= phaseShift(shift, float2_t{v, x});
                } else {
                    value *= phaseShift(m_shift, float2_t{v, x});
                }
            }

            m_output(i, y, x) = value;
        }

    private:
        NOA_IHD data_t interpolate_(float2_t frequency, index_t i) const noexcept {
            if constexpr (LAYERED)
                return m_input(frequency, i);
            else
                return m_input(frequency);
        }

    private:
        interpolator_t m_input;
        output_accessor m_output;
        matrix_t m_matrix;
        shift_or_empty_t m_shift;
        index2_t m_shape;
        float2_t m_f_shape;
        float m_cutoff_sqd;
    };

    template<Remap REMAP, bool LAYERED, typename index_t,
             typename data_t, typename matrix_t,
             typename shift_or_empty_t, typename interpolator_t,
             typename offset_t>
    auto transform2D(const interpolator_t& input,
                     const AccessorRestrict<data_t, 3, offset_t>& output,
                     dim4_t shape, matrix_t matrix, shift_or_empty_t shift, float cutoff) noexcept {
        return Transform2D<REMAP, LAYERED, index_t, data_t, matrix_t, shift_or_empty_t, interpolator_t, offset_t>(
                input, output, shape, matrix, shift, cutoff);
    }

    template<Remap REMAP, bool LAYERED, typename index_t,
            typename data_t, typename matrix_or_empty_t,
            typename shift_or_empty_t, typename interpolator_t,
            typename offset_t>
    class TransformSymmetry2D {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        static constexpr bool IS_DST_CENTERED = REMAP_ & Layout::DST_CENTERED;
        static_assert(REMAP_ & Layout::SRC_CENTERED &&
                      !(REMAP_ & Layout::SRC_FULL) &&
                      !(REMAP_ & Layout::DST_FULL));

        static_assert(traits::is_any_v<data_t, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_any_v<matrix_or_empty_t, empty_t, float22_t>);
        static_assert(traits::is_any_v<shift_or_empty_t, empty_t, float2_t>);
        static_assert(traits::is_int_v<index_t>);

        using real_t = traits::value_type_t<data_t>;
        using index2_t = Int2<index_t>;
        using output_accessor = AccessorRestrict<data_t, 3, offset_t>;

    public:
        TransformSymmetry2D(interpolator_t input, output_accessor output, dim4_t shape,
                            matrix_or_empty_t matrix, const float33_t* symmetry_matrices,
                            index_t symmetry_count, float scaling,
                            shift_or_empty_t shift, float cutoff) noexcept
                : m_input(input), m_output(output),
                  m_matrix(matrix), m_shift(shift),
                  m_symmetry_matrices(symmetry_matrices),
                  m_symmetry_count(symmetry_count),
                  m_scaling(scaling) {

            if constexpr (std::is_pointer_v<matrix_or_empty_t>) {
                NOA_ASSERT(matrix != nullptr);
            } else if constexpr (std::is_pointer_v<shift_or_empty_t>) {
                NOA_ASSERT(shift != nullptr);
            }
            NOA_ASSERT(shape[1] == 1);

            const auto i_shape = safe_cast<index2_t>(dim2_t(shape.get(2)));
            m_size_y = i_shape[0];
            m_f_shape = float2_t(i_shape / 2 * 2 + int2_t(i_shape == 1)); // if odd, n-1

            m_cutoff_sqd = noa::math::clamp(cutoff, 0.f, 0.5f);
            m_cutoff_sqd *= m_cutoff_sqd;

            if constexpr (traits::is_float2_v<shift_or_empty_t>)
                m_shift *= math::Constants<float>::PI2 / float2_t(i_shape);
        }

        NOA_IHD void operator()(index_t i, index_t y, index_t x) const noexcept { // x == u
            const int32_t v = index2frequency<IS_DST_CENTERED>(y, m_size_y);
            float2_t freq{v, x};
            freq /= m_f_shape; // [-0.5, 0.5]
            if (math::dot(freq, freq) > m_cutoff_sqd) {
                m_output(i, y, x) = 0;
                return;
            }

            if constexpr (!std::is_empty_v<matrix_or_empty_t>)
                freq = m_matrix * freq;

            data_t value = interpolateFFT_(freq, i);
            for (index_t ii = 0; ii < m_symmetry_count; ++ii) {
                const float33_t& m = m_symmetry_matrices[ii];
                const float22_t sym_matrix{m[1][1], m[1][2],
                                           m[2][1], m[2][2]};
                const float2_t i_freq = sym_matrix * freq;
                value += interpolateFFT_(i_freq, i);
            }

            value *= m_scaling;
            if constexpr (traits::is_complex_v<data_t> && !std::is_empty_v<shift_or_empty_t>)
                value *= phaseShift(m_shift, float2_t{v, x});

            m_output(i, y, x) = value;
        }

    private:
        // Interpolates the (complex) value at the normalized frequency "freq".
        NOA_IHD data_t interpolateFFT_(float2_t freq, index_t i) const noexcept{
            real_t conj = 1;
            if (freq[1] < 0.f) {
                freq = -freq;
                if constexpr (traits::is_complex_v<data_t>)
                    conj = -1;
            }

            freq[0] += 0.5f;
            freq *= m_f_shape;
            data_t value = interpolate_(freq, i);
            if constexpr (traits::is_complex_v<data_t>)
                value.imag *= conj;
            else
                (void) conj;
            return value;
        }

        NOA_IHD data_t interpolate_(float2_t frequency, index_t i) const noexcept {
            if constexpr (LAYERED)
                return m_input(frequency, i);
            else
                return m_input(frequency);
        }

    private:
        interpolator_t m_input;
        output_accessor m_output;
        matrix_or_empty_t m_matrix;
        shift_or_empty_t m_shift;
        const float33_t* m_symmetry_matrices;
        index_t m_symmetry_count;
        index_t m_size_y;
        float2_t m_f_shape;
        float m_cutoff_sqd;
        float m_scaling;
    };

    template<Remap REMAP, bool LAYERED, typename index_t,
            typename data_t, typename matrix_or_empty_t,
            typename shift_or_empty_t, typename interpolator_t,
            typename offset_t>
    auto transformSymmetry2D(const interpolator_t& input,
                             const AccessorRestrict<data_t, 3, offset_t>& output,
                             dim4_t shape, matrix_or_empty_t matrix,
                             const float33_t* symmetry_matrices, index_t symmetry_count, float scaling,
                             shift_or_empty_t shift, float cutoff) noexcept {
        return TransformSymmetry2D<
                REMAP, LAYERED, index_t, data_t, matrix_or_empty_t, shift_or_empty_t, interpolator_t, offset_t>(
                input, output, shape, matrix, symmetry_matrices, symmetry_count, scaling, shift, cutoff);
    }
}
