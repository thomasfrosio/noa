#pragma once
#include "noa/common/Types.h"

namespace noa::geometry::details {
    // Affine 3D transformations. Otherwise same as Transform2D.
    template<typename Index, typename Data, typename Matrix, typename Interpolator, typename Offset>
    class Transform3D {
    public:
        static_assert(traits::is_any_v<Matrix, float34_t, const float34_t*, const float44_t*>);
        static_assert(traits::is_any_v<Data, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_int_v<Index>);

        using data_type = Data;
        using matrix_type = Matrix;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;
        using accessor_type = AccessorRestrict<data_type, 4, offset_type>;

    public:
        Transform3D(interpolator_type input, accessor_type output, matrix_type matrix) noexcept
                : m_input(input), m_output(output), m_matrix(matrix) {
            if constexpr (std::is_pointer_v<matrix_type>) {
                NOA_ASSERT(matrix != nullptr);
            }
        }

        NOA_IHD void operator()(index_type batch, index_type z, index_type y, index_type x) const noexcept {
            const float4_t coordinates{z, y, x, 1.f};
            if constexpr (traits::is_any_v<matrix_type, const float34_t*, const float44_t*>) {
                m_output(batch, z, y, x) = m_input(float34_t(m_matrix[batch]) * coordinates, batch);
            } else if constexpr (std::is_same_v<matrix_type, float34_t>) {
                m_output(batch, z, y, x) = m_input(m_matrix * coordinates, batch);
            }
        }

    private:
        interpolator_type m_input;
        accessor_type m_output;
        matrix_type m_matrix;
    };

    template<typename Index, typename Data, typename Matrix, typename Interpolator, typename Offset>
    auto transform3D(const Interpolator& input,
                     const AccessorRestrict<Data, 4, Offset>& output,
                     Matrix matrix) noexcept {
        return Transform3D<Index, Data, Matrix, Interpolator, Offset>(input, output, matrix);
    }
}

namespace noa::geometry::details {
    template<typename Index, typename Data, typename Interpolator, typename Offset>
    class TransformSymmetry3D {
    public:
        static_assert(traits::is_any_v<Data, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_int_v<Index>);

        using data_type = Data;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;
        using accessor_type = AccessorRestrict<data_type, 4, offset_type>;
        using real_type = traits::value_type_t<data_type>;

    public:
        TransformSymmetry3D(interpolator_type input, accessor_type output,
                            float3_t shift, float33_t matrix, float3_t center,
                            const float33_t* symmetry_matrices, index_type symmetry_count,
                            real_type scaling) noexcept
                : m_input(input), m_output(output), m_matrix(matrix), m_center_shift(center + shift),
                  m_center(center), m_symmetry_matrices(symmetry_matrices),
                  m_symmetry_count(symmetry_count), m_scaling(scaling) {
            NOA_ASSERT(symmetry_matrices != nullptr || symmetry_count == 0);
        }

        NOA_IHD void operator()(index_type batch, index_type z, index_type y, index_type x) const noexcept {
            float3_t coordinates{z, y, x};
            coordinates -= m_center;
            coordinates = m_matrix * coordinates;
            data_type value = m_input(coordinates + m_center_shift, batch);
            for (index_type i = 0; i < m_symmetry_count; ++i) {
                const float3_t i_coordinates = m_symmetry_matrices[i] * coordinates;
                value += m_input(i_coordinates + m_center_shift, batch);
            }

            m_output(batch, z, y, x) = value * m_scaling;
        }

    private:
        interpolator_type m_input;
        accessor_type m_output;
        float33_t m_matrix;
        float3_t m_center_shift;
        float3_t m_center;

        const float33_t* m_symmetry_matrices;
        index_type m_symmetry_count;
        real_type m_scaling;
    };

    template<typename Index, typename Data, typename Interpolator, typename Offset,
             typename Real = traits::value_type_t<Data>>
    auto transformSymmetry3D(Interpolator input, const AccessorRestrict<Data, 4, Offset>& output,
                             float3_t shift, float33_t matrix, float3_t center,
                             const float33_t* symmetry_matrices, Index symmetry_count,
                             Real scaling) noexcept {
        return TransformSymmetry3D<Index, Data, Interpolator, Offset>(
                input, output, shift, matrix, center, symmetry_matrices, symmetry_count, scaling);
    }
}

namespace noa::geometry::details {
    template<typename Index, typename Data, typename Interpolator, typename Offset>
    class Symmetry3D {
    public:
        static_assert(traits::is_any_v<Data, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_int_v<Index>);

        using data_type = Data;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;
        using accessor_type = AccessorRestrict<data_type, 4, offset_type>;
        using real_type = traits::value_type_t<data_type>;

    public:
        Symmetry3D(interpolator_type input, accessor_type output,
                   float3_t center, const float33_t* symmetry_matrices,
                   index_type symmetry_count, real_type scaling) noexcept
                : m_input(input), m_output(output),
                  m_center(center), m_symmetry_matrices(symmetry_matrices),
                  m_symmetry_count(symmetry_count), m_scaling(scaling) {
            NOA_ASSERT(symmetry_matrices != nullptr || symmetry_count == 0);
        }

        NOA_IHD void operator()(index_type batch, index_type z, index_type y, index_type x) const noexcept {
            float3_t coordinates{z, y, x};
            data_type value = m_input(coordinates, batch);
            coordinates -= m_center;
            for (index_type i = 0; i < m_symmetry_count; ++i) {
                const float3_t i_coordinates = m_symmetry_matrices[i] * coordinates;
                value += m_input(i_coordinates + m_center, batch);
            }

            m_output(batch, z, y, x) = value * m_scaling;
        }

    private:
        interpolator_type m_input;
        accessor_type m_output;
        float3_t m_center;

        const float33_t* m_symmetry_matrices;
        index_type m_symmetry_count;
        real_type m_scaling;
    };

    template<typename Index, typename Data, typename Interpolator, typename Offset,
             typename Real = traits::value_type_t<Data>>
    auto symmetry3D(Interpolator input, const AccessorRestrict<Data, 4, Offset>& output,
                    float3_t center, const float33_t* symmetry_matrices, Index symmetry_count,
                    Real scaling) noexcept {
        return Symmetry3D<Index, Data, Interpolator, Offset>(
                input, output, center, symmetry_matrices, symmetry_count, scaling);
    }
}
