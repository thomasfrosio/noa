#pragma once
#include "noa/common/Types.h"

namespace noa::geometry::details {
    // Linear transformations for 2D arrays:
    //  * Works on real and complex arrays.
    //  * Use (truncated) affine matrices to transform coordinates: float23_t or float33_t.
    //    A single matrix can be used for every output batch. Otherwise, an array of matrices
    //    is expected. In this case, there should be as many matrices as they are output batches.
    //  * The matrices should be inverted since the inverse transformation are performed.
    //  * Multiple batches can be processed. The operator except a "batch" index, which is passed
    //    to the interpolator. It is up to the interpolator to decide what to do with this index.
    //    The interpolator can ignore that batch index, effectively broadcasting the input to
    //    every dimension of the output.
    template<typename Index, typename Data, typename Matrix, typename Interpolator, typename Offset>
    class Transform2D {
    public:
        static_assert(traits::is_any_v<Matrix, float23_t, const float23_t*, const float33_t*>);
        static_assert(traits::is_any_v<Data, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_int_v<Index>);

        using data_type = Data;
        using matrix_type = Matrix;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;
        using accessor_type = AccessorRestrict<data_type, 3, offset_type>;

    public:
        Transform2D(interpolator_type input, accessor_type output, matrix_type inv_matrix) noexcept
                : m_input(input), m_output(output), m_inv_matrix(inv_matrix) {
            if constexpr (std::is_pointer_v<Matrix>) {
                NOA_ASSERT(inv_matrix != nullptr);
            }
        }

        NOA_IHD void operator()(index_type batch, index_type y, index_type x) const noexcept {
            const float3_t coordinates{y, x, 1.f};
            if constexpr (traits::is_any_v<matrix_type, const float23_t*, const float33_t*>) {
                m_output(batch, y, x) = m_input(float23_t(m_inv_matrix[batch]) * coordinates, batch);
            } else if constexpr (std::is_same_v<matrix_type, float23_t>) {
                m_output(batch, y, x) = m_input(m_inv_matrix * coordinates, batch);
            }
        }

    private:
        interpolator_type m_input;
        accessor_type m_output;
        matrix_type m_inv_matrix;
    };

    template<typename Index, typename Data, typename Matrix, typename Interpolator, typename Offset>
    auto transform2D(const Interpolator& input,
                     const AccessorRestrict<Data, 3, Offset>& output,
                     Matrix inv_matrix) noexcept {
        return Transform2D<Index, Data, Matrix, Interpolator, Offset>(input, output, inv_matrix);
    }
}

namespace noa::geometry::details {
    // Linear 2D transformations, with symmetry.
    //  * Same as above, but a single transformation only.
    //  * After the first transformation, the symmetry matrices are applied.
    template<typename Index, typename Data, typename Interpolator, typename Offset>
    class TransformSymmetry2D {
    public:
        static_assert(traits::is_any_v<Data, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_int_v<Index>);

        using data_type = Data;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;
        using accessor_type = AccessorRestrict<data_type, 3, offset_type>;
        using real_type = traits::value_type_t<data_type>;

    public:
        TransformSymmetry2D(interpolator_type input, accessor_type output,
                            float2_t shift, float22_t matrix, float2_t center,
                            const float33_t* symmetry_matrices, index_type symmetry_count,
                            real_type scaling) noexcept
                : m_input(input), m_output(output), m_matrix(matrix), m_center_shift(center + shift),
                  m_center(center), m_symmetry_matrices(symmetry_matrices),
                  m_symmetry_count(symmetry_count), m_scaling(scaling) {
            NOA_ASSERT(symmetry_matrices != nullptr || symmetry_count == 0);
        }

        NOA_IHD void operator()(index_type batch, index_type y, index_type x) const noexcept {
            float2_t coordinates{y, x};
            coordinates -= m_center;
            coordinates = m_matrix * coordinates;
            data_type value = m_input(coordinates + m_center_shift, batch);
            for (index_type i = 0; i < m_symmetry_count; ++i) {
                const float33_t& m = m_symmetry_matrices[i];
                const float22_t sym_matrix{m[1][1], m[1][2],
                                           m[2][1], m[2][2]};
                const float2_t i_coordinates = sym_matrix * coordinates;
                value += m_input(i_coordinates + m_center_shift, batch);
            }
            m_output(batch, y, x) = value * m_scaling;
        }

    private:
        interpolator_type m_input;
        accessor_type m_output;
        float22_t m_matrix;
        float2_t m_center_shift;
        float2_t m_center;

        const float33_t* m_symmetry_matrices;
        index_type m_symmetry_count;
        real_type m_scaling;
    };

    template<typename Index, typename Data, typename Interpolator, typename Offset,
             typename Real = traits::value_type_t<Data>>
    auto transformSymmetry2D(Interpolator input, const AccessorRestrict<Data, 3, Offset>& output,
                             float2_t shift, float22_t matrix, float2_t center,
                             const float33_t* symmetry_matrices, Index symmetry_count,
                             Real scaling) noexcept {
        return TransformSymmetry2D<Index, Data, Interpolator, Offset>(
                input, output, shift, matrix, center, symmetry_matrices, symmetry_count, scaling);
    }
}

namespace noa::geometry::details {
    // 2D symmetry. Same as above, but without the affine transformation.
    template<typename Index, typename Data, typename Interpolator, typename Offset>
    class Symmetry2D {
    public:
        static_assert(traits::is_any_v<Data, float, double, cfloat_t, cdouble_t>);
        static_assert(traits::is_int_v<Index>);

        using data_type = Data;
        using interpolator_type = Interpolator;
        using index_type = Index;
        using offset_type = Offset;
        using accessor_type = AccessorRestrict<Data, 3, Offset>;
        using real_type = traits::value_type_t<data_type>;

    public:
        Symmetry2D(interpolator_type input, accessor_type output,
                   float2_t center, const float33_t* symmetry_matrices,
                   index_type symmetry_count, real_type scaling) noexcept
                : m_input(input), m_output(output),
                  m_center(center), m_symmetry_matrices(symmetry_matrices),
                  m_symmetry_count(symmetry_count), m_scaling(scaling) {
            NOA_ASSERT(symmetry_matrices != nullptr || symmetry_count == 0);
        }

        NOA_IHD void operator()(index_type batch, index_type y, index_type x) const noexcept {
            data_type value = m_input.at(batch, y, x); // skip interpolator if possible
            float2_t coordinates{y, x};
            coordinates -= m_center;
            for (index_type ii = 0; ii < m_symmetry_count; ++ii) {
                const float33_t& m = m_symmetry_matrices[ii];
                float22_t sym_matrix{m[1][1], m[1][2],
                                     m[2][1], m[2][2]};
                float2_t i_coordinates = sym_matrix * coordinates;
                value += m_input(i_coordinates + m_center, batch);
            }

            m_output(batch, y, x) = value * m_scaling;
        }

    private:
        interpolator_type m_input;
        accessor_type m_output;
        float2_t m_center;

        const float33_t* m_symmetry_matrices;
        index_type m_symmetry_count;
        real_type m_scaling;
    };

    template<typename Index, typename Data, typename Interpolator, typename Offset,
             typename Real = traits::value_type_t<Data>>
    auto symmetry2D(Interpolator input, const AccessorRestrict<Data, 3, Offset>& output,
                    float2_t center, const float33_t* symmetry_matrices, Index symmetry_count,
                    Real scaling) noexcept {
        return Symmetry2D<Index, Data, Interpolator, Offset>(
                input, output, center, symmetry_matrices, symmetry_count, scaling);
    }
}
