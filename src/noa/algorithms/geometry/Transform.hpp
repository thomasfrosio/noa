#pragma once

#include "noa/core/Types.hpp"
#include "noa/core/geometry/Transform.hpp"

namespace noa::algorithm::geometry {
    // 2D or 3D affine transformations:
    //  * Works on real and complex arrays.
    //  * The interpolated value is static_cast to the value_type of the output, so the interpolator
    //    use a different precision for its value_type.
    //  * The floating-point precision of the transformation is equal to the precision of the matrices
    //    and should match the coord_type of the Interpolator.
    //  * Use (truncated) affine matrices to transform coordinates.
    //    A single matrix can be used for every output batch. Otherwise, an array of matrices
    //    is expected. In this case, there should be as many matrices as they are output batches.
    //  * The matrices should be inverted since the inverse transformation is performed.
    //  * Multiple batches can be processed. The operator expects a "batch" index, which is passed
    //    to the interpolator. It is up to the interpolator to decide what to do with this index.
    //    The interpolator can ignore that batch index, effectively broadcasting the input to
    //    every dimension of the output.
    template<size_t N, typename Index, typename Value, typename Matrix, typename Interpolator, typename Offset>
    class Transform {
    public:
        static_assert(noa::traits::is_int_v<Index>);
        static_assert(noa::traits::is_real_or_complex_v<Value>);
        static_assert(N == 2 || N == 3);

        using index_type = Index;
        using value_type = Value;
        using matrix_type = Matrix;
        using interpolator_type = Interpolator;
        using offset_type = Offset;

        using coord_type = typename Interpolator::coord_type;
        using mat23_type = Mat23<coord_type>;
        using mat33_type = Mat33<coord_type>;
        using mat34_type = Mat34<coord_type>;
        using mat44_type = Mat44<coord_type>;

        static_assert(std::is_same_v<coord_type, typename std::remove_pointer_t<Matrix>::value_type>);
        static_assert((N == 2 && noa::traits::is_any_v<Matrix, mat23_type, const mat23_type*, const mat33_type*>) ||
                      (N == 3 && noa::traits::is_any_v<Matrix, mat34_type, const mat34_type*, const mat44_type*>));

        using accessor_type = AccessorRestrict<value_type, N + 1, offset_type>;

    public:
        Transform(const interpolator_type& input,
                  const accessor_type& output,
                  const matrix_type& inv_matrix) noexcept
                : m_input(input), m_output(output), m_inv_matrix(inv_matrix) {
            if constexpr (std::is_pointer_v<Matrix>) {
                NOA_ASSERT(inv_matrix != nullptr);
            }
        }

        template<typename Void = void, typename = std::enable_if_t<(N == 2) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type y, index_type x) const noexcept {
            const Vec3<coord_type> coordinates{y, x, coord_type{1}};

            if constexpr (std::is_same_v<matrix_type, const mat23_type*>) {
                m_output(batch, y, x) = static_cast<value_type>(m_input(m_inv_matrix[batch] * coordinates, batch));

            } else if constexpr (std::is_same_v<matrix_type, const mat33_type*>) {
                const auto inv_matrix = noa::geometry::affine2truncated(m_inv_matrix[batch]);
                m_output(batch, y, x) = static_cast<value_type>(m_input(inv_matrix * coordinates, batch));

            } else if constexpr (noa::traits::is_mat23_v<matrix_type>) {
                m_output(batch, y, x) = static_cast<value_type>(m_input(m_inv_matrix * coordinates, batch));

            } else {
                static_assert(noa::traits::always_false_v<value_type>);
            }
        }

        template<typename Void = void, typename = std::enable_if_t<(N == 3) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type z, index_type y, index_type x) const noexcept {
            const Vec4<coord_type> coordinates{z, y, x, coord_type{1}};

            if constexpr (std::is_same_v<matrix_type, const mat34_type*>) {
                m_output(batch, z, y, x) = static_cast<value_type>(m_input(m_inv_matrix[batch] * coordinates, batch));

            } else if constexpr (std::is_same_v<matrix_type, const mat44_type*>) {
                const auto inv_matrix = noa::geometry::affine2truncated(m_inv_matrix[batch]);
                m_output(batch, z, y, x) = static_cast<value_type>(m_input(inv_matrix * coordinates, batch));

            } else if constexpr (noa::traits::is_mat34_v<matrix_type>) {
                m_output(batch, z, y, x) = static_cast<value_type>(m_input(m_inv_matrix * coordinates, batch));

            } else {
                static_assert(noa::traits::always_false_v<value_type>);
            }
        }

    private:
        interpolator_type m_input;
        accessor_type m_output;
        matrix_type m_inv_matrix;
    };

    // Linear 2D or 3D transformations, with symmetry.
    //  * Same as above, but a single transformation only.
    //  * After the first transformation, the symmetry matrices are applied.
    template<size_t N, typename Index, typename Value, typename Interpolator, typename Offset>
    class TransformSymmetry {
    public:
        static_assert(noa::traits::is_int_v<Index>);
        static_assert(noa::traits::is_real_or_complex_v<Value>);
        static_assert(N == 2 || N == 3);

        using index_type = Index;
        using value_type = Value;
        using interpolator_type = Interpolator;
        using offset_type = Offset;

        using coord_type = typename Interpolator::coord_type;
        using vec_type = Vec<coord_type, N>;
        using matrix_type = std::conditional_t<N == 2, Mat22<coord_type>, Mat33<coord_type>>;
        using accessor_type = AccessorRestrict<value_type, N + 1, offset_type>;
        using real_value_type = noa::traits::value_type_t<value_type>;

    public:
        TransformSymmetry(
                const interpolator_type& input, const accessor_type& output,
                const vec_type& shift, const matrix_type& matrix, const vec_type& center,
                const Float33* symmetry_matrices, index_type symmetry_count,
                real_value_type scaling) noexcept
                : m_input(input), m_output(output),
                  m_matrix(matrix), m_center_shift(center + shift),
                  m_center(center), m_symmetry_matrices(symmetry_matrices),
                  m_symmetry_count(symmetry_count), m_scaling(scaling) {
            NOA_ASSERT(symmetry_matrices != nullptr || symmetry_count == 0);
        }

        template<typename Void = void, typename = std::enable_if_t<(N == 2) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type y, index_type x) const noexcept {
            vec_type coordinates{y, x};
            coordinates -= m_center;
            coordinates = m_matrix * coordinates;
            auto value = m_input(coordinates + m_center_shift, batch);
            for (index_type i = 0; i < m_symmetry_count; ++i) {
                const Float33& m = m_symmetry_matrices[i]; // zyx matrix
                const matrix_type sym_matrix_2d{m[1][1], m[1][2],
                                                m[2][1], m[2][2]}; // remove z
                const vec_type i_coordinates = sym_matrix_2d * coordinates;
                value += m_input(i_coordinates + m_center_shift, batch);
            }
            m_output(batch, y, x) = static_cast<value_type>(value) * m_scaling;
        }

        template<typename Void = void, typename = std::enable_if_t<(N == 3) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type z, index_type y, index_type x) const noexcept {
            vec_type coordinates{z, y, x};
            coordinates -= m_center;
            coordinates = m_matrix * coordinates;
            auto value = static_cast<value_type>(m_input(coordinates + m_center_shift, batch));
            for (index_type i = 0; i < m_symmetry_count; ++i) {
                const vec_type i_coordinates = matrix_type(m_symmetry_matrices[i]) * coordinates;
                value += static_cast<value_type>(m_input(i_coordinates + m_center_shift, batch));
            }
            m_output(batch, z, y, x) = value * m_scaling;
        }

    private:
        interpolator_type m_input;
        accessor_type m_output;
        matrix_type m_matrix;
        vec_type m_center_shift;
        vec_type m_center;

        const Float33* m_symmetry_matrices;
        index_type m_symmetry_count;
        real_value_type m_scaling;
    };

    // 2D or 3D symmetry. Same as above, but without the affine transformation.
    template<size_t N, typename Index, typename Value, typename Interpolator, typename Offset>
    class Symmetry {
    public:
        static_assert(noa::traits::is_real_or_complex_v<Value>);
        static_assert(noa::traits::is_int_v<Index>);
        static_assert(N == 2 || N == 3);

        using index_type = Index;
        using value_type = Value;
        using interpolator_type = Interpolator;
        using offset_type = Offset;

        using coord_type = typename Interpolator::coord_type;
        using vec_type = Vec<coord_type, N>;
        using matrix_type = std::conditional_t<N == 2, Mat22<coord_type>, Mat33<coord_type>>;

        using accessor_type = AccessorRestrict<Value, N + 1, Offset>;
        using real_value_type = traits::value_type_t<value_type>;

    public:
        Symmetry(const interpolator_type& input, const accessor_type& output,
                 const vec_type& center, const Float33* symmetry_matrices,
                 index_type symmetry_count, real_value_type scaling) noexcept
                : m_input(input), m_output(output),
                  m_center(center), m_symmetry_matrices(symmetry_matrices),
                  m_symmetry_count(symmetry_count), m_scaling(scaling) {
            NOA_ASSERT(symmetry_matrices != nullptr || symmetry_count == 0);
        }

        template<typename Void = void, typename = std::enable_if_t<(N == 2) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type y, index_type x) const noexcept {
            auto value = m_input.at(batch, y, x); // skip interpolation if possible
            vec_type coordinates{y, x};
            coordinates -= m_center;
            for (index_type ii = 0; ii < m_symmetry_count; ++ii) {
                const Float33& m = m_symmetry_matrices[ii];
                const matrix_type sym_matrix{m[1][1], m[1][2],
                                             m[2][1], m[2][2]};
                vec_type i_coordinates = sym_matrix * coordinates;
                value += m_input(i_coordinates + m_center, batch);
            }
            m_output(batch, y, x) = static_cast<value_type>(value) * m_scaling;
        }

        template<typename Void = void, typename = std::enable_if_t<(N == 3) && std::is_void_v<Void>>>
        NOA_HD void operator()(index_type batch, index_type z, index_type y, index_type x) const noexcept {
            auto value = m_input.at(batch, z, y, x); // skip interpolation if possible
            vec_type coordinates{z, y, x};
            coordinates -= m_center;
            for (index_type i = 0; i < m_symmetry_count; ++i) {
                const vec_type i_coordinates = matrix_type(m_symmetry_matrices[i]) * coordinates;
                value += m_input(i_coordinates + m_center, batch);
            }
            m_output(batch, z, y, x) = static_cast<value_type>(value) * m_scaling;
        }

    private:
        interpolator_type m_input;
        accessor_type m_output;
        vec_type m_center;

        const Float33* m_symmetry_matrices;
        index_type m_symmetry_count;
        real_value_type m_scaling;
    };
}

namespace noa::algorithm::geometry {
    template<typename Index, typename Value, typename Matrix, typename Interpolator, typename Offset>
    auto transform_2d(const Interpolator& input,
                      const AccessorRestrict<Value, 3, Offset>& output,
                      const Matrix& inv_matrix) noexcept {
        return Transform<2, Index, Value, Matrix, Interpolator, Offset>(input, output, inv_matrix);
    }

    template<typename Index, typename Value, typename Matrix, typename Interpolator, typename Offset>
    auto transform_3d(const Interpolator& input,
                      const AccessorRestrict<Value, 4, Offset>& output,
                      const Matrix& matrix) noexcept {
        return Transform<3, Index, Value, Matrix, Interpolator, Offset>(input, output, matrix);
    }

    template<typename Index, typename Value, typename Interpolator, typename Coord, typename Offset,
             typename Real = traits::value_type_t<Value>>
    auto transform_symmetry_2d(
            const Interpolator& input, const AccessorRestrict<Value, 3, Offset>& output,
            const Vec2<Coord>& shift, const Mat22<Coord>& matrix, const Vec2<Coord>& center,
            const Float33* symmetry_matrices, Index symmetry_count,
            Real scaling) noexcept {
        return TransformSymmetry<2, Index, Value, Interpolator, Offset>(
                input, output, shift, matrix, center, symmetry_matrices, symmetry_count, scaling);
    }

    template<typename Index, typename Value, typename Interpolator, typename Coord, typename Offset,
             typename Real = traits::value_type_t<Value>>
    auto transform_symmetry_3d(
            const Interpolator& input, const AccessorRestrict<Value, 4, Offset>& output,
            const Vec3<Coord>& shift, const Mat33<Coord>& matrix, const Vec3<Coord>& center,
            const Float33* symmetry_matrices, Index symmetry_count,
            Real scaling) noexcept {
        return TransformSymmetry<3, Index, Value, Interpolator, Offset>(
                input, output, shift, matrix, center, symmetry_matrices, symmetry_count, scaling);
    }

    template<typename Index, typename Value, typename Interpolator, typename Coord, typename Offset,
             typename Real = traits::value_type_t<Value>>
    auto symmetry_2d(const Interpolator& input, const AccessorRestrict<Value, 3, Offset>& output,
                     const Vec2<Coord>& center, const Float33* symmetry_matrices, Index symmetry_count,
                     Real scaling) noexcept {
        return Symmetry<2, Index, Value, Interpolator, Offset>(
                input, output, center, symmetry_matrices, symmetry_count, scaling);
    }

    template<typename Index, typename Value, typename Interpolator, typename Coord, typename Offset,
             typename Real = traits::value_type_t<Value>>
    auto symmetry_3d(const Interpolator& input, const AccessorRestrict<Value, 4, Offset>& output,
                     const Vec3<Coord>& center, const Float33* symmetry_matrices, Index symmetry_count,
                     Real scaling) noexcept {
        return Symmetry<3, Index, Value, Interpolator, Offset>(
                input, output, center, symmetry_matrices, symmetry_count, scaling);
    }
}
