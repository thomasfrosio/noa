#pragma once

#include "noa/core/Types.hpp"
#include "noa/algorithms/Utilities.hpp"

namespace noa::algorithm::geometry {
    using Remap = ::noa::fft::Remap;

    template<Remap REMAP, typename Index, typename Offset, typename Value, typename Coord,
             typename GeomShape, typename MatrixOrEmpty, typename Functor>
    class Shape3D {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static_assert(!(u8_REMAP & Layout::SRC_HALF || u8_REMAP & Layout::DST_HALF));

        static_assert(noa::traits::is_any_v<MatrixOrEmpty, Mat33<Coord>, Mat34<Coord>, Empty>);
        static_assert(noa::traits::are_int_v<Index, Offset>);
        static_assert(noa::traits::is_real_v<Coord>);

        using index_type = Index;
        using offset_type = Offset;
        using value_type = Value;
        using coord_type = Coord;
        using geom_shape_type = GeomShape;
        using matrix_or_empty_type = MatrixOrEmpty;
        using functor_type = Functor;
        using input_accessor_type = Accessor<const Value, 4, offset_type>;
        using output_accessor_type = Accessor<Value, 4, offset_type>;
        using index3_type = Vec3<index_type>;
        using shape3_type = Shape3<index_type>;
        using shape4_type = Shape4<index_type>;
        using coord3_type = Vec3<coord_type>;

    public:
        Shape3D(const input_accessor_type& input,
                const output_accessor_type& output,
                const shape4_type& shape,
                const geom_shape_type& geom_shape,
                const matrix_or_empty_type& inv_matrix,
                const functor_type& functor)
                : m_input(input), m_output(output),
                  m_inv_matrix(inv_matrix),
                  m_geometric_shape(geom_shape),
                  m_shape(shape.pop_front()), m_batch(shape[0]),
                  m_functor(functor) {
            NOA_ASSERT((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || input.get() != output.get());
        }

        constexpr NOA_IHD void operator()(index_type i, index_type j, index_type k, index_type l) const noexcept {
            index3_type index{j, k, l};
            const index3_type i_idx = to_centered_indexes<REMAP>(index, m_shape);
            const index3_type o_idx = to_output_indexes<REMAP>(index, m_shape);
            const auto mask_value = m_geometric_shape(coord3_type(i_idx), m_inv_matrix);
            const auto value = m_input ? m_functor(m_input(i, i_idx[0], i_idx[1], i_idx[2]), mask_value) : mask_value;
            m_output(i, o_idx[0], o_idx[1], o_idx[2]) = value;
        }

        constexpr NOA_IHD void operator()(index_type j, index_type k, index_type l) const noexcept {
            index3_type index{j, k, l};
            const index3_type i_idx = to_centered_indexes<REMAP>(index, m_shape);
            const index3_type o_idx = to_output_indexes<REMAP>(index, m_shape);
            const auto mask = m_geometric_shape(coord3_type(i_idx), m_inv_matrix);
            for (index_type i = 0; i < m_batch; ++i)
                m_output[i](o_idx) = m_input ? m_functor(m_input[i](i_idx), mask) : mask;
        }

    public:
        input_accessor_type m_input;
        output_accessor_type m_output;
        matrix_or_empty_type m_inv_matrix;
        geom_shape_type m_geometric_shape;
        shape3_type m_shape;
        index_type m_batch;
        functor_type m_functor;
    };

    template<Remap REMAP, typename Index, typename Offset, typename Value, typename Coord,
             typename GeomShape, typename MatrixOrEmpty, typename Functor>
    class Shape2D {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto u8_REMAP = static_cast<uint8_t>(REMAP);
        static_assert(!(u8_REMAP & Layout::SRC_HALF || u8_REMAP & Layout::DST_HALF));

        static_assert(noa::traits::is_any_v<MatrixOrEmpty, Mat22<Coord>, Mat23<Coord>, Empty>);
        static_assert(noa::traits::are_int_v<Index, Offset>);
        static_assert(noa::traits::is_real_v<Coord>);

        using index_type = Index;
        using offset_type = Offset;
        using value_type = Value;
        using coord_type = Coord;
        using geom_shape_type = GeomShape;
        using matrix_or_empty_type = MatrixOrEmpty;
        using functor_type = Functor;
        using input_accessor_type = Accessor<const Value, 3, offset_type>;
        using output_accessor_type = Accessor<Value, 3, offset_type>;
        using index2_type = Vec2<index_type>;
        using shape2_type = Shape2<index_type>;
        using shape3_type = Shape3<index_type>;
        using coord2_type = Vec2<coord_type>;

    public:
        Shape2D(const input_accessor_type& input,
                const output_accessor_type& output,
                const shape3_type& shape,
                const geom_shape_type& geom_shape,
                const matrix_or_empty_type& inv_matrix,
                const functor_type& functor)
                : m_input(input), m_output(output),
                  m_inv_matrix(inv_matrix),
                  m_geometric_shape(geom_shape),
                  m_shape(shape.pop_front()), m_batch(shape[0]),
                  m_functor(functor) {
            NOA_ASSERT((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || input.get() != output.get());
        }

        constexpr NOA_IHD void operator()(index_type i, index_type k, index_type l) const noexcept {
            index2_type index{k, l};
            const index2_type i_idx = to_centered_indexes<REMAP>(index, m_shape);
            const index2_type o_idx = to_output_indexes<REMAP>(index, m_shape);
            const auto mask = m_geometric_shape(coord2_type(i_idx), m_inv_matrix);
            const auto value = m_input ? m_functor(m_input(i, i_idx[0], i_idx[1]), mask) : mask;
            m_output(i, o_idx[0], o_idx[1]) = value;
        }

        constexpr NOA_IHD void operator()(index_type k, index_type l) const noexcept {
            index2_type index{k, l};
            const index2_type i_idx = to_centered_indexes<REMAP>(index, m_shape);
            const index2_type o_idx = to_output_indexes<REMAP>(index, m_shape);
            const auto mask = m_geometric_shape(coord2_type(i_idx), m_inv_matrix);
            for (index_type i = 0; i < m_batch; ++i)
                m_output[i](o_idx) = m_input ? m_functor(m_input[i](i_idx), mask) : mask;
        }

    public:
        input_accessor_type m_input;
        output_accessor_type m_output;
        matrix_or_empty_type m_inv_matrix;
        geom_shape_type m_geometric_shape;
        shape2_type m_shape;
        index_type m_batch;
        functor_type m_functor;
    };
}

namespace noa::algorithm::geometry {
    template<Remap REMAP, typename Coord, typename Index, typename Offset, typename Value,
             typename GeomShape, typename MatrixOrEmpty, typename Functor>
    auto shape_3d(const Accessor<const Value, 4, Offset>& input,
                  const Accessor<Value, 4, Offset>& output,
                  const Shape4<Index>& shape,
                  const GeomShape& geom_shape,
                  const MatrixOrEmpty& inv_matrix,
                  const Functor& functor) {
        using output_t = Shape3D<REMAP, Index, Offset, Value, Coord, GeomShape, MatrixOrEmpty, Functor>;
        return output_t(input, output, shape, geom_shape, inv_matrix, functor);
    }

    template<Remap REMAP, typename Coord, typename Index, typename Offset, typename Value,
             typename GeomShape, typename MatrixOrEmpty, typename Functor>
    auto shape_2d(const Accessor<const Value, 3, Offset>& input,
                  const Accessor<Value, 3, Offset>& output,
                  const Shape3<Index>& shape,
                  const GeomShape& geom_shape,
                  const MatrixOrEmpty& inv_matrix,
                  const Functor& functor) {
        using output_t = Shape2D<REMAP, Index, Offset, Value, Coord, GeomShape, MatrixOrEmpty, Functor>;
        return output_t(input, output, shape, geom_shape, inv_matrix, functor);
    }
}
