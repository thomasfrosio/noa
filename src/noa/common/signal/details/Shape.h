#pragma once

#include "noa/common/Types.h"

namespace noa::signal::fft::details {
    // To compute the shape, we need centered coordinates, so FFTshift if the input isn't centered.
    template<noa::fft::Remap REMAP, typename Int>
    NOA_FHD Int3<Int> gid2CenteredIndexes(const Int3<Int>& gid, const Int4<Int>& shape) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & noa::fft::Layout::SRC_CENTERED;
        return {IS_SRC_CENTERED ? gid[0] : math::FFTShift(gid[0], shape[1]),
                IS_SRC_CENTERED ? gid[1] : math::FFTShift(gid[1], shape[2]),
                IS_SRC_CENTERED ? gid[2] : math::FFTShift(gid[2], shape[3])};
    }

    template<noa::fft::Remap REMAP, typename Int>
    NOA_FHD Int2<Int> gid2CenteredIndexes(const Int2<Int>& gid, const Int3<Int>& shape) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & noa::fft::Layout::SRC_CENTERED;
        return {IS_SRC_CENTERED ? gid[0] : math::FFTShift(gid[0], shape[1]),
                IS_SRC_CENTERED ? gid[1] : math::FFTShift(gid[1], shape[2])};
    }

    // For the output, we need to compare with the input. If there's no remap, then the indexes
    // match and we can use the gid. Otherwise, FFTshift for F2FC, or iFFTshift for FC2F.
    template<noa::fft::Remap REMAP, typename Int>
    NOA_FHD Int3<Int> gid2OutputIndexes(const Int3<Int>& gid, const Int4<Int>& shape) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & noa::fft::Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & noa::fft::Layout::DST_CENTERED;
        if constexpr (IS_SRC_CENTERED == IS_DST_CENTERED) {
            return gid;
        } else if constexpr (IS_SRC_CENTERED && !IS_DST_CENTERED) { // FC2F
            return {math::iFFTShift(gid[0], shape[1]),
                    math::iFFTShift(gid[1], shape[2]),
                    math::iFFTShift(gid[2], shape[3])};
        } else { // F2FC
            return {math::FFTShift(gid[0], shape[1]),
                    math::FFTShift(gid[1], shape[2]),
                    math::FFTShift(gid[2], shape[3])};
        }
    }

    template<noa::fft::Remap REMAP, typename Int>
    NOA_FHD Int2<Int> gid2OutputIndexes(const Int2<Int>& gid, const Int3<Int>& shape) {
        constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        constexpr bool IS_SRC_CENTERED = REMAP_ & noa::fft::Layout::SRC_CENTERED;
        constexpr bool IS_DST_CENTERED = REMAP_ & noa::fft::Layout::DST_CENTERED;
        if constexpr (IS_SRC_CENTERED == IS_DST_CENTERED) {
            return gid;
        } else if constexpr (IS_SRC_CENTERED && !IS_DST_CENTERED) { // FC2F
            return {math::iFFTShift(gid[0], shape[1]),
                    math::iFFTShift(gid[1], shape[2])};
        } else { // F2FC
            return {math::FFTShift(gid[0], shape[1]),
                    math::FFTShift(gid[1], shape[2])};
        }
    }
}

namespace noa::signal::fft::details {
    using Remap = ::noa::fft::Remap;

    template<Remap REMAP, typename Index, typename Coord,
             typename GeomShape, typename MatrixOrEmpty,
             typename Functor, typename Value, typename Offset>
    class Shape3D {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        static_assert(!(REMAP_ & Layout::SRC_HALF || REMAP_ & Layout::DST_HALF));

        static_assert(traits::is_any_v<MatrixOrEmpty, float33_t, float34_t, empty_t>);
        static_assert(traits::are_int_v<Index, Offset>);
        static_assert(traits::is_float_v<Coord>);

        using value_type = Value;
        using coord_type = Coord;
        using index_type = Index;
        using offset_type = Offset;
        using geom_shape_type = GeomShape;
        using matrix_or_empty_type = MatrixOrEmpty;
        using functor_type = Functor;
        using input_accessor_type = Accessor<const Value, 4, offset_type>;
        using output_accessor_type = Accessor<Value, 4, offset_type>;
        using index4_type = Int4<index_type>;
        using index3_type = Int3<index_type>;
        using coord3_type = Float3<coord_type>;

    public:
        Shape3D(const input_accessor_type& input,
                const output_accessor_type& output,
                const index4_type& shape,
                const geom_shape_type& geom_shape,
                const matrix_or_empty_type& inv_matrix,
                const functor_type& functor)
                : m_input(input), m_output(output),
                  m_inv_matrix(inv_matrix),
                  m_geom_shape(geom_shape),
                  m_shape(shape),
                  m_functor(functor) {
            NOA_ASSERT((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || input.get() != output.get());
        }

        constexpr NOA_IHD void operator()(index_type i, index_type j, index_type k, index_type l) const noexcept {
            index3_type index{j, k, l};
            const index3_type i_idx = gid2CenteredIndexes<REMAP>(index, m_shape);
            const index3_type o_idx = gid2OutputIndexes<REMAP>(index, m_shape);
            const auto mask = m_geom_shape(coord3_type(i_idx), m_inv_matrix);
            const auto value = m_input ? m_functor(m_input(i, i_idx[0], i_idx[1], i_idx[2]), mask) : mask;
            m_output(i, o_idx[0], o_idx[1], o_idx[2]) = value;
        }

        constexpr NOA_IHD void operator()(index_type j, index_type k, index_type l) const noexcept {
            index3_type index{j, k, l};
            const index3_type i_idx = gid2CenteredIndexes<REMAP>(index, m_shape);
            const index3_type o_idx = gid2OutputIndexes<REMAP>(index, m_shape);
            const auto mask = m_geom_shape(coord3_type(i_idx), m_inv_matrix);
            for (index_type batch = 0; batch < m_shape[0]; ++batch)
                m_output[batch](o_idx) = m_input ? m_functor(m_input[batch](i_idx), mask) : mask;
        }

    public:
        input_accessor_type m_input;
        output_accessor_type m_output;
        matrix_or_empty_type m_inv_matrix;
        geom_shape_type m_geom_shape;
        index4_type m_shape;
        functor_type m_functor;
    };

    template<Remap REMAP, typename Index, typename Coord = float,
             typename GeomShape, typename MatrixOrEmpty, typename Functor,
             typename Value, typename Offset>
    auto shape3D(const Accessor<const Value, 4, Offset>& input,
                 const Accessor<Value, 4, Offset>& output,
                 const Int4<Index>& shape,
                 const GeomShape& geom_shape,
                 const MatrixOrEmpty& inv_matrix,
                 const Functor& functor) {
        using output_t = Shape3D<REMAP, Index, Coord, GeomShape, MatrixOrEmpty, Functor, Value, Offset>;
        return output_t(input, output, shape, geom_shape, inv_matrix, functor);
    }
}

namespace noa::signal::fft::details {
    template<Remap REMAP, typename Index, typename Coord,
             typename GeomShape, typename MatrixOrEmpty,
             typename Functor, typename Value, typename Offset>
    class Shape2D {
    public:
        using Layout = ::noa::fft::Layout;
        static constexpr auto REMAP_ = static_cast<uint8_t>(REMAP);
        static_assert(!(REMAP_ & Layout::SRC_HALF || REMAP_ & Layout::DST_HALF));

        static_assert(traits::is_any_v<MatrixOrEmpty, float22_t, float23_t, empty_t>);
        static_assert(traits::are_int_v<Index, Offset>);
        static_assert(traits::is_float_v<Coord>);

        using value_type = Value;
        using coord_type = Coord;
        using index_type = Index;
        using offset_type = Offset;
        using geom_shape_type = GeomShape;
        using matrix_or_empty_type = MatrixOrEmpty;
        using functor_type = Functor;
        using input_accessor_type = Accessor<const Value, 3, offset_type>;
        using output_accessor_type = Accessor<Value, 3, offset_type>;
        using index3_type = Int3<index_type>;
        using index2_type = Int2<index_type>;
        using coord2_type = Float2<coord_type>;

    public:
        Shape2D(const input_accessor_type& input,
                const output_accessor_type& output,
                const index3_type& shape,
                const geom_shape_type& geom_shape,
                const matrix_or_empty_type& inv_matrix,
                const functor_type& functor)
                : m_input(input), m_output(output),
                  m_inv_matrix(inv_matrix),
                  m_geom_shape(geom_shape),
                  m_shape(shape),
                  m_functor(functor) {
            NOA_ASSERT((REMAP == Remap::F2F || REMAP == Remap::FC2FC) || input.get() != output.get());
        }

        constexpr NOA_IHD void operator()(index_type i, index_type k, index_type l) const noexcept {
            index2_type index{k, l};
            const index2_type i_idx = gid2CenteredIndexes<REMAP>(index, m_shape);
            const index2_type o_idx = gid2OutputIndexes<REMAP>(index, m_shape);
            const auto mask = m_geom_shape(coord2_type(i_idx), m_inv_matrix);
            const auto value = m_input ? m_functor(m_input(i, i_idx[0], i_idx[1]), mask) : mask;
            m_output(i, o_idx[0], o_idx[1]) = value;
        }

        constexpr NOA_IHD void operator()(index_type k, index_type l) const noexcept {
            index2_type index{k, l};
            const index2_type i_idx = gid2CenteredIndexes<REMAP>(index, m_shape);
            const index2_type o_idx = gid2OutputIndexes<REMAP>(index, m_shape);
            const auto mask = m_geom_shape(coord2_type(i_idx), m_inv_matrix);
            for (index_type batch = 0; batch < m_shape[0]; ++batch)
                m_output[batch](o_idx) = m_input ? m_functor(m_input[batch](i_idx), mask) : mask;
        }

    public:
        input_accessor_type m_input;
        output_accessor_type m_output;
        matrix_or_empty_type m_inv_matrix;
        geom_shape_type m_geom_shape;
        index3_type m_shape;
        functor_type m_functor;
    };

    template<Remap REMAP, typename Index, typename Coord = float,
             typename GeomShape, typename MatrixOrEmpty,
             typename Functor, typename Value, typename Offset>
    auto shape2D(const Accessor<const Value, 3, Offset>& input,
                 const Accessor<Value, 3, Offset>& output,
                 const Int3<Index>& shape,
                 const GeomShape& geom_shape,
                 const MatrixOrEmpty& inv_matrix,
                 const Functor& functor) {
        using output_t = Shape2D<REMAP, Index, Coord, GeomShape, MatrixOrEmpty, Functor, Value, Offset>;
        return output_t(input, output, shape, geom_shape, inv_matrix, functor);
    }
}

