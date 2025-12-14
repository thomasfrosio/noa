#pragma once

#include "noa/unified/Array.hpp"
#include "noa/unified/Indexing.hpp"
#include "noa/unified/Iwise.hpp"

namespace noa::details {
    /// Extract subregions from one or multiple arrays.
    /// \details Subregions are defined by their 3d shape and their 2d (hw) or 4d (batch + dhw) origins.
    ///          If the subregion falls (even partially) out of the input bounds, the border mode is used
    ///          to handle that case.
    /// \note The origins dimensions might not correspond to the input/subregion dimensions because of the
    ///       rearranging before the index-wise transformation. Thus, this operator keeps the dimension "order"
    ///       and rearranges the origin on-the-fly (instead of allocating a new "origins" vector).
    template<Border MODE,
             nt::integer Index,
             nt::vec_integer_size<2, 4> Origins,
             nt::readable_nd<4> InputAccessor,
             nt::writable_nd<4> SubregionAccessor>
    class ExtractSubregion {
    public:
        using input_accessor_type = std::remove_const_t<InputAccessor>;
        using subregion_accessor_type = std::remove_const_t<SubregionAccessor>;
        using subregion_value_type = nt::value_type_t<subregion_accessor_type>;
        using index_type = std::remove_const_t<Index>;

        using origins_type = std::remove_const_t<Origins>;
        using origins_pointer_type = const origins_type*;

        using index4_type = Vec<index_type, 4>;
        using shape4_type = Shape<index_type, 4>;
        using subregion_value_or_empty_type = std::conditional_t<MODE == Border::VALUE, subregion_value_type, Empty>;

    public:
        constexpr ExtractSubregion(
            const input_accessor_type& input_accessor,
            const subregion_accessor_type& subregion_accessor,
            const shape4_type& input_shape,
            origins_pointer_type origins,
            subregion_value_type cvalue,
            const origins_type& order
        ) :
            m_input(input_accessor),
            m_subregions(subregion_accessor),
            m_subregion_origins(origins),
            m_input_shape(input_shape),
            m_order(order)
        {
            if constexpr (not nt::empty<subregion_value_or_empty_type>)
                m_cvalue = cvalue;
            else
                (void) cvalue;
        }

        constexpr void operator()(const index4_type& output_indices) const {
            // TODO For CUDA, the origins could copied to constant memory.
            //      Although these can be loaded in a single vectorized instruction.
            const auto corner_left = m_subregion_origins[output_indices[0]].reorder(m_order).template as<index_type>();

            index4_type input_indices;
            if constexpr (origins_type::SIZE == 2) {
                input_indices = {
                        0,
                        output_indices[1],
                        output_indices[2] + corner_left[0],
                        output_indices[3] + corner_left[1],
                };
            } else if constexpr (origins_type::SIZE == 4) {
                input_indices = {
                        corner_left[0],
                        output_indices[1] + corner_left[1],
                        output_indices[2] + corner_left[2],
                        output_indices[3] + corner_left[3],
                };
            } else {
                static_assert(nt::always_false<>);
            }

            if constexpr (MODE == Border::NOTHING) {
                if (ni::is_inbound(m_input_shape, input_indices))
                    m_subregions(output_indices) = cast_or_abs_squared<subregion_value_type>(m_input(input_indices));

            } else if constexpr (MODE == Border::ZERO) {
                m_subregions(output_indices) = ni::is_inbound(m_input_shape, input_indices) ?
                                               cast_or_abs_squared<subregion_value_type>(m_input(input_indices)) :
                                               subregion_value_type{};

            } else if constexpr (MODE == Border::VALUE) {
                m_subregions(output_indices) = ni::is_inbound(m_input_shape, input_indices) ?
                                               cast_or_abs_squared<subregion_value_type>(m_input(input_indices)) :
                                               m_cvalue;

            } else {
                const index4_type bounded_indices = ni::index_at<MODE>(input_indices, m_input_shape);
                m_subregions(output_indices) = cast_or_abs_squared<subregion_value_type>(m_input(bounded_indices));
            }
        }

    private:
        input_accessor_type m_input;
        subregion_accessor_type m_subregions;
        origins_pointer_type m_subregion_origins;
        shape4_type m_input_shape;
        origins_type m_order;
        NOA_NO_UNIQUE_ADDRESS subregion_value_or_empty_type m_cvalue;
    };

    /// Insert subregions into one or multiple arrays.
    /// \details This works as expected and is similar to ExtractSubregion. Subregions can be (even partially) out
    ///          of the output bounds. The only catch here is that overlapped subregions are not explicitly supported
    ///          since it is not clear what we want in these cases (add?), so for now, just ignore it out.
    template<nt::integer Index,
             nt::vec_integer_size<2, 4> Origins,
             nt::readable_nd<4> SubregionAccessor,
             nt::writable_nd<4> OutputAccessor>
    class InsertSubregion {
    public:
        using output_accessor_type = std::remove_const_t<OutputAccessor>;
        using output_value_type = nt::value_type_t<output_accessor_type>;
        using subregion_accessor_type = std::remove_const_t<SubregionAccessor>;
        using index_type = std::remove_const_t<Index>;

        using origins_type = std::remove_const_t<Origins>;
        using origins_pointer_type = const origins_type*;

        using index4_type = Vec<Index, 4>;
        using shape4_type = Shape<Index, 4>;

    public:
        constexpr InsertSubregion(
            const subregion_accessor_type& subregion_accessor,
            const output_accessor_type& output_accessor,
            const shape4_type& output_shape,
            origins_pointer_type origins,
            const origins_type& order
        ) :
            m_output(output_accessor),
            m_subregions(subregion_accessor),
            m_subregion_origins(origins),
            m_output_shape(output_shape),
            m_order(order) {}

        constexpr void operator()(const index4_type& input_indices) const {
            // TODO For CUDA, the origins could copied to constant memory.
            //      Although these can be loaded in a single vectorized instruction.
            const auto corner_left = m_subregion_origins[input_indices[0]].reorder(m_order).template as<index_type>();

            index4_type output_indices;
            if constexpr (origins_type::SIZE == 2) {
                output_indices = {
                        0,
                        input_indices[1],
                        input_indices[2] + corner_left[0],
                        input_indices[3] + corner_left[1],
                };
            } else if constexpr (origins_type::SIZE == 4) {
                output_indices = {
                        corner_left[0],
                        input_indices[1] + corner_left[1],
                        input_indices[2] + corner_left[2],
                        input_indices[3] + corner_left[3],
                };
            } else {
                static_assert(nt::always_false<>);
            }

            // We assume no overlap in the output between subregions.
            if (ni::is_inbound(m_output_shape, output_indices))
                m_output(output_indices) = cast_or_abs_squared<output_value_type>(m_subregions(input_indices));
        }

    private:
        output_accessor_type m_output;
        subregion_accessor_type m_subregions;
        origins_pointer_type m_subregion_origins;
        shape4_type m_output_shape;
        origins_type m_order;
    };

}

namespace noa {
    /// Extracts one or multiple {1|2|3}d subregions at various locations in the input array.
    /// \param[in] input        Input array to extract from.
    /// \param[out] subregions  Output subregion(s).
    ///                         Input values are cast to the subregion value type.
    ///                         It should not overlap with \p input.
    /// \param[in] origins      Contiguous vector with the (BD)HW indexes of the subregions to extract. There should be
    ///                         one set of indices per \p subregion batch. These indexes define the origin where to
    ///                         extract subregions from \p input. Indices can be specified as a Vec4 or Vec2 (in which
    ///                         case the BD are assumed to be 0). While usually within the input frame, subregions can
    ///                         be (partially) out-of-bound.
    /// \param border_mode      Border mode used for out-of-bound conditions.
    ///                         Can be Border::{NOTHING|ZERO|VALUE|CLAMP|MIRROR|REFLECT}.
    /// \param border_value     Constant value to use for out-of-bound conditions.
    ///                         Only used if \p border_mode is Border::VALUE.
    template<nt::readable_varray_decay Input,
             nt::writable_varray_decay Subregion,
             nt::readable_varray_decay Origin>
    requires (nt::varray_decay_with_compatible_or_spectrum_types<Input, Subregion> and
              nt::vec_integer_size<nt::value_type_t<Origin>, 2, 4>)
    void extract_subregions(
        Input&& input,
        Subregion&& subregions,
        Origin&& origins,
        Border border_mode = Border::ZERO,
        nt::value_type_t<Subregion> border_value = {}
    ) {
        check(nd::are_arrays_valid(input, subregions), "Empty array detected");
        check(not ni::are_overlapped(input, subregions),
              "The input and subregion(s) arrays should not overlap");
        check(ni::is_contiguous_vector(origins) and origins.n_elements() == subregions.shape()[0],
              "The origin should be a contiguous vector of {} elements but got shape={} and strides={}",
              subregions.shape()[0], origins.shape(), origins.strides());

        const Device device = subregions.device();
        check(device == input.device() and device == origins.device(),
              "The input and output arrays must be on the same device, "
              "but got input:device={}, origins:device={}, subregions:device={}",
              input.device(), origins.device(), device);

        auto input_shape = input.shape();
        auto input_strides = input.strides();
        auto subregion_shape = subregions.shape();
        auto subregion_strides = subregions.strides();

        // Reorder the dimensions to the rightmost order.
        // We cannot move the batch dimension, and we can only move the depth if 4d indices are passed.
        using indice_t = nt::mutable_value_type_t<Origin>;
        indice_t order;
        Vec<isize, 4> order_4d;
        if constexpr (indice_t::SIZE == 4) {
            const auto order_3d = ni::order(subregion_strides.pop_front(), subregion_shape.pop_front()) + 1;
            order_4d = order_3d.push_front(0);
            order = indice_t::from_values(0, order_3d[0], order_3d[1], order_3d[2]);
        } else {
            const auto order_2d = ni::order(subregion_strides.filter(2, 3), subregion_shape.filter(2, 3));
            order_4d = (order_2d + 2).push_front(Vec<isize, 2>{0, 1});
            order = indice_t::from_vec(order_2d);
        }
        if (order_4d != Vec<isize, 4>{0, 1, 2, 3}) {
            input_strides = ni::reorder(input_strides, order_4d);
            input_shape = ni::reorder(input_shape, order_4d);
            subregion_strides = ni::reorder(subregion_strides, order_4d);
            subregion_shape = ni::reorder(subregion_shape, order_4d);
        }

        using input_accessor_t = AccessorRestrict<nt::const_value_type_t<Input>, 4, isize>;
        using subregion_accessor_t = AccessorRestrict<nt::value_type_t<Subregion>, 4, isize>;
        const auto input_accessor = input_accessor_t(input.get(), input_strides);
        const auto subregion_accessor = subregion_accessor_t(subregions.get(), subregion_strides);

        switch (border_mode) {
            #define NOA_GENERATE_SUBREGION_(border)                                                             \
            case border: {                                                                                      \
                auto op = nd::ExtractSubregion<border, isize, indice_t, input_accessor_t, subregion_accessor_t>(\
                        input_accessor, subregion_accessor, input_shape, origins.get(), border_value, order);   \
                return iwise(subregion_shape, device, std::move(op),                                            \
                             std::forward<Input>(input),                                                        \
                             std::forward<Subregion>(subregions),                                               \
                             std::forward<Origin>(origins));                                                    \
            }
            NOA_GENERATE_SUBREGION_(Border::NOTHING)
            NOA_GENERATE_SUBREGION_(Border::ZERO)
            NOA_GENERATE_SUBREGION_(Border::VALUE)
            NOA_GENERATE_SUBREGION_(Border::CLAMP)
            NOA_GENERATE_SUBREGION_(Border::MIRROR)
            NOA_GENERATE_SUBREGION_(Border::PERIODIC)
            NOA_GENERATE_SUBREGION_(Border::REFLECT)

            default:
                panic("{} not supported", border_mode);
        }
    }

    /// Inserts into the output array one or multiple {1|2|3}d subregions at various locations.
    /// \param[in] subregions   Subregion(s) to insert into \p output.
    /// \param[out] output      Output array.
    /// \param[in] origins      Contiguous vector with the BDHW indexes of the subregions to insert into \p output.
    ///                         There should be one Vec4 per \p subregion batch. While usually within the output frame,
    ///                         subregions can be (partially) out-of-bound. This function assumes no overlap between
    ///                         subregions and an overlap may trigger a data race.
    template<nt::readable_varray_decay Subregion,
             nt::writable_varray_decay Output,
             nt::readable_varray_decay Origin>
    requires (nt::varray_decay_with_compatible_or_spectrum_types<Subregion, Output> and
              nt::vec_integer_size<nt::value_type_t<Origin>, 2, 4>)
    void insert_subregions(
        Subregion&& subregions,
        Output&& output,
        Origin&& origins
    ) {
        check(nd::are_arrays_valid(output, subregions), "Empty array detected");
        check(not ni::are_overlapped(output, subregions),
              "The subregion(s) and output arrays should not overlap");
        check(ni::is_contiguous_vector(origins) and origins.n_elements() == subregions.shape()[0],
              "The origins should be a contiguous vector of {} elements but got shape={} and strides={}",
              subregions.shape()[0], origins.shape(), origins.strides());

        const Device device = output.device();
        check(device == subregions.device() and device == origins.device(),
              "The input and output arrays must be on the same device, "
              "but got output:device={}, origins:device={}, subregions:device={}",
              device, origins.device(), subregions.device());

        auto output_shape = output.shape();
        auto output_strides = output.strides();
        auto subregion_shape = subregions.shape();
        auto subregion_strides = subregions.strides();

        // Reorder the dimensions to the rightmost order.
        // We cannot move the batch dimension, and we can only move the depth if 4d indices are passed.
        using indice_t = nt::mutable_value_type_t<Origin>;
        indice_t order;
        Vec<isize, 4> order_4d;
        if constexpr (indice_t::SIZE == 4) {
            const auto order_3d = ni::order(subregion_strides.pop_front(), subregion_shape.pop_front()) + 1;
            order_4d = order_3d.push_front(0);
            order = indice_t::from_values(0, order_3d[0], order_3d[1], order_3d[2]);
        } else {
            const auto order_2d = ni::order(subregion_strides.filter(2, 3), subregion_shape.filter(2, 3));
            order_4d = (order_2d + 2).push_front(Vec<isize, 2>{0, 1});
            order = indice_t::from_vec(order_2d);
        }
        if (order_4d != Vec<isize, 4>{0, 1, 2, 3}) {
            output_strides = ni::reorder(output_strides, order_4d);
            output_shape = ni::reorder(output_shape, order_4d);
            subregion_strides = ni::reorder(subregion_strides, order_4d);
            subregion_shape = ni::reorder(subregion_shape, order_4d);
        }

        using subregion_accessor_t = AccessorRestrict<nt::const_value_type_t<Subregion>, 4, isize>;
        using output_accessor_t = AccessorRestrict<nt::value_type_t<Output>, 4, isize>;
        const auto subregion_accessor = subregion_accessor_t(subregions.get(), subregion_strides);
        const auto output_accessor = output_accessor_t(output.get(), output_strides);

        auto op = nd::InsertSubregion<isize, indice_t, subregion_accessor_t, output_accessor_t>(
            subregion_accessor, output_accessor, output_shape, origins.get(), order);
        iwise(subregion_shape, device, std::move(op),
              std::forward<Subregion>(subregions),
              std::forward<Output>(output),
              std::forward<Origin>(origins));
    }

    /// Given a set of subregions with the same 2d/3d shape, compute an atlas layout (atlas shape + subregion origins).
    /// \details The atlas places the 2d/3d subregions on the same height-width plane. The depth of the atlas is the
    ///          same as the subregions'. Note that the shape of the atlas is not necessary a square. For instance,
    ///          with four subregions, the atlas layout is 2x2, but with five subregions it goes to `3x2` with one
    ///          empty region.
    /// \param subregion_shape      BDHW shape of the subregion(s).
    ///                             The batch dimension is the number of subregion(s) to place into the atlas.
    /// \param[out] output_origins  Subregion origin(s), relative to the atlas shape. One per batch.
    ///                             This function is effectively un-batching the 2d/3d subregions into a 2d/3d atlas.
    ///                             As such, the batch and depth dimensions are always set to 0, hence the possibility
    ///                             to return only the height and width dimension.
    /// \return                     Atlas shape.
    template<typename Origin>
    requires nt::vec_integer_size<nt::value_type_t<Origin>, 2, 4>
    auto atlas_layout(const Shape4& subregion_shape, const Origin& output_origins) -> Shape4 {
        NOA_ASSERT(not subregion_shape.is_empty());

        check(not output_origins.is_empty(), "Empty array detected");
        check(output_origins.device().is_cpu() and
              ni::is_contiguous_vector(output_origins) and
              output_origins.n_elements() == subregion_shape[0],
              "The output subregion origins should be a CPU contiguous vector with {} elements, "
              "but got device={}, shape={} and strides={}",
              subregion_shape[0], output_origins.device(), output_origins.shape(), output_origins.strides());

        const auto columns = static_cast<isize>(ceil(sqrt(static_cast<f32>(subregion_shape[0]))));
        const isize rows = (subregion_shape[0] + columns - 1) / columns;
        const auto atlas_shape = Shape4{
            1,
            subregion_shape[1],
            rows * subregion_shape[2],
            columns * subregion_shape[3]
        };

        output_origins.eval();
        const auto origins_1d = output_origins.span_1d_contiguous();
        for (isize y{}; y < rows; ++y) {
            for (isize x{}; x < columns; ++x) {
                const isize idx = y * columns + x;
                if (idx >= subregion_shape[0])
                    break;
                if constexpr (nt::value_type_t<Origin>::SIZE == 4)
                    origins_1d[idx] = {0, 0, y * subregion_shape[2], x * subregion_shape[3]};
                else
                    origins_1d[idx] = {y * subregion_shape[2], x * subregion_shape[3]};
            }
        }
        return atlas_shape;
    }

    /// Given a set of subregions, compute an atlas layout (atlas shape + subregion origins).
    /// \example
    /// \code
    /// // Constructs an atlas of maps.
    /// auto maps = ...;
    /// auto [atlas_shape, atlas_origins] = noa::atlas_layout<i32, 2>(maps.shape());
    /// auto atlas = noa::zeros<f32>(atlas_shape);
    /// noa::insert_subregions(maps, atlas, atlas_origins);
    /// \endcode
    template<nt::integer Int = isize, usize N = 4> requires (N == 2 or N == 4)
    auto atlas_layout(const Shape4& subregion_shape) -> Pair<Shape4, Array<Vec<Int, N>>> {
        auto output_subregion_origins = Array<Vec<Int, N>>(subregion_shape.batch());
        return {atlas_layout(subregion_shape, output_subregion_origins), output_subregion_origins};
    }
}
