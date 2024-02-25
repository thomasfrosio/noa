#pragma once

#include "noa/core/Subregion.hpp"
#include "noa/unified/Indexing.hpp"
#include "noa/unified/Iwise.hpp"
#include "noa/unified/Array.hpp"

namespace noa {
    /// Extracts one or multiple {1|2|3}d subregions at various locations in the input array.
    /// \param[in] input        Input array to extract from.
    /// \param[out] subregions  Output subregion(s).
    ///                         Input values are casted to the subregions value type.
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
    template<typename Input, typename Subregion, typename Origin,
             typename Value = nt::value_type_t<Subregion>>
    requires (nt::are_varray_v<Input, Subregion, Origin> and
              nt::is_almost_same_v<nt::value_type_t<Subregion>, Value> and
              (nt::is_vec_int_size_v<nt::value_type_t<Origin>, 2> or
               nt::is_vec_int_size_v<nt::value_type_t<Origin>, 4>))
    void extract_subregions(
            const Input& input,
            const Subregion& subregions,
            const Origin& origins,
            Border border_mode = Border::ZERO,
            Value border_value = Value{0}
    ) {
        check(not input.is_empty() and not subregions.is_empty(), "Empty array detected");
        check(not ni::are_overlapped(input, subregions),
              "The input and subregion(s) arrays should not overlap");
        check(ni::is_contiguous_vector(origins) and origins.elements() == subregions.shape()[0],
              "The origin should be a contiguous vector of {} elements but got shape {} and strides {}",
              subregions.shape()[0], origins.shape(), origins.strides());

        const Device device = subregions.device();
        check(device == input.device() and device == origins.device(),
              "The input and output arrays must be on the same device, but got input={}, origins={}, subregions={}",
              input.device(), origins.device(), device);

        auto input_shape = input.shape();
        auto input_strides = input.strides();
        auto subregion_shape = subregions.shape();
        auto subregion_strides = subregions.strides();

        // Reorder the dimensions to the rightmost order.
        // We cannot move the batch dimension, and we can only move the depth if 4d indices are passed.
        using indice_t = nt::mutable_value_type_t<Origin>;
        indice_t order;
        Vec4<i64> order_4d;
        if constexpr (indice_t::SIZE == 4) {
            const auto order_3d = ni::order(subregion_strides.pop_front(), subregion_shape.pop_front()) + 1;
            order_4d = order_3d.push_front(0);
            order = indice_t::from_values(0, order_3d[0], order_3d[1], order_3d[2]);
        } else {
            const auto order_2d = ni::order(subregion_strides.filter(2, 3), subregion_shape.filter(2, 3)) + 2;
            order_4d = order_2d.push_front({0, 1});
            order = indice_t::from_vec(order_2d);
        }
        if (any(order_4d != Vec4<i64>{0, 1, 2, 3})) {
            input_strides = ni::reorder(input_strides, order);
            input_shape = ni::reorder(input_shape, order);
            subregion_strides = ni::reorder(subregion_strides, order);
            subregion_shape = ni::reorder(subregion_shape, order);
        }

        using input_accessor_t = AccessorRestrictI64<const nt::mutable_value_type_t<Input>, 4>;
        using subregion_accessor_t = AccessorRestrictI64<nt::value_type_t<Subregion>, 4>;
        const auto input_accessor = input_accessor_t(input.get(), input_strides);
        const auto subregion_accessor = subregion_accessor_t(subregions.get(), subregion_strides);

        switch (border_mode) {
            case Border::NOTHING: {
                auto op = ExtractSubregion<Border::NOTHING, i64, indice_t, input_accessor_t, subregion_accessor_t>(
                        input_accessor, subregion_accessor, input_shape, origins.get(), border_value, order);
                return iwise(subregion_shape, device, std::move(op), input, subregions, origins);
            }
            case Border::ZERO: {
                auto op = ExtractSubregion<Border::ZERO, i64, indice_t, input_accessor_t, subregion_accessor_t>(
                        input_accessor, subregion_accessor, input_shape, origins.get(), border_value, order);
                return iwise(subregion_shape, device, std::move(op), input, subregions, origins);
            }
            case Border::VALUE: {
                auto op = ExtractSubregion<Border::VALUE, i64, indice_t, input_accessor_t, subregion_accessor_t>(
                        input_accessor, subregion_accessor, input_shape, origins.get(), border_value, order);
                return iwise(subregion_shape, device, std::move(op), input, subregions, origins);
            }
            case Border::CLAMP: {
                auto op = ExtractSubregion<Border::CLAMP, i64, indice_t, input_accessor_t, subregion_accessor_t>(
                        input_accessor, subregion_accessor, input_shape, origins.get(), border_value, order);
                return iwise(subregion_shape, device, std::move(op), input, subregions, origins);
            }
            case Border::MIRROR: {
                auto op = ExtractSubregion<Border::MIRROR, i64, indice_t, input_accessor_t, subregion_accessor_t>(
                        input_accessor, subregion_accessor, input_shape, origins.get(), border_value, order);
                return iwise(subregion_shape, device, std::move(op), input, subregions, origins);
            }
            case Border::PERIODIC: {
                auto op = ExtractSubregion<Border::PERIODIC, i64, indice_t, input_accessor_t, subregion_accessor_t>(
                        input_accessor, subregion_accessor, input_shape, origins.get(), border_value, order);
                return iwise(subregion_shape, device, std::move(op), input, subregions, origins);
            }
            case Border::REFLECT: {
                auto op = ExtractSubregion<Border::REFLECT, i64, indice_t, input_accessor_t, subregion_accessor_t>(
                        input_accessor, subregion_accessor, input_shape, origins.get(), border_value, order);
                return iwise(subregion_shape, device, std::move(op), input, subregions, origins);
            }
            default:
                panic("{} not supported", border_mode);
        }
    }

    /// Inserts into the output array one or multiple {1|2|3}d subregions at various locations.
    /// \tparam T               Any data type.
    /// \param[in] subregions   Subregion(s) to insert into \p output.
    /// \param[out] output      Output array.
    /// \param[in] origins      Contiguous vector with the BDHW indexes of the subregions to insert into \p output.
    ///                         There should be one Vec4 per \p subregion batch. While usually within the output frame,
    ///                         subregions can be (partially) out-of-bound. This function assumes no overlap between
    ///                         subregions and an overlap may trigger a data race.
    template<typename Subregion, typename Output, typename Origin>
    requires (nt::are_varray_v<Output, Subregion, Origin> and
              (nt::is_vec_int_size_v<nt::value_type_t<Origin>, 2> or
               nt::is_vec_int_size_v<nt::value_type_t<Origin>, 4>))
    void insert_subregions(
            const Subregion& subregions,
            const Output& output,
            const Origin& origins
    ) {
        check(not output.is_empty() and not subregions.is_empty(), "Empty array detected");
        check(not ni::are_overlapped(output, subregions),
              "The subregion(s) and output arrays should not overlap");
        check(ni::is_contiguous_vector(origins) and origins.elements() == subregions.shape()[0],
              "The origins should be a contiguous vector of {} elements but got shape {} and strides {}",
              subregions.shape()[0], origins.shape(), origins.strides());

        const Device device = output.device();
        check(device == subregions.device() and device == origins.device(),
              "The input and output arrays must be on the same device, but got output={}, origins={}, subregions={}",
              device, origins.device(), subregions.device());

        auto output_shape = output.shape();
        auto output_strides = output.strides();
        auto subregion_shape = subregions.shape();
        auto subregion_strides = subregions.strides();

        // Reorder the dimensions to the rightmost order.
        // We cannot move the batch dimension, and we can only move the depth if 4d indices are passed.
        using indice_t = nt::mutable_value_type_t<Origin>;
        indice_t order;
        Vec4<i64> order_4d;
        if constexpr (indice_t::SIZE == 4) {
            const auto order_3d = ni::order(subregion_strides.pop_front(), subregion_shape.pop_front()) + 1;
            order_4d = order_3d.push_front(0);
            order = indice_t::from_values(0, order_3d[0], order_3d[1], order_3d[2]);
        } else {
            const auto order_2d = ni::order(subregion_strides.filter(2, 3), subregion_shape.filter(2, 3)) + 2;
            order_4d = order_2d.push_front({0, 1});
            order = indice_t::from_vec(order_2d);
        }
        if (any(order_4d != Vec4<i64>{0, 1, 2, 3})) {
            output_strides = ni::reorder(output_strides, order);
            output_shape = ni::reorder(output_shape, order);
            subregion_strides = ni::reorder(subregion_strides, order);
            subregion_shape = ni::reorder(subregion_shape, order);
        }

        using subregion_accessor_t = AccessorRestrictI64<const nt::mutable_value_type_t<Subregion>, 4>;
        using output_accessor_t = AccessorRestrictI64<nt::value_type_t<Output>, 4>;
        const auto subregion_accessor = subregion_accessor_t(subregions.get(), subregion_strides);
        const auto output_accessor = output_accessor_t(output.get(), output_strides);

        auto op = InsertSubregion<i64, indice_t, subregion_accessor_t, output_accessor_t>(
                subregion_accessor, output_accessor, output_shape, origins.get(), order);
        iwise(subregion_shape, device, std::move(op), subregions, output, origins);
    }

    /// Given a set of subregions with the same 2d/3d shape, compute an atlas layout (atlas shape + subregion origins).
    /// \details The atlas places the 2d/3d subregions in the same height-width plane. The depth of the atlas is the
    ///          same as the subregions'. Note that the shape of the atlas is not necessary a square. For instance,
    ///          with 4 subregions the atlas layout is 2x2, but with 5 subregions it goes to `3x2` with one empty
    ///          region.
    /// \param subregion_shape      BDHW shape of the subregion(s).
    ///                             The batch dimension is the number of subregion(s) to place into the atlas.
    /// \param[out] output_origins  Subregion origin(s), relative to the atlas shape. One per batch.
    ///                             This function is effectively un-batching the 2d/3d subregions into a 2d/3d atlas.
    ///                             As such, the batch and depth dimensions are always set to 0, hence the possibility
    ///                             to return only the height and width dimension.
    /// \return                     Atlas shape.
    template<typename Origin>
    requires (nt::is_varray_v<Origin> and
              (nt::is_vec_int_size_v<nt::value_type_t<Origin>, 2> or
               nt::is_vec_int_size_v<nt::value_type_t<Origin>, 4>))
    Shape4<i64> atlas_layout(const Shape4<i64>& subregion_shape, const Origin& output_origins) {
        NOA_ASSERT(noa::all(subregion_shape > 0));
        check(not output_origins.is_empty(), "Empty array detected");
        check(output_origins.device().is_cpu() and
              ni::is_contiguous_vector(output_origins) and
              output_origins.elements() == subregion_shape[0],
              "The output subregion origins should be a CPU contiguous vector,"
              "but got device={}, shape={} and strides={}",
              output_origins.device(), output_origins.shape(), output_origins.strides());

        const auto columns = static_cast<i64>(ceil(sqrt(static_cast<f32>(subregion_shape[0]))));
        const i64 rows = (subregion_shape[0] + columns - 1) / columns;
        const auto atlas_shape = Shape4<i64>{
                1,
                subregion_shape[1],
                rows * subregion_shape[2],
                columns * subregion_shape[3]};

        output_origins.eval();
        const auto origins_1d = output_origins.accessor_contiguous_1d();
        for (i64 y = 0; y < rows; ++y) {
            for (i64 x = 0; x < columns; ++x) {
                const i64 idx = y * columns + x;
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
    /// \example Construct an atlas of maps.
    /// \code
    /// auto maps = ...;
    /// auto [atlas_shape, atlas_origins] = noa::atlas_layout<i32, 2>(maps.shape());
    /// auto atlas = noa::zeros<f32>(atlas_shape);
    /// noa::insert_subregions(maps, atlas, atlas_origins);
    /// \endcode
    template<typename Int = i64, size_t N = 4>
    requires (std::is_integral_v<Int> and (N == 2 or N == 4))
    auto atlas_layout(const Shape4<i64>& subregion_shape) -> Pair<Shape4<i64>, Array<Vec<Int, N>>> {
        Array<Vec<Int, N>> output_subregion_origins(subregion_shape.batch());
        return {atlas_layout(subregion_shape, output_subregion_origins), output_subregion_origins};
    }
}
