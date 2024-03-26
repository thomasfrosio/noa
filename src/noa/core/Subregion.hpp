#pragma once

#include "noa/core/Types.hpp"

namespace noa {
    /// Extract subregions from one or multiple arrays.
    /// \details Subregions are defined by their 3d shape and their 2d (hw) or 4d (batch + dhw) origins.
    ///          If the subregion falls (even partially) out of the input bounds, the border mode is used
    ///          to handle that case.
    /// \note The origins dimensions might not correspond to the input/subregion dimensions because of the
    ///       rearranging before the index-wise transformation. Thus, this operator keeps the dimension "order"
    ///       and rearranges the origin on-the-fly (instead of allocating a new "origins" vector).
    template<Border MODE, typename Index, typename Origins, typename InputAccessor, typename SubregionAccessor>
    requires (nt::are_accessor_nd<4, InputAccessor, SubregionAccessor>::value and
              (nt::is_vec_int_size_v<Origins, 4> or
               nt::is_vec_int_size_v<Origins, 2>))
    class ExtractSubregion {
    public:
        using input_accessor_type = InputAccessor;
        using subregion_accessor_type = SubregionAccessor;
        using subregion_value_type = subregion_accessor_type::value_type;
        using index_type = Index;

        using origins_type = std::remove_const_t<Origins>; // Vec2<int> or Vec4<int>
        using origins_pointer_type = const origins_type*;

        using index4_type = Vec4<Index>;
        using shape4_type = Shape4<Index>;
        using subregion_value_or_empty_type = std::conditional_t<MODE == Border::VALUE, subregion_value_type, Empty>;

    public:
        ExtractSubregion(
                const input_accessor_type& input_accessor,
                const subregion_accessor_type& subregion_accessor,
                const shape4_type& input_shape,
                origins_pointer_type origins,
                subregion_value_type cvalue,
                const origins_type& order
        ) : m_input(input_accessor),
            m_subregions(subregion_accessor),
            m_subregion_origins(origins),
            m_input_shape(input_shape),
            m_order(order)
        {
            if constexpr (not std::is_empty_v<subregion_value_or_empty_type>)
                m_cvalue = cvalue;
            else
                (void) cvalue;
        }

        NOA_HD constexpr void operator()(const index4_type& output_indices) const {
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
                static_assert(nt::always_false_v<origins_type>);
            }

            if constexpr (MODE == Border::NOTHING) {
                if (input_indices[0] < 0 or input_indices[0] >= m_input_shape[0] or
                    input_indices[1] < 0 or input_indices[1] >= m_input_shape[1] or
                    input_indices[2] < 0 or input_indices[2] >= m_input_shape[2] or
                    input_indices[3] < 0 or input_indices[3] >= m_input_shape[3])
                    return;
                m_subregions(output_indices) = static_cast<subregion_value_type>(m_input(input_indices));

            } else if constexpr (MODE == Border::ZERO) {
                const bool is_valid = input_indices[0] >= 0 and input_indices[0] < m_input_shape[0] and
                                      input_indices[1] >= 0 and input_indices[1] < m_input_shape[1] and
                                      input_indices[2] >= 0 and input_indices[2] < m_input_shape[2] and
                                      input_indices[3] >= 0 and input_indices[3] < m_input_shape[3];
                m_subregions(output_indices) =
                        is_valid ? static_cast<subregion_value_type>(m_input(input_indices)) : subregion_value_type{};

            } else if constexpr (MODE == Border::VALUE) {
                const bool is_valid = input_indices[0] >= 0 and input_indices[0] < m_input_shape[0] and
                                      input_indices[1] >= 0 and input_indices[1] < m_input_shape[1] and
                                      input_indices[2] >= 0 and input_indices[2] < m_input_shape[2] and
                                      input_indices[3] >= 0 and input_indices[3] < m_input_shape[3];
                m_subregions(output_indices) =
                        is_valid ? static_cast<subregion_value_type>(m_input(input_indices)) : m_cvalue;

            } else {
                const index4_type bounded_indices{
                        ni::index_at<MODE>(input_indices[0], m_input_shape[0]),
                        ni::index_at<MODE>(input_indices[1], m_input_shape[1]),
                        ni::index_at<MODE>(input_indices[2], m_input_shape[2]),
                        ni::index_at<MODE>(input_indices[3], m_input_shape[3]),
                };
                m_subregions(output_indices) = static_cast<subregion_value_type>(m_input(bounded_indices));
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
    template<typename Index, typename Origins, typename SubregionAccessor, typename OutputAccessor>
    requires (nt::are_accessor_nd<4, OutputAccessor, SubregionAccessor>::value and
              (nt::is_vec_int_size_v<Origins, 4> or
               nt::is_vec_int_size_v<Origins, 2>))
    class InsertSubregion {
    public:
        using output_accessor_type = OutputAccessor;
        using output_value_type = output_accessor_type::value_type;
        using subregion_accessor_type = SubregionAccessor;
        using index_type = Index;

        using origins_type = std::remove_const_t<Origins>; // Vec2<int> or Vec4<int>
        using origins_pointer_type = const origins_type*;

        using index4_type = Vec4<Index>;
        using shape4_type = Shape4<Index>;

    public:
        InsertSubregion(
                const subregion_accessor_type& subregion_accessor,
                const output_accessor_type& output_accessor,
                const shape4_type& output_shape,
                origins_pointer_type origins,
                const origins_type& order
        ) : m_output(output_accessor),
            m_subregions(subregion_accessor),
            m_subregion_origins(origins),
            m_output_shape(output_shape),
            m_order(order) {}

        NOA_HD constexpr void operator()(const index4_type& input_indices) const noexcept {
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
                static_assert(nt::always_false_v<origins_type>);
            }

            if (output_indices[0] < 0 || output_indices[0] >= m_output_shape[0] ||
                output_indices[1] < 0 || output_indices[1] >= m_output_shape[1] ||
                output_indices[2] < 0 || output_indices[2] >= m_output_shape[2] ||
                output_indices[3] < 0 || output_indices[3] >= m_output_shape[3])
                return;

            // We assume no overlap in the output between subregions.
            m_output(output_indices) = static_cast<output_value_type>(m_subregions(input_indices));
        }

    private:
        output_accessor_type m_output;
        subregion_accessor_type m_subregions;
        origins_pointer_type m_subregion_origins;
        shape4_type m_output_shape;
        origins_type m_order;
    };
}
