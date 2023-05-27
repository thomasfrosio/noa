#pragma once

#include "noa/core/Types.hpp"

namespace noa::algorithm::memory {
    // Extract subregions from one or multiple arrays.
    // Subregions are defined by their 3D shape and their 4D origin.
    // The origin is the index in the input array that is the first element of the subregion.
    // If the subregion falls (partially) out of the input bounds, a border mode is used to handle that case.
    // Note: the origins dimensions might not correspond to the input/subregion dimensions because of the
    //       rearranging before the index-wise transformation. Thus, we keep the "order" and rearrange the
    //       origin on-the-fly (instead of allocating a new "origins" vector).
    template<BorderMode MODE, typename Index,  typename Offset, typename Value, typename SubregionIndex>
    class ExtractSubregion {
    public:
        using value_type = Value;
        using index_type = Index;
        using offset_type = Offset;
        using subregion_index_type = SubregionIndex;

        using index4_type = Vec4<Index>;
        using shape4_type = Shape4<Index>;
        using value_or_empty_type = std::conditional_t<MODE == BorderMode::VALUE, Value, Empty>;

        using input_accessor_type = AccessorRestrict<const value_type, 4, offset_type>;
        using subregion_accessor_type = AccessorRestrict<value_type, 4, offset_type>;
        using subregion_origins_type = const Vec4<subregion_index_type>*;

    public:
        ExtractSubregion(const input_accessor_type& input_accessor,
                         const subregion_accessor_type& subregion_accessor,
                         const shape4_type& input_shape,
                         const subregion_origins_type& origins,
                         value_type cvalue,
                         const Vec4<i64>& order)
                : m_input(input_accessor),
                  m_subregions(subregion_accessor),
                  m_subregion_origins(origins),
                  m_input_shape(input_shape),
                  m_order(order.as_safe<subregion_index_type>()) {
            if constexpr (!std::is_empty_v<value_or_empty_type>)
                m_cvalue = cvalue;
            else
                (void) cvalue;
        }

        NOA_HD constexpr void operator()(index_type oi, index_type oj, index_type ok, index_type ol) const {
            // TODO For CUDA, the origins could copied to constant memory.
            const auto corner_left = static_cast<index4_type>(m_subregion_origins[oi].reorder(m_order));

            const index_type ii = corner_left[0];
            const index_type ij = oj + corner_left[1];
            const index_type ik = ok + corner_left[2];
            const index_type il = ol + corner_left[3];

            if constexpr (MODE == BorderMode::NOTHING) {
                if (ii < 0 || ii >= m_input_shape[0] ||
                    ij < 0 || ij >= m_input_shape[1] ||
                    ik < 0 || ik >= m_input_shape[2] ||
                    il < 0 || il >= m_input_shape[3])
                    return;
                m_subregions(oi, oj, ok, ol) = m_input(ii, ij, ik, il);

            } else if constexpr (MODE == BorderMode::ZERO) {
                const bool is_valid = ii >= 0 && ii < m_input_shape[0] &&
                                      ij >= 0 && ij < m_input_shape[1] &&
                                      ik >= 0 && ik < m_input_shape[2] &&
                                      il >= 0 && il < m_input_shape[3];
                m_subregions(oi, oj, ok, ol) = is_valid ? m_input(ii, ij, ik, il) : value_type{0};

            } else if constexpr (MODE == BorderMode::VALUE) {
                const bool is_valid = ii >= 0 && ii < m_input_shape[0] &&
                                      ij >= 0 && ij < m_input_shape[1] &&
                                      ik >= 0 && ik < m_input_shape[2] &&
                                      il >= 0 && il < m_input_shape[3];
                m_subregions(oi, oj, ok, ol) = is_valid ? m_input(ii, ij, ik, il) : m_cvalue;

            } else {
                const index_type ii_bounded = noa::indexing::at<MODE>(ii, m_input_shape[0]);
                const index_type ij_bounded = noa::indexing::at<MODE>(ij, m_input_shape[1]);
                const index_type ik_bounded = noa::indexing::at<MODE>(ik, m_input_shape[2]);
                const index_type il_bounded = noa::indexing::at<MODE>(il, m_input_shape[3]);
                m_subregions(oi, oj, ok, ol) = m_input(ii_bounded, ij_bounded, ik_bounded, il_bounded);
            }
        }

    private:
        input_accessor_type m_input;
        subregion_accessor_type m_subregions;
        subregion_origins_type m_subregion_origins;
        shape4_type m_input_shape;
        Vec4<subregion_index_type> m_order;
        NOA_NO_UNIQUE_ADDRESS value_or_empty_type m_cvalue;
    };

    template<BorderMode MODE, typename Index, typename Offset, typename Value, typename SubregionIndex>
    auto extract_subregion(const Value* input, const Strides4<i64>& input_strides, const Shape4<i64>& input_shape,
                           Value* subregions, const Strides4<i64>& subregion_strides,
                           const Vec4<SubregionIndex>* origins, const Vec4<i64>& order, Value cvalue = Value{}) {
        return ExtractSubregion<MODE, Index, Offset, Value, SubregionIndex>(
                AccessorRestrict<const Value, 4, Offset>(input, input_strides.as_safe<Offset>()),
                AccessorRestrict<Value, 4, Offset>(subregions, subregion_strides.as_safe<Offset>()),
                input_shape.as_safe<Index>(), origins, cvalue, order);
    }

    // Insert subregions into one or multiple arrays.
    // This works as expected and is similar to ExtractSubregion.
    // Subregions can be (partially) out of the output bounds.
    // The only catch here is that overlapped subregions are not allowed (data-race).
    // It is not clear what we want in these cases (add?), so for now, just rule it out.
    template<typename Index, typename Offset, typename Value, typename SubregionIndex>
    class InsertSubregion {
    public:
        using value_type = Value;
        using index_type = Index;
        using offset_type = Offset;
        using subregion_index_type = SubregionIndex;
        using index4_type = Vec4<Index>;
        using shape4_type = Shape4<Index>;

        using output_accessor_type = AccessorRestrict<value_type, 4, offset_type>;
        using subregion_accessor_type = AccessorRestrict<const value_type, 4, offset_type>;
        using subregion_origins_type = const Vec4<subregion_index_type>*;

    public:
        InsertSubregion(const subregion_accessor_type& subregion_accessor,
                        const output_accessor_type& output_accessor,
                        const shape4_type& output_shape,
                        const subregion_origins_type& origins,
                        const Vec4<i64>& order)
                : m_output(output_accessor),
                  m_subregions(subregion_accessor),
                  m_subregion_origins(origins),
                  m_output_shape(output_shape),
                  m_order(order.as_safe<subregion_index_type>()) {}

        NOA_HD constexpr void operator()(index_type ii, index_type ij, index_type ik, index_type il) const {
            // TODO For CUDA, the origins could copied to constant memory.
            const auto corner_left = static_cast<index4_type>(m_subregion_origins[ii].reorder(m_order));

            const index_type oi = corner_left[0];
            const index_type oj = ij + corner_left[1];
            const index_type ok = ik + corner_left[2];
            const index_type ol = il + corner_left[3];
            if (oi < 0 || oi >= m_output_shape[0] ||
                oj < 0 || oj >= m_output_shape[1] ||
                ok < 0 || ok >= m_output_shape[2] ||
                ol < 0 || ol >= m_output_shape[3])
                return;

            // We assume no overlap in the output between subregions.
            m_output(oi, oj, ok, ol) = m_subregions(ii, ij, ik, il);
        }

    private:
        output_accessor_type m_output;
        subregion_accessor_type m_subregions;
        subregion_origins_type m_subregion_origins;
        shape4_type m_output_shape;
        Vec4<subregion_index_type> m_order;
    };

    template<typename Index, typename Offset, typename Value, typename SubregionIndex>
    auto insert_subregion(const Value* subregions, const Strides4<i64>& subregion_strides,
                          Value* output, const Strides4<i64>& output_strides, const Shape4<i64>& output_shape,
                          const Vec4<SubregionIndex>* origins, Vec4<i64> order) {
        return InsertSubregion<Index, Offset, Value, SubregionIndex>(
                AccessorRestrict<const Value, 4, Offset>(subregions, subregion_strides.as_safe<Offset>()),
                AccessorRestrict<Value, 4, Offset>(output, output_strides.as_safe<Offset>()),
                output_shape.as_safe<Index>(), origins, order);
    }
}
