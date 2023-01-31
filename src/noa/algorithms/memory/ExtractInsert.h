#pragma once

#include "noa/common/Types.h"

namespace noa::memory::details {
    // Extract subregions from one or multiple arrays.
    // Subregions are defined by their 3D shape and their 4D origin.
    // The origin is the index in the input array that is the first element of the subregion.
    // If the subregion falls (partially) out of the input bounds, a border mode is used to handle that case.
    // Note: the origins dimensions might not correspond to the input/subregion dimensions because of the
    //       rearranging before the index-wise transformation. Thus, we keep the "order" and rearrange the
    //       origin on-the-fly (instead of allocating a new "origins" vector).
    template<BorderMode MODE, typename Index,  typename Offset, typename Value>
    class Extract {
    public:
        using value_type = Value;
        using index_type = Index;
        using offset_type = Offset;
        using index4_type = Int4<Index>;
        using value_or_empty_type = std::conditional_t<MODE == BORDER_VALUE, Value, empty_t>;

        using input_accessor_type = AccessorRestrict<const value_type, 4, offset_type>;
        using subregion_accessor_type = AccessorRestrict<value_type, 4, offset_type>;
        using subregion_origins_type = const int4_t*;

    public:
        Extract(input_accessor_type input_accessor,
                subregion_accessor_type subregion_accessor,
                const index4_type input_shape,
                subregion_origins_type origins,
                value_type cvalue,
                int4_t order)
                : m_input(input_accessor),
                  m_subregions(subregion_accessor),
                  m_subregion_origins(origins),
                  m_input_shape(input_shape),
                  m_order(order),
                  m_cvalue(cvalue) {}

        NOA_HD constexpr void operator()(index_type oi, index_type oj, index_type ok, index_type ol) const {
            // TODO For CUDA, the origins could copied to constant memory.
            const auto origin = noa::indexing::reorder(m_subregion_origins[oi], m_order);
            const auto corner_left = static_cast<index4_type>(origin);

            const index_type ii = corner_left[0];
            const index_type ij = oj + corner_left[1];
            const index_type ik = ok + corner_left[2];
            const index_type il = ol + corner_left[3];

            if constexpr (MODE == BORDER_NOTHING) {
                if (ii < 0 || ii >= m_input_shape[0] ||
                    ij < 0 || ij >= m_input_shape[1] ||
                    ik < 0 || ik >= m_input_shape[2] ||
                    il < 0 || il >= m_input_shape[3])
                    return;
                m_subregions(oi, oj, ok, ol) = m_input(ii, ij, ik, il);

            } else if constexpr (MODE == BORDER_ZERO) {
                const bool is_valid = ii >= 0 && ii < m_input_shape[0] &&
                                      ij >= 0 && ij < m_input_shape[1] &&
                                      ik >= 0 && ik < m_input_shape[2] &&
                                      il >= 0 && il < m_input_shape[3];
                m_subregions(oi, oj, ok, ol) = is_valid ? m_input(ii, ij, ik, il) : value_type{0};

            } else if constexpr (MODE == BORDER_VALUE) {
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
        index4_type m_input_shape;
        int4_t m_order;
        value_or_empty_type m_cvalue;
    };

    template<BorderMode MODE, typename Index, typename Offset, typename Value>
    auto extract(const Value* input, const dim4_t& input_strides, const dim4_t& input_shape,
                 Value* subregions, const dim4_t& subregion_strides,
                 const int4_t* origins, int4_t order, Value cvalue = Value{}) {

        using index4_t = Int4<Index>;
        using offset4_t = Int4<Offset>;
        const auto i_shape = safe_cast<index4_t>(input_shape);
        const auto input_accessor = AccessorRestrict<const Value, 4, Offset>(
                input, safe_cast<offset4_t>(input_strides));
        const auto subregions_accessor = AccessorRestrict<Value, 4, Offset>(
                subregions, safe_cast<offset4_t>(subregion_strides));

        return Extract<MODE, Index, Offset, Value>(
                input_accessor, subregions_accessor,
                i_shape, origins, cvalue, order);
    }

    // Insert subregions into one or multiple arrays.
    // This works as expected and is similar to Extract.
    // Subregions can be (partially) out of the output bounds.
    // The only catch here is that overlapped subregions are not allowed (data-race).
    // It is not clear what we want in these cases (add?), so for now, just rule it out.
    template<typename Index,  typename Offset, typename Value>
    class Insert {
    public:
        using value_type = Value;
        using index_type = Index;
        using offset_type = Offset;
        using index4_type = Int4<Index>;

        using output_accessor_type = AccessorRestrict<value_type, 4, offset_type>;
        using subregion_accessor_type = AccessorRestrict<const value_type, 4, offset_type>;
        using subregion_origins_type = const int4_t*;

    public:
        Insert(subregion_accessor_type subregion_accessor,
               output_accessor_type output_accessor,
               const index4_type output_shape,
               subregion_origins_type origins,
               int4_t order)
                : m_output(output_accessor),
                  m_subregions(subregion_accessor),
                  m_subregion_origins(origins),
                  m_output_shape(output_shape),
                  m_order(order) {}

        NOA_HD constexpr void operator()(index_type ii, index_type ij, index_type ik, index_type il) const {
            // TODO For CUDA, the origins could copied to constant memory.
            const auto origin = indexing::reorder(m_subregion_origins[ii], m_order);
            const auto corner_left = static_cast<index4_type>(origin);

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
        index4_type m_output_shape;
        int4_t m_order;
    };

    template<typename Index, typename Offset, typename Value>
    auto insert(const Value* subregions, const dim4_t& subregion_strides,
                Value* output, const dim4_t& output_strides, const dim4_t& output_shape,
                const int4_t* origins, int4_t order) {

        using index4_t = Int4<Index>;
        using offset4_t = Int4<Offset>;
        const auto o_shape = safe_cast<index4_t>(output_shape);
        const auto output_accessor = AccessorRestrict<Value, 4, Offset>(
                output, safe_cast<offset4_t>(output_strides));
        const auto subregions_accessor = AccessorRestrict<const Value, 4, Offset>(
                subregions, safe_cast<offset4_t>(subregion_strides));

        return Insert<Index, Offset, Value>(
                subregions_accessor, output_accessor,
                o_shape, origins, order);
    }
}
