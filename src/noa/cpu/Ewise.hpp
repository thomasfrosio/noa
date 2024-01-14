#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Tuple.hpp"
#include "noa/core/utils/Misc.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/indexing/Offset.hpp"
#include "noa/core/indexing/Layout.hpp"
#include "noa/cpu/Iwise.hpp"

namespace noa::cpu {
    /// Index-wise operator that wraps an element-wise operator.
    ///
    /// \details The inputs and outputs should be wrapped into a Tuple. The actual types inside the tuple are not
    ///          enforced, and while this was designed for Accessor and AccessorValue, the only type requirement
    ///          is the definition of the operator: \code operator()(Indexes indices...) -> value_type& \endcode.
    ///
    /// \tparam TupleInputAccessors  Tuple of input accessors.
    /// \tparam TupleOutputAccessors Tuple of output accessors.
    /// \tparam Op Element-wise operator, such as:
    ///            \code operator()(InputValueTypes&&..., OutputValueTypes&...) \endcode
    ///            For instance, a copy operator can be something like:
    ///            \code operator()(const f64& src, f64& dst) { dst = src; } \endcode
    template<guts::EwiseAdaptorTags tags, typename TupleInputAccessors, typename TupleOutputAccessors, typename Op>
    requires nt::are_tuple<TupleInputAccessors, TupleOutputAccessors>::value
    class EwiseToIwise {
    private:
        TupleInputAccessors m_inputs;
        TupleOutputAccessors m_outputs;

        // We need to store the operator by value here.
        // 1. keeping a reference will be too error-prone in the asynchronous or multithreaded case.
        // 2. the GPU implementation stores it by value too.
        // If the operator cannot be copied, the user can still use the index-wise operator...
        Op m_op;

    public:
        /// Move everything into this operator.
        constexpr EwiseToIwise(
                TupleInputAccessors&& inputs,
                TupleOutputAccessors&& outputs,
                const Op& op
        ) : m_inputs{std::move(inputs)}, m_outputs{std::move(outputs)}, m_op{op} {}

        constexpr EwiseToIwise(
                TupleInputAccessors&& inputs,
                TupleOutputAccessors&& outputs,
                Op&& op
        ) : m_inputs{std::move(inputs)}, m_outputs{std::move(outputs)}, m_op{std::move(op)} {}

        /// This does: ewise_operator(inputs(indices...)..., outputs(indices..)...);
        /// The ewise_operator is free to take these parameters by (const-)reference or by value,
        /// but usually the outputs are taken by reference.
        constexpr void operator()(auto... indices) {
            // FIXME Replace with EwiseOpWrapper::call<ZIP_INPUTS, ZIP_OUTPUTS>(op, m_inputs, m_outputs, indices...)
            [this, indices...]<typename... I, typename... O>(nt::TypeList<I...>, nt::TypeList<O...>) {
                if constexpr (tags.zip_inputs && tags.zip_outputs) {
                    m_op(forward_as_tuple(std::forward<nt::type_type_t<I>>(m_inputs.::std::type_identity_t<I>::value(indices...))...),
                         forward_as_tuple(std::forward<nt::type_type_t<O>>(m_outputs.::std::type_identity_t<O>::value(indices...))...));
                } else if constexpr (tags.zip_inputs) {
                    m_op(forward_as_tuple(std::forward<nt::type_type_t<I>>(m_inputs.::std::type_identity_t<I>::value(indices...))...),
                         m_outputs.::std::type_identity_t<O>::value(indices...)...);
                } else if constexpr (tags.zip_outputs) {
                    m_op(m_inputs.::std::type_identity_t<I>::value(indices...)...,
                         forward_as_tuple(std::forward<nt::type_type_t<O>>(m_outputs.::std::type_identity_t<O>::value(indices...))...));
                } else {
                    m_op(m_inputs.::std::type_identity_t<I>::value(indices...)...,
                         m_outputs.::std::type_identity_t<O>::value(indices...)...);
                }
            }(typename TupleInputAccessors::base_list{},
              typename TupleOutputAccessors::base_list{});
        }

        template<class Index> requires requires (Index i) { m_op.init(i); }
        constexpr void init(const Index& index) {
            m_op.init(index);
        }

        template<class Index> requires requires (Index i) { m_op.final(i); }
        constexpr void final(const Index& index) {
            m_op.final(index);
        }
    };

    template<guts::EwiseAdaptorTags tags, typename Inputs, typename Outputs, typename Op>
    auto ewise_to_iwise(Inputs&& inputs, Outputs&& outputs, Op&& op) {
        using op_t = EwiseToIwise<tags, Inputs, Outputs, Op>;
        return op_t(std::forward<Inputs>(inputs), std::forward<Outputs>(outputs), std::forward<Op>(op));
    }

    /// Launches the element-wise core-function.
    /// \param input_accessors  Input 4d accessors or accessor-values.
    /// \param output_accessors Output 4d accessors or accessor-values.
    /// \param shape            4d shape.
    /// \param op               Element-wise operator.
    template<guts::EwiseAdaptorTags tags,
             size_t PARALLEL_ELEMENTS_THRESHOLD = Iwise::PARALLEL_ELEMENTS_THRESHOLD,
             typename Inputs, typename Outputs, typename Index, typename Op>
    requires nt::are_tuple_of_accessor_v<Inputs, Outputs>
    void ewise(
            Inputs&& input_accessors,
            Outputs&& output_accessors,
            const Shape4<Index>& shape,
            Op&& op,
            i64 n_threads = 1
    ) {
        // Check contiguity.
        const bool are_all_contiguous =
                ni::are_contiguous(input_accessors) &&
                ni::are_contiguous(output_accessors);

        // FIXME We could try collapse contiguous dimensions to still have a contiguous loop.
        if (are_all_contiguous) {
            // In this case, check for aliasing, in the hope to force
            if (guts::are_accessors_aliased(input_accessors, output_accessors)) {
                iwise<PARALLEL_ELEMENTS_THRESHOLD>(
                        Shape1<Index>{shape.elements()},
                        ewise_to_iwise(guts::to_1d_accessors(std::forward<Inputs>>(input_accessors)),
                                       guts::to_1d_accessors(std::forward<Outputs>>(output_accessors)),
                                       std::forward<Op>(op)),
                        n_threads);
            } else {
                constexpr bool RESTRICT = true;
                iwise<PARALLEL_ELEMENTS_THRESHOLD>(
                        Shape1<Index>{shape.elements()},
                        ewise_to_iwise(guts::to_1d_accessors<RESTRICT>(std::forward<Inputs>>(input_accessors)),
                                       guts::to_1d_accessors<RESTRICT>(std::forward<Outputs>>(output_accessors)),
                                       std::forward<Op>(op)),
                        n_threads);
            }
        } else {
            iwise(shape,
                  ewise_to_iwise(std::forward<Inputs>>(input_accessors),
                                 std::forward<Outputs>>(output_accessors),
                                 std::forward<Op>(op)),
                  n_threads);
        }
    }
}
