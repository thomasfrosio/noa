#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Tuple.hpp"
#include "noa/core/utils/Misc.hpp"
#include "noa/core/types/Shape.hpp"
#include "noa/core/indexing/Offset.hpp"
#include "noa/core/indexing/Layout.hpp"

namespace noa {
    /// Index-wise operator wrapper element-wise operators, supporting variadic amount of inputs and outputs.
    ///
    /// \details The inputs and outputs should be wrapped into a noa::Tuple object. The actual types of these
    ///          ranges are not enforced, and while this was designed for noa::Accessor, the only type requirement
    ///          is the definition of the operator: \code operator()(Indexes indexes...) -> value_type& \endcode.
    ///          The indexes, usually integers, are forwarded from the evaluate() function directly to this operator.
    ///          The input values are all retrieve that way (using the same indexes), and forwarded to the
    ///          element-wise operator. The returned value of this operator should be a tuple, containing the output
    ///          value for each output range. These values are saved in their corresponding output range at the same
    ///          indexes. This is effectively: \code ewise(inputs(indexes...)....) -> outputs(indexes...)... \endcode
    ///
    /// \note The input and output tuples, and the operator, are all moved in the operator.
    ///       Keeping a copy of these is necessary for the GPU implementations (because everything needs to be copied
    ///       to the GPU memory during the kernel launch, so we need to make sure values are stored, not references).
    ///       For the CPU implementations, we could save the reference.
    ///
    /// \tparam InputAccessors  Tuple of input accessors.
    /// \tparam OutputAccessors Tuple of output accessors.
    /// \tparam Op              Element-wise operator.
    template<typename InputAccessors, typename OutputAccessors, typename Op>
    class EwiseVariadicOp {
    private:
        InputAccessors m_inputs;
        OutputAccessors m_outputs;
        Op m_op;

    public:
        using operator_type = Op;
        struct save_output_t {};

    public:
        /// Move everything into this operator.
        EwiseVariadicOp(
                InputAccessors&& inputs,
                OutputAccessors&& outputs,
                Op&& op
        ) : m_inputs{std::move(inputs)}, m_outputs{std::move(outputs)}, m_op{std::move(op)} {}

        /// Read each input value at the nd-indices, forward them to the operator.
        /// The output from the operator should be a tuple, with one value per output accessor.
        /// Each value is saved at the nd-indices by the output accessors.
        template<typename... Indexes>
        inline constexpr void operator()(const Indexes& ... indexes) {
            static_for_each<OutputAccessors::SIZE>(
                    *this, save_output_t{}, apply_operator_(indexes...), indexes...);
        }

        template<size_t I, typename OutputTuple, typename... Indexes>
        void operator()(save_output_t, OutputTuple&& output, Indexes... indexes) {
            static_assert(OutputAccessors::SIZE == OutputTuple::SIZE);
            auto& output_ref = m_outputs[Tag<I>{}](indexes...);
            using output_t = std::remove_reference_t<decltype(output_ref)>;
            output_ref = static_cast<output_t>(output[Tag<I>{}]);
        }

        template<class Index, nt::enable_if_bool_t<nt::is_detected_v<nt::has_initialize, operator_type>> = true>
        void block_initialize(const Index& index) {
            m_op.block_initialize(index);
        }

        template<class Index, nt::enable_if_bool_t<nt::is_detected_v<nt::has_closure, operator_type>> = true>
        void block_closure(const Index& index) {
            m_op.block_closure(index);
        }

    private:
        template<typename... B, typename... Indexes>
        inline constexpr decltype(auto) apply_operator_(noa::TypeList<B...>, const Indexes& ... indexes) {
            return m_op(m_inputs.::noa::traits::identity_t<B>::value(indexes...)...);
        }

        template<typename... B, typename... Indexes>
        inline constexpr decltype(auto) apply_operator_(Indexes&& ... indexes) {
            using base_list = typename std::remove_reference_t<InputAccessors>::base_list;
            return apply_operator_(base_list{}, std::forward<Indexes>(indexes)...);
        }
    };
}
