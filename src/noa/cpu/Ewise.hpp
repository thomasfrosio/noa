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
    /// \details The inputs and outputs should be wrapped into a noa::Tuple object. The actual types inside the tuple
    ///          are not enforced, and while this was designed for Accessor and AccessorValue, the only type requirement
    ///          is the definition of the operator: \code operator()(Indexes indices...) -> value_type& \endcode.
    ///
    /// \tparam TupleInputAccessors  Tuple of input accessors.
    /// \tparam TupleOutputAccessors Tuple of output accessors.
    /// \tparam Op              Element-wise operator.
    template<typename TupleInputAccessors, typename TupleOutputAccessors, typename Op>
    requires nt::are_tuple<TupleInputAccessors, TupleOutputAccessors>::value
    class EwiseWrapper {
    private:
        TupleInputAccessors m_inputs;
        TupleOutputAccessors m_outputs;

        // We need to store the operator by value here.
        // 1. keeping a reference will be too error-prone in the asynchronous or multithreaded case.
        // 2. the GPU implementation stores it by value too
        Op m_op;

    public:
        /// Move everything into this operator.
        constexpr EwiseWrapper(
                TupleInputAccessors&& inputs,
                TupleOutputAccessors&& outputs,
                const Op& op
        ) : m_inputs{std::move(inputs)}, m_outputs{std::move(outputs)}, m_op{op} {}

        constexpr EwiseWrapper(
                TupleInputAccessors&& inputs,
                TupleOutputAccessors&& outputs,
                Op&& op
        ) : m_inputs{std::move(inputs)}, m_outputs{std::move(outputs)}, m_op{std::move(op)} {}

        /// This does: ewise_operator(inputs(indices...)..., outputs(indices..)...);
        /// The ewise_operator is free to take these parameters by (const-)reference or by value,
        /// but usually the outputs are taken by reference.
        constexpr void operator()(auto... indices) {
            [this, indices...]<typename... I, typename... O>(nt::TypeList<I...>, nt::TypeList<O...>) {
                m_op(m_inputs.::noa::traits::identity_t<I>::value(indices...)...,
                     m_outputs.::noa::traits::identity_t<O>::value(indices...)...);
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

    /// Launches the element-wise core-function.
    /// \param input_accessors  Input 4d accessors.
    /// \param output_accessors Output 4d accessors.
    /// \param shape            4d shape.
    /// \param op               Element-wise operator.
    template<size_t PARALLEL_ELEMENTS_THRESHOLD = Iwise::PARALLEL_ELEMENTS_THRESHOLD,
             typename... Inputs,
             typename... Outputs,
             typename Index,
             typename Op>
    void ewise(
            const Tuple<Inputs...>& input_accessors,
            const Tuple<Outputs...>& output_accessors,
            const Shape4<Index>& shape,
            Op&& op,
            i64 n_threads = 1
    ) {
        // The goal of this function is to create the Ewise operator and pass it to the IwiseParallel::execute.
        // First, reorder

        // Check contiguity.
        auto check_contiguous = [&shape](const auto& accessor) {
            return ni::are_contiguous(accessor.strides(), shape); // FIXME AccessorValue
        };
        const bool are_all_contiguous =
                input_accessors.all(check_contiguous) &&
                output_accessors.all(check_contiguous);

        if (are_all_contiguous) {
            // Convert everything to 1d contiguous.
            auto to_1d = []<typename T>(T&& accessor) -> decltype(auto) {
                using accessor_t = decltype(accessor);
                if constexpr (nt::is_accessor_v<accessor_t> && !nt::is_accessor_value_v<accessor_t>) {
                    using value_t = typename accessor_t::value_type;
                    using accessor_1d = AccessorContiguous<value_t, 1, Index>;
                    return accessor_1d(accessor.get());
                } else {
                    return std::forward<T>(accessor);
                }
            };

            auto input_accessor_1d = std::move(input_accessors).map(to_1d);
            auto output_accessor_1d = std::move(output_accessors).map(to_1d);

            Iwise::launch<PARALLEL_ELEMENTS_THRESHOLD>(
                    Shape1<Index>{shape.elements()},
                    Ewise(input_accessor_1d, output_accessor_1d, std::forward<Op>(op)),
                    n_threads);

        } else {
            Iwise::launch(shape, std::forward<Op>(op), n_threads);
        }
        // Check special cases for aliasing?


    }
}
