#pragma once

#include "noa/cpu/ReduceIwise.hpp"

namespace noa::cpu {
    /// Index-wise reduce operator wrapper of an element-wise reduce operator.
    /// \tparam N       Number of dimensions
    /// \tparam Index   Integer type for indices.
    template<typename InputAccessors, typename Op>
    class ReduceEwise {
    public:
        /// Default threshold to trigger the multi-threaded implementation.
        /// This is a large number of elements to make sure there's enough work for each thread.
        static constexpr i64 PARALLEL_ELEMENTS_THRESHOLD = 1'048'576; // 1024x1024

    private:
        const InputAccessors& m_inputs;
        Op m_reduce_ewise_op; // TODO Can we take this by reference?

    public:
        /// Move everything into this operator.
        constexpr ReduceEwise(const InputAccessors& inputs, Op&& op)
                : m_inputs{inputs}, m_reduce_ewise_op{std::forward(op)} {}

        /// Read each input value at the nd-indices, forward them to the init function of the reduce-ewise operator.
        template<typename... Indexes, typename... Reduced>
        constexpr void init(Indexes... indexes, Reduced&... reduced) {
            using base_list = typename std::remove_reference_t<InputAccessors>::base_list;
            [&, this]<typename... I>(nt::TypeList<I...>) {
                m_reduce_ewise_op.init(m_inputs.::noa::traits::identity_t<I>::value(indexes...)..., reduced...);
            }(base_list{});
        }

        template<typename... Args>
        constexpr void join(Args&&... args) {
            m_reduce_ewise_op.join(std::forward<Args>(args)...);
        }

        template<typename... Args>
        constexpr void final(Args&&... args) {
            m_reduce_ewise_op.final(std::forward<Args>(args)...);
        }
    };
}
