#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/types/Tuple.hpp"
#include "noa/core/utils/EwiseAdaptor.hpp"

/// Check core-operators and define core concepts for the operators.

// TODO Move check and wrapper to the same type.
//      Check should only call requires/is_invocable on the actual function
namespace noa::guts {
    template<typename Op, typename Index, Index... indices>
    struct IwiseOpChecker {
        using index_type = Index;
        static constexpr size_t SIZE = sizeof...(indices);
        static constexpr bool has_unpacked_call = std::is_invocable_v<Op, decltype(indices)...>;
        static constexpr bool has_packed_call = std::is_invocable_v<Op, Vec<index_type, SIZE>>;
        static constexpr bool has_call = has_unpacked_call or has_packed_call;
        static constexpr bool has_init = requires(Op op) { op.init(index_type{}); };
        static constexpr bool has_final = requires(Op op) { op.final(index_type{}); };
    };

    struct IwiseOpWrapper {
        template<typename... Indices>
        static constexpr void call(auto& op, Indices&& ... indices) {
            if constexpr (requires { op(Vec{indices...}); }) {
                op(Vec{std::forward<Indices>(indices)...});
            } else if constexpr (requires { op(indices...); }) {
                op(std::forward<Indices>(indices)...);
            } else {
                static_assert(nt::always_false_v<decltype(op)>);
            }
        };
    };
}

namespace noa::guts {
    template<typename Op, EwiseAdaptorTags tags, typename InputTuple, typename OutputTuple>
    struct EwiseOpChecker {
        template<typename T>
        using remove_ref_t = std::remove_reference_t<T>;

        using input_accesors = std::remove_reference_t<InputTuple>::element_list;
        using output_accessors = std::remove_reference_t<OutputTuple>::element_list;

        static constexpr bool has_operator_call = has_operator_call_(input_accesors{}, output_accessors{});
        static constexpr bool has_init = requires(Op op) { op.init(int{}); };
        static constexpr bool has_final = requires(Op op) { op.final(int{}); };

    private:
        template<typename... Inputs, typename... Outputs>
        bool has_operator_call_(nt::TypeList<Inputs...>, nt::TypeList<Outputs...>) {
            using input_tuple = Tuple<typename remove_ref_t<Inputs>::reference_type...>&&;
            using output_tuple = Tuple<typename remove_ref_t<Outputs>::reference_type...>&&;

            if constexpr (tags.zip_inputs and tags.zip_outputs) {
                return std::is_invocable_v<Op, input_tuple, output_tuple>;
            } else if constexpr (tags.zip_inputs) {
                return std::is_invocable_v<Op, input_tuple, typename remove_ref_t<Outputs>::reference_type...>;
            } else if constexpr (tags.zip_outputs) {
                return std::is_invocable_v<Op, typename remove_ref_t<Inputs>::reference_type..., output_tuple>;
            } else {
                return std::is_invocable_v<Op,
                                           typename remove_ref_t<Inputs>::reference_type...,
                                           typename remove_ref_t<Outputs>::reference_type...>;
            }
        }
    };

    struct EwiseOpWrapper {
        template<typename T>
        using id_t = ::std::type_identity_t<T>;

        template<bool ZIP_INPUTS, bool ZIP_OUTPUTS, typename Inputs, typename Outputs, typename... Indices>
        static constexpr void call(auto& op, Inputs&& inputs, Outputs&& outputs, Indices&& ... indices) {
            [&]<typename... I, typename... O>(nt::TypeList<I...>, nt::TypeList<O...>) {
                if constexpr (ZIP_INPUTS && ZIP_OUTPUTS) {
                    op(forward_as_tuple(inputs.id_t<I>::value(indices...)...),
                       forward_as_tuple(outputs.id_t<O>::value(indices...)...));
                } else if constexpr (ZIP_INPUTS) {
                    op(forward_as_tuple(inputs.id_t<I>::value(indices...)...),
                       outputs.id_t<O>::value(indices...)...);
                } else if constexpr (ZIP_OUTPUTS) {
                    op(inputs.id_t<I>::value(indices...)...,
                       forward_as_tuple(outputs.id_t<O>::value(indices...)...));
                } else {
                    op(inputs.id_t<I>::value(indices...)...,
                       outputs.id_t<O>::value(indices...)...);
                }
            }(typename Inputs::base_list{},
              typename Outputs::base_list{});
        }
    };
}

namespace noa::guts {
    struct ReduceIwiseOpWrapper {
        template<typename T> using id_t = ::std::type_identity_t<T>;
        template<typename T> using remove_ref_t = std::remove_reference_t<T>;

        template<bool ZIP_INPUTS, bool ZIP_OUTPUTS,
                 typename Operator, typename Reduced, typename... Indices>
        static constexpr void init(
                Operator& op,
                Reduced&& reduced,
                const Indices& ... indices
        ) {
            using packed_indices = Vec<nt::first_t<Indices...>, sizeof...(indices)>;
            [&]<typename... R>(nt::TypeList<R...>) {
                if constexpr (ZIP_INPUTS && ZIP_OUTPUTS) {
                    init_or_call_(
                            op,
                            forward_as_tuple(packed_indices{indices...}),
                            forward_as_tuple(forward_as_tuple(reduced.id_t<R>::value.deref()...)));
                } else if constexpr (ZIP_INPUTS) {
                    init_or_call_(
                            op,
                            forward_as_tuple(packed_indices{indices...}),
                            forward_as_tuple(reduced.id_t<R>::value.deref()...));
                } else if constexpr (ZIP_OUTPUTS) {
                    init_or_call_(
                            op,
                            forward_as_tuple(indices...),
                            forward_as_tuple(forward_as_tuple(reduced.id_t<R>::value.deref()...)));
                } else {
                    init_or_call_(
                            op,
                            forward_as_tuple(indices...),
                            forward_as_tuple(reduced.id_t<R>::value.deref()...));
                }
            }(typename remove_ref_t<Reduced>::base_list{});
        }

        template<bool ZIP_INPUTS, bool ZIP_OUTPUTS,
                 typename Operator, typename Inputs, typename Outputs>
        static constexpr void join(
                Operator& op,
                Inputs&& inputs,
                Outputs&& outputs
        ) {
            [&]<typename... I, typename... O>(nt::TypeList<I...>, nt::TypeList<O...>) {
                if constexpr (ZIP_INPUTS && ZIP_OUTPUTS) {
                    join_or_call_(
                            op,
                            forward_as_tuple(forward_as_tuple(inputs.id_t<I>::value.deref()...)),
                            forward_as_tuple(forward_as_tuple(outputs.id_t<O>::value.deref()...)));
                } else if constexpr (ZIP_INPUTS) {
                    join_or_call_(
                            op,
                            forward_as_tuple(forward_as_tuple(inputs.id_t<I>::value.deref()...)),
                            forward_as_tuple(outputs.id_t<O>::value.deref()...));
                } else if constexpr (ZIP_OUTPUTS) {
                    join_or_call_(
                            op,
                            forward_as_tuple(inputs.id_t<I>::value.deref()...),
                            forward_as_tuple(forward_as_tuple(outputs.id_t<O>::value.deref()...)));
                } else {
                    join_or_call_(
                            op,
                            forward_as_tuple(inputs.id_t<I>::value.deref()...),
                            forward_as_tuple(outputs.id_t<O>::value.deref()...));
                }
            }(typename remove_ref_t<Inputs>::base_list{},
              typename remove_ref_t<Outputs>::base_list{});
        }

        template<bool ZIP_INPUTS, bool ZIP_OUTPUTS,
                typename Operator, typename Inputs, typename Outputs>
        static constexpr void final(
                Operator& op,
                Inputs&& inputs,
                Outputs&& outputs
        ) {
            [&]<typename... I, typename... O>(nt::TypeList<I...>, nt::TypeList<O...>) {
                if constexpr (ZIP_INPUTS && ZIP_OUTPUTS) {
                    final_or_assign(
                            op,
                            forward_as_tuple(forward_as_tuple(inputs.id_t<I>::value.deref()...)),
                            forward_as_tuple(forward_as_tuple(outputs.id_t<O>::value.deref()...)));
                } else if constexpr (ZIP_INPUTS) {
                    final_or_assign(
                            op,
                            forward_as_tuple(forward_as_tuple(inputs.id_t<I>::value.deref()...)),
                            forward_as_tuple(outputs.id_t<O>::value.deref()...));
                } else if constexpr (ZIP_OUTPUTS) {
                    final_or_assign(
                            op,
                            forward_as_tuple(inputs.id_t<I>::value.deref()...),
                            forward_as_tuple(forward_as_tuple(outputs.id_t<O>::value.deref()...)));
                } else {
                    final_or_assign(
                            op,
                            forward_as_tuple(inputs.id_t<I>::value.deref()...),
                            forward_as_tuple(outputs.id_t<O>::value.deref()...));
                }
            }(typename remove_ref_t<Inputs>::base_list{},
              typename remove_ref_t<Outputs>::base_list{});
        }

    private:
        // FIXME We never move a value here, so Inputs and Outputs are tuple of l-values references.
        //       So we want to take by forward reference (which we do), but the std::forward is useless, so remove it?
        template<typename Op, typename Inputs, typename Outputs>
        static constexpr void init_or_call(Op& op, Inputs&& inputs, Outputs&& outputs) {
            [&op, &inputs, &outputs]<typename... I, typename... O>(nt::TypeList<I...>, nt::TypeList<O...>) {
                [&op]<typename... T>(T&& ...args) {
                    if constexpr (requires { op.init(std::forward<T>(args)...); }) {
                        op.init(std::forward<T>(args)...);
                    } else {
                        op(std::forward<T>(args)...);
                    }
                }(std::forward<Inputs>(inputs).id_t<I>::value...,
                  std::forward<Outputs>(outputs).id_t<O>::value...);
            }(typename remove_ref_t<Inputs>::base_list{},
              typename remove_ref_t<Outputs>::base_list{});
        }

        template<typename Op, typename Inputs, typename Outputs>
        static constexpr void join_or_call(Op& op, Inputs&& inputs, Outputs&& outputs) {
            [&op, &inputs, &outputs]<typename... I, typename... O>(nt::TypeList<I...>, nt::TypeList<O...>) {
                [&op]<typename... T>(T&& ...args) {
                    if constexpr (requires { op.join(std::forward<T>(args)...); }) {
                        op.join(std::forward<T>(args)...);
                    } else {
                        op(std::forward<T>(args)...);
                    }
                }(std::forward<Inputs>(inputs).id_t<I>::value...,
                  std::forward<Outputs>(outputs).id_t<O>::value...);
            }(typename remove_ref_t<Inputs>::base_list{},
              typename remove_ref_t<Outputs>::base_list{});
        }

        template<typename Op, typename Inputs, typename Outputs>
        static constexpr void final_or_assign(Op& op, Inputs&& inputs, Outputs&& outputs) {
            [&op, &inputs, &outputs]<typename... I, typename... O>(nt::TypeList<I...>, nt::TypeList<O...>) {
                [&op, &inputs, &outputs]<typename... T>(T&& ...args) {
                    if constexpr (requires { op.init(std::forward<T>(args)...); }) {
                        op.final(std::forward<T>(args)...);
                    } else {
                        ((outputs.id_t<O>::value =
                                  static_cast<std::decay_t<typename O::type>>(inputs.id_t<I>::value)
                        ), ...);
                    }
                }(std::forward<Inputs>(inputs).id_t<I>::value...,
                  std::forward<Outputs>(outputs).id_t<O>::value...);
            }(typename remove_ref_t<Inputs>::base_list{},
              typename remove_ref_t<Outputs>::base_list{});
        }
    };

    ///
    /// \tparam N
    /// \tparam Op
    /// \tparam Index
    /// \tparam Reduced
    /// \tparam Output
    template<size_t N, typename Op, typename Index, typename Reduced, typename Output>
    requires std::is_integral_v<Index> && nt::are_tuple_v<Reduced, Output>
    struct ReduceIwiseChecker {
        static constexpr size_t N_DIMENSIONS = N;
        using index_type = Index;

        static constexpr auto has_init() -> bool { return m_has_init.first; }
        static constexpr auto is_init_packed() -> bool { return m_has_init.second; }
        static constexpr auto has_join() -> bool { return m_has_join; }
        static constexpr auto has_final() -> bool { return m_has_final.first; }
        static constexpr auto is_final_defaulted() -> bool { return m_has_final.second; }
        static constexpr auto is_valid() -> bool { return has_init() && has_join() && has_final(); }

    private:
        using reduced_element_list = std::remove_reference_t<Reduced>::element_list;
        using output_element_list = std::remove_reference_t<Output>::element_list;
        static constexpr Pair m_has_init = check_init_(nt::repeat<Index, N, nt::TypeList>{}, reduced_element_list{});
        static constexpr bool m_has_join = check_join_(reduced_element_list{});
        static constexpr Pair m_has_final = check_final_(reduced_element_list{}, output_element_list{});

        template<typename... Indices, typename... Accessors>
        static constexpr Pair<bool, bool> check_init_(nt::TypeList<Indices...>, nt::TypeList<Accessors...>) noexcept {
            if constexpr (true) {
                return {std::is_invocable_v<Op, Vec<Index, N>, typename Accessors::reference_type...>,
                        std::is_invocable_v<Op, Indices..., typename Accessors::reference_type...>};
            } else {
                return {std::is_invocable_v<Op, Vec<Index, N>, Tuple<typename Accessors::reference_type...>>,
                        std::is_invocable_v<Op, Indices..., Tuple<typename Accessors::reference_type...>>};
            }
        }

        template<typename... Accessors>
        static constexpr bool check_join_(nt::TypeList<Accessors...>) noexcept {
            constexpr bool valid = requires (Op op, Accessors::value_type... values) {
                op.join(values..., values...);
            };
            return valid;
        }

        template<typename... R, typename... O>
        static constexpr auto check_final_(nt::TypeList<R...>, nt::TypeList<O...>) noexcept {
            constexpr bool valid = requires (Op op, R::value_type... reduced, O::value_type... outputs) {
                op.final(reduced..., outputs...);
            };
            constexpr bool is_defaulted = valid ? false :
            requires (R::value_type... reduced, O::value_type... outputs) {
                ((outputs = static_cast<O::value_type>(reduced)), ...);
            };
            return Pair{valid || is_defaulted, is_defaulted};
        }
    };

    template<typename Op, typename Input, typename Reduced, typename Output>
    requires nt::is_tuple_v<Input> && nt::are_tuple_v<Reduced, Output>
    struct ReduceEwiseChecker {
        static constexpr auto has_init() -> bool { return m_has_init; }
        static constexpr auto has_join() -> bool { return m_has_join; }
        static constexpr auto has_final() -> bool { return m_has_final.first; }
        static constexpr auto is_final_defaulted() -> bool { return m_has_final.second; }
        static constexpr auto is_valid() -> bool { return has_init() && has_join() && has_final(); }

    private:
        using input_element_list = std::remove_reference_t<Input>::element_list;
        using reduced_element_list = std::remove_reference_t<Reduced>::element_list;
        using output_element_list = std::remove_reference_t<Output>::element_list;
        static constexpr bool m_has_init = check_init_(input_element_list{}, reduced_element_list{});
        static constexpr bool m_has_join = check_join_(reduced_element_list{});
        static constexpr Pair m_has_final = check_final_(reduced_element_list{}, output_element_list{});

        template<typename... I, typename... R>
        static constexpr bool check_init_(nt::TypeList<I...>, nt::TypeList<R...>) noexcept {
            return requires (Op op, I::value_type... inputs, R::value_type... values) {
                op.init(inputs..., values...);
            };
        }

        template<typename... Accessors>
        static constexpr bool check_join_(nt::TypeList<Accessors...>) noexcept {
            return requires (Op op, Accessors::value_type... values) {
                op.join(values..., values...);
            };
        }

        template<typename... R, typename... O>
        static constexpr auto check_final_(nt::TypeList<R...>, nt::TypeList<O...>) noexcept {
            constexpr bool valid = requires (Op op, R::value_type... reduced, O::value_type... outputs) {
                op.join(reduced..., outputs...);
            };
            constexpr bool is_defaulted = valid ? false :
            requires (R::value_type... reduced, O::value_type... outputs) {
                ((outputs = static_cast<O::value_type>(reduced)), ...);
            };
            return Pair{valid || is_defaulted, is_defaulted};
        }
    };
}

namespace noa {
    /// Index-wise operator
    template<typename Op, typename Index, size_t N>
    concept IwiseOp = guts::IwiseOpChecker<Op, Index, N>::has_operator_call;

    template<typename Op, guts::EwiseAdaptorTags tags, typename InputTuple, typename OutputTuple>
    concept EwiseOp = guts::EwiseOpChecker<Op, tags, InputTuple, OutputTuple>::has_operator_call;
}
