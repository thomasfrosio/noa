#pragma once

#include "noa/core/Traits.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/types/Tuple.hpp"

/// These interfaces define the expected interface (duh) for each type of operators,
/// and provide the common functionalities of zip/unzip, optional member functions
/// and function redirections.

namespace noa::traits {
    template<typename T, typename = void>
    struct has_remove_defaulted_final : std::false_type {};
    template<typename T>
    struct has_remove_defaulted_final<T, std::void_t<typename T::remove_defaulted_final>> : std::true_type {};

    /// One issue with the defaulted final of the reduction interface, is that if the user provides a .final()
    /// member function but makes an error in the reduced/output types, instead of giving a compiler error, the
    /// interface will skip it and fall back on the default. This is a case of "it's not a bug it's a feature"
    /// because the user can use this to their advantage and provide "specialisation" for specific types. However,
    /// it is probably best to use function overloading in this case (and explicit write the default case). As a
    /// result, we need a way to make sure the .final() gets used: to do so, operators can define a type-alias
    /// named "remove_defaulted_final" (e.g. using remove_defaulted_final=bool) which the interface will detect
    /// and turn off the defaulted final. Then, if the provided .fina() doesn't exist or is not valid, a compiler
    /// error will be raised.
    template<typename T>
    constexpr bool has_remove_defaulted_final_v = has_remove_defaulted_final<std::decay_t<T>>::value;
}

namespace noa::guts {
    /// Index-wise interface.
    struct IwiseInterface {
        static constexpr void init(auto& op, auto id) {
            if constexpr (requires { op.init(id); }) // optional
                op.init(id);
        }

        template<typename Op, typename... Indices>
        requires nt::are_int_v<Indices...>
        static constexpr void call(Op& op, Indices... indices) {
            using packed_indices_t = Vec<nt::first_t<Indices...>, sizeof...(indices)>;
            if constexpr (requires { op(std::declval<packed_indices_t&>()); }) { // first try packed...
                // Do not allow the operator to take the indices by rvalue reference.
                auto v = Vec{indices...};
                op(v);
            } else if constexpr (requires { op(indices...); }) { // ...then unpacked
                op(indices...);
            } else { // this is not optional, so fail otherwise
                static_assert(nt::always_false_v<Op>,
                        "The provided index-wise operator is not compatible with the provided indices");
            }
        }

        static constexpr void final(auto& op, auto id) {
            if constexpr (requires { op.final(id); }) // optional
                op.final(id);
        }
    };

    template<bool ZipInput, bool ZipOutput>
    struct EwiseInterface {
        static constexpr void init(auto& op, auto integer) {
            if constexpr (requires { op.init(integer); }) // optional
                op.init(integer);
        }

        template<typename Op, typename Input, typename Output, typename... Indices>
        requires nt::are_tuple_v<Input, Output>
        static constexpr void call(Op& op, Input& input, Output& output, Indices... indices) {
            [&]<size_t... I, size_t... O>(std::index_sequence<I...>, std::index_sequence<O...>) {
                // "input" and "output" are accessors, so we know that the operator() returns a lvalue reference.
                // forward_as_tuple will create a tuple of these lvalue references; there's nothing being
                // moved or taken by value.
                // Also, the zipped parameters should not be passed as rvalues.
                if constexpr (ZipInput && ZipOutput) {
                    auto pi = forward_as_tuple(input[Tag<I>{}](indices...)...);
                    auto po = forward_as_tuple(output[Tag<O>{}](indices...)...);
                    op(pi, po);
                } else if constexpr (ZipInput) {
                    auto pi = forward_as_tuple(input[Tag<I>{}](indices...)...);
                    op(pi, output[Tag<O>{}](indices...)...);
                } else if constexpr (ZipOutput) {
                    auto po = forward_as_tuple(output[Tag<O>{}](indices...)...);
                    op(input[Tag<I>{}](indices...)..., po);
                } else {
                    op(input[Tag<I>{}](indices...)...,
                       output[Tag<O>{}](indices...)...);
                }
            }(nt::index_list_t<Input>{},
              nt::index_list_t<Output>{});
        }

        static constexpr void final(auto& op, auto id) {
            if constexpr (requires { op.final(id); }) // optional
                op.final(id);
        }
    };

    template<bool ZipReduced>
    struct ReduceInterfaceJoin {
        template<typename Op, typename Reduced>
        requires nt::is_tuple_of_accessor_value_v<Reduced>
        static constexpr void join(Op& op, Reduced& to_reduce, Reduced& reduced) {
            [&]<size_t... R>(std::index_sequence<R...> index_sequence) {
                if constexpr (ZipReduced) {
                    join_or_call_(
                            op,
                            forward_as_tuple(forward_as_tuple(to_reduce[Tag<R>{}].deref()...)),
                            forward_as_tuple(forward_as_tuple(reduced[Tag<R>{}].deref()...)),
                            std::index_sequence<0>{});
                } else {
                    join_or_call_(
                            op,
                            forward_as_tuple(to_reduce[Tag<R>{}].deref()...),
                            forward_as_tuple(reduced[Tag<R>{}].deref()...),
                            index_sequence);
                }
            }(nt::index_list_t<Reduced>{});
        }

    private:
        template<size_t... R>
        static constexpr void join_or_call_(auto& op, auto&& to_reduce, auto&& reduced, std::index_sequence<R...>) {
            if constexpr (requires { op.join(to_reduce[Tag<R>{}]..., reduced[Tag<R>{}]...); }) {
                op.join(to_reduce[Tag<R>{}]..., reduced[Tag<R>{}]...);
            } else {
                op(to_reduce[Tag<R>{}]..., reduced[Tag<R>{}]...);
            }
        }
    };

    template<bool ZipReduced, bool ZipOutput>
    struct ReduceInterfaceFinal {
        template<typename Op, typename Reduced, typename Output, typename... Indices>
        requires (nt::is_tuple_of_accessor_value_v<Reduced> and nt::is_tuple_of_accessor_or_empty_v<Output>)
        static constexpr void final(Op& op, Reduced& reduced, Output& output, Indices... indices) {
            [&]<size_t... R, size_t... O, typename... T>
                    (std::index_sequence<R...>, std::index_sequence<O...>, nt::TypeList<T...>) {

                using packed_reduced_t = decltype(forward_as_tuple(reduced[Tag<R>{}].deref()...));
                using packed_output_t = decltype(forward_as_tuple(output[Tag<O>{}](indices...)...));

                if constexpr (ZipReduced and ZipOutput and requires(packed_reduced_t pr, packed_output_t po) { op.final(pr, po); }) {
                    auto pr = forward_as_tuple(reduced[Tag<R>{}].deref()...);
                    auto po = forward_as_tuple(output[Tag<O>{}](indices...)...);
                    op.final(pr, po);

                } else if constexpr (ZipReduced and not ZipOutput and requires(packed_reduced_t pr) { op.final(pr, output[Tag<O>{}](indices...)...); }) {
                    auto pr = forward_as_tuple(reduced[Tag<R>{}].deref()...);
                    op.final(pr, output[Tag<O>{}](indices...)...);

                } else if constexpr (not ZipReduced and ZipOutput and requires(packed_output_t po) { op.final(reduced[Tag<R>{}].deref()..., po); }) {
                    auto po = forward_as_tuple(output[Tag<O>{}](indices...)...);
                    op.final(reduced[Tag<R>{}].deref()..., po);

                } else if constexpr (not ZipReduced and not ZipOutput and requires{ op.final(reduced[Tag<R>{}].deref()..., output[Tag<O>{}](indices...)...); }) {
                    op.final(reduced[Tag<R>{}].deref()..., output[Tag<O>{}](indices...)...);

                } else if constexpr (nt::is_tuple_of_accessor_v<Output>) { // turn off if no outputs
                    if constexpr (nt::has_remove_defaulted_final_v<Op>) {
                        static_assert(nt::always_false_v<Op>,
                                "Implicit .final() was removed, but no explicit .final() was detected");
                    } else {
                        // Default copy assignment, with explicit cast.
                        // TODO Here we could perfectly forward the reduced values into the outputs.
                        ((output[Tag<O>{}](indices...) = static_cast<T::mutable_value_type>(reduced[Tag<R>{}].deref())), ...);
                    }
                }
            }(nt::index_list_t<Reduced>{}, nt::index_list_t<Output>{}, nt::type_list_t<Output>{});
        }
    };

    template<bool ZipReduced, bool ZipOutput>
    struct ReduceIwiseInterface : ReduceInterfaceJoin<ZipReduced>, ReduceInterfaceFinal<ZipReduced, ZipOutput> {
        template<typename Op, typename Reduced, typename... Indices>
        requires (nt::is_tuple_of_accessor_value_v<Reduced> and nt::are_int_v<Indices...>)
        static constexpr void init(Op& op, Reduced& reduced, Indices... indices) {
            [&]<size_t... R>(std::index_sequence<R...> index_sequence) {
                if constexpr (ZipReduced) {
                    init_or_call_(
                            op,
                            forward_as_tuple(forward_as_tuple(reduced[Tag<R>{}].deref()...)),
                            std::index_sequence<0>{}, indices...);
                } else {
                    init_or_call_(
                            op,
                            forward_as_tuple(reduced[Tag<R>{}].deref()...),
                            index_sequence, indices...);
                }
            }(nt::index_list_t<Reduced>{});
        }

    private:
        template<size_t... I, size_t... R, typename... Indices>
        static constexpr void init_or_call_(auto& op, auto&& reduced, std::index_sequence<R...>, Indices... indices) {
            using packed_indices_t = Vec<nt::first_t<Indices...>, sizeof...(indices)>;
            if constexpr (requires (packed_indices_t packed) { op.init(packed, reduced[Tag<R>{}]...); }) {
                auto packed = packed_indices_t{indices...};
                op.init(packed, reduced[Tag<R>{}]...);
            } else if constexpr (requires { op.init(indices..., reduced[Tag<R>{}]...); }) {
                op.init(indices..., reduced[Tag<R>{}]...);
            } else if constexpr (requires (packed_indices_t packed) { op(packed, reduced[Tag<R>{}]...); }) {
                auto packed = packed_indices_t{indices...};
                op(packed, reduced[Tag<R>{}]...);
            } else {
                op(indices..., reduced[Tag<R>{}]...);
            }
        }
    };

    template<bool ZipInput, bool ZipReduced, bool ZipOutput>
    struct ReduceEwiseInterface : ReduceInterfaceJoin<ZipReduced>, ReduceInterfaceFinal<ZipReduced, ZipOutput> {
        template<typename Op, typename Input, typename Reduced, typename... Indices>
        requires (nt::is_tuple_of_accessor_v<Input> and
                  nt::is_tuple_of_accessor_value_v<Reduced> and
                  nt::are_int_v<Indices...>)
        static constexpr void init(Op& op, Input& input, Reduced& reduced, Indices... indices) {
            [&]<size_t... I, size_t... R>
            (std::index_sequence<I...> input_sequence, std::index_sequence<R...> reduced_sequence) {
                if constexpr (ZipInput && ZipReduced) {
                    init_or_call_(
                            op,
                            forward_as_tuple(forward_as_tuple(input[Tag<I>{}](indices...)...)),
                            forward_as_tuple(forward_as_tuple(reduced[Tag<R>{}].deref()...)),
                            std::index_sequence<0>{}, std::index_sequence<0>{});
                } else if constexpr (ZipInput) {
                    init_or_call_(
                            op,
                            forward_as_tuple(forward_as_tuple(input[Tag<I>{}](indices...)...)),
                            forward_as_tuple(reduced[Tag<R>{}].deref()...),
                            std::index_sequence<0>{}, reduced_sequence);
                } else if constexpr (ZipReduced) {
                    init_or_call_(
                            op,
                            forward_as_tuple(input[Tag<I>{}](indices...)...),
                            forward_as_tuple(forward_as_tuple(reduced[Tag<R>{}].deref()...)),
                            input_sequence, std::index_sequence<0>{});
                } else {
                    init_or_call_(
                            op,
                            forward_as_tuple(input[Tag<I>{}](indices...)...),
                            forward_as_tuple(reduced[Tag<R>{}].deref()...),
                            input_sequence, reduced_sequence);
                }
            }(nt::index_list_t<Input>{}, nt::index_list_t<Reduced>{});
        }

    private:
        // We don't want to perfect forward the tuples here: the operator should take the tuple elements
        // as (const) lvalue reference or by value, not by rvalue reference. As such, do not std::forward.
        template<size_t... I, size_t... R>
        static constexpr void init_or_call_(
                auto& op, auto&& input, auto&& reduced,
                std::index_sequence<I...>, std::index_sequence<R...>
        ) {
            if constexpr (requires { op.init(input[Tag<I>{}]..., reduced[Tag<R>{}]...); }) {
                op.init(input[Tag<I>{}]..., reduced[Tag<R>{}]...);
            } else {
                op(input[Tag<I>{}]..., reduced[Tag<R>{}]...);
            }
        }
    };
}
