#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Accessor.hpp"
#include "noa/core/types/Vec.hpp"
#include "noa/core/types/Tuple.hpp"

/// These define the expected interface for each type of operators, and provide the common functionalities
/// of zip/unzip, optional member functions, and function redirections.

namespace noa::traits {
    template<typename T, typename = void>
    struct has_allow_vectorization : std::false_type {};
    template<typename T>
    struct has_allow_vectorization<T, std::void_t<typename T::allow_vectorization>> : std::true_type {};

    /// By default, the element-wise interfaces (ewise, reduce_ewise and reduce_axes_ewise) allow the operators
    /// to write from the inputs and read from the outputs. While this can be useful for some operations, it may also
    /// constrain some backends when it comes to vectorization (e.g. vectorized load/write on CUDA). Operators can
    /// define the optional type alias "allow_vectorization" to indicate that the input values are read-only and
    /// that output values are write-only, which can help backends to generate vectorized code.
    template<typename T>
    constexpr bool has_allow_vectorization_v = has_allow_vectorization<std::decay_t<T>>::value;

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

// nvcc (12.0-12.3) easily breaks (or worse, give wrong results) with `if constexpr (requires...)` statements,
// so we have to walk on eggshells and define "friendlier" structures to check these type substitutions...
// std::declval seems necessary, as using the name of variadics seem to freak out nvcc...
namespace noa::guts {
    struct IwiseChecker {
        template<typename Op, typename... I>
        NOA_HD constexpr auto operator()(Tag<1>, Op, I...) requires requires {
            std::declval<Op&>()(std::declval<decltype(Vec{std::declval<I>()...})&>());
        } {}

        template<typename Op, typename... I>
        NOA_HD constexpr auto operator()(Tag<2>, Op, I...) requires requires {
            std::declval<Op&>()(std::declval<I>()...);
        } {}
    };

    struct ReduceJoinChecker {
        template<typename Op, typename T, typename U, size_t... R>
        NOA_HD constexpr auto operator()(Op, T, U, std::index_sequence<R...>) requires requires {
            std::declval<Op>().join(std::declval<T&>()[Tag<R>{}]..., std::declval<U&>()[Tag<R>{}]...);
        } {}
    };

    struct ReduceFinalChecker {
        NOA_HD static constexpr size_t tag(bool zip_reduced, bool zip_output) {
            return static_cast<size_t>(zip_reduced) * 10 + static_cast<size_t>(zip_output);
        };

        template<typename Op, typename R, typename O, size_t... R0, size_t... O0, typename... I>
        NOA_HD constexpr auto operator()(Tag<11>, Op, R, O, std::index_sequence<R0...>, std::index_sequence<O0...>, I...) requires requires {
            std::declval<Op>().final(
                    std::declval<decltype(forward_as_tuple(std::declval<R&>()[Tag<R0>{}].deref()...))&>(),
                    std::declval<decltype(forward_as_tuple(std::declval<O&>()[Tag<O0>{}](std::declval<I>()...)...))&>()
                    );
        } {}

        template<typename Op, typename R, typename O, size_t... R0, size_t... O0, typename... I>
        NOA_HD constexpr auto operator()(Tag<10>, Op, R, O, std::index_sequence<R0...>, std::index_sequence<O0...>, I...) requires requires {
            std::declval<Op>().final(
                    std::declval<decltype(forward_as_tuple(std::declval<R&>()[Tag<R0>{}].deref()...))&>(),
                    std::declval<O&>()[Tag<O0>{}](std::declval<I>()...)...
            );
        } {}

        template<typename Op, typename R, typename O, size_t... R0, size_t... O0, typename... I>
        NOA_HD constexpr auto operator()(Tag<1>, Op, R, O, std::index_sequence<R0...>, std::index_sequence<O0...>, I...) requires requires {
            std::declval<Op>().final(
                    std::declval<R&>()[Tag<R0>{}].deref()...,
                    std::declval<decltype(forward_as_tuple(std::declval<O&>()[Tag<O0>{}](std::declval<I>()...)...))&>()
            );
        } {}

        template<typename Op, typename R, typename O, size_t... R0, size_t... O0, typename... I>
        NOA_HD constexpr auto operator()(Tag<0>, Op, R, O, std::index_sequence<R0...>, std::index_sequence<O0...>, I...) requires requires {
            std::declval<Op&>().final(
                    std::declval<R&>()[Tag<R0>{}].deref()...,
                    std::declval<O&>()[Tag<O0>{}](std::declval<I>()...)...
            );
        } {}
    };

    struct ReduceIwiseChecker {
        template<typename Op, typename R, size_t... R0, typename... I>
        NOA_HD constexpr auto operator()(Tag<0>, Op, R, std::index_sequence<R0...>, I...) requires requires {
            std::declval<Op>().init(
                    std::declval<decltype(Vec{std::declval<I>()...})&>(),
                    std::declval<R&>()[Tag<R0>{}]...
            );
        } {}

        template<typename Op, typename R, size_t... R0, typename... I>
        NOA_HD constexpr auto operator()(Tag<1>, Op, R, std::index_sequence<R0...>, I...) requires requires {
            std::declval<Op>().init(std::declval<I>()..., std::declval<R&>()[Tag<R0>{}]...);
        } {}

        template<typename Op, typename R, size_t... R0, typename... I>
        NOA_HD constexpr auto operator()(Tag<2>, Op, R, std::index_sequence<R0...>, I...) requires requires {
            std::declval<Op>()(
                    std::declval<decltype(Vec{std::declval<I>()...})&>(),
                    std::declval<R&>()[Tag<R0>{}]...
            );
        } {}

        template<typename Op, typename R, size_t... R0, typename... I>
        NOA_HD constexpr auto operator()(Tag<3>, Op, R, std::index_sequence<R0...>, I...) requires requires {
            std::declval<Op>()(std::declval<I>()..., std::declval<R&>()[Tag<R0>{}]...);
        } {}
    };

    struct ReduceEwiseChecker {
        template<typename Op, typename Input, typename Reduced, size_t... I, size_t... R>
        NOA_HD constexpr auto operator()(Op, Input, Reduced, std::index_sequence<I...>, std::index_sequence<R...>) requires requires {
            std::declval<Op>().init(std::declval<Input&>()[Tag<I>{}]..., std::declval<Reduced&>()[Tag<R>{}]...);
        } {}
    };
}

namespace noa::guts {
    /// Index-wise interface.
    struct IwiseInterface {
        NOA_HD static constexpr void init(auto& op, auto id) {
            if constexpr (requires { op.init(id); }) // optional
                op.init(id);
        }

        template<typename Op, nt::integer... Indices>
        NOA_FHD static constexpr void call(Op& op, Indices... indices) {
            if constexpr (std::is_invocable_v<IwiseChecker, Tag<1>, Op, Indices...>) { // first try packed...
                auto v = Vec{indices...};
                op(v);
            } else if constexpr (std::is_invocable_v<IwiseChecker, Tag<2>, Op, Indices...>) { // ...then unpacked
                op(indices...);
            } else { // this is not optional, so fail otherwise
                static_assert(nt::always_false<Op>,
                        "The provided index-wise operator is not compatible with the provided indices");
            }
        }

        NOA_HD static constexpr void final(auto& op, auto id) {
            if constexpr (requires { op.final(id); }) // optional
                op.final(id);
        }
    };

    template<bool ZipInput, bool ZipOutput>
    struct EwiseInterface {
        NOA_FHD static constexpr void init(auto& op, auto integer) {
            if constexpr (requires { op.init(integer); }) // optional
                op.init(integer);
        }

        template<typename Op, nt::tuple Input, nt::tuple Output, nt::integer... Indices>
        NOA_FHD static constexpr void call(Op& op, Input& input, Output& output, Indices... indices) {
            [&]<size_t... I, size_t... O>(std::index_sequence<I...>, std::index_sequence<O...>) {
                // "input" and "output" are accessors, so we know that the operator() returns a lvalue reference.
                // forward_as_tuple will create a tuple of these lvalue references; there's nothing being
                // moved or taken by value.
                // Also, the zipped parameters should not be passed as rvalues.
                if constexpr (ZipInput and ZipOutput) {
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

        NOA_FHD static constexpr void final(auto& op, auto id) {
            if constexpr (requires { op.final(id); }) // optional
                op.final(id);
        }
    };

    template<bool ZipReduced>
    struct ReduceJoinInterface {
        template<typename Op, nt::tuple_of_accessor_value Reduced>
        NOA_FHD static constexpr void join(Op& op, Reduced& to_reduce, Reduced& reduced) {
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
        template<typename O, typename T, typename U, size_t... R>
        NOA_FHD static constexpr void join_or_call_(O& op, T&& to_reduce, U&& reduced, std::index_sequence<R...>) {
            if constexpr (std::is_invocable_v<ReduceJoinChecker, O, T, U, std::index_sequence<R...>>) {
                op.join(to_reduce[Tag<R>{}]..., reduced[Tag<R>{}]...);
            } else {
                op(to_reduce[Tag<R>{}]..., reduced[Tag<R>{}]...);
            }
        }
    };

    template<bool ZipReduced, bool ZipOutput>
    struct ReduceFinalInterface {
        template<typename Op,
                 nt::tuple_of_accessor_value Reduced,
                 nt::tuple_of_accessor_or_empty Output,
                 nt::integer... Indices>
        NOA_FHD static constexpr void final(Op& op, Reduced& reduced, Output& output, Indices... indices) {
            [&]<size_t... R, size_t... O, typename... T>
                    (std::index_sequence<R...> isr, std::index_sequence<O...> iso, nt::TypeList<T...> tl) {

                using has_final = std::is_invocable<
                        ReduceFinalChecker, Tag<ReduceFinalChecker::tag(ZipReduced, ZipOutput)>,
                        Op, Reduced, Output, std::index_sequence<R...>, std::index_sequence<O...>, Indices...>;

                // While we could have the following "inlined", nvcc doesn't like it...
                // so we have to split it in two different functions.
                if constexpr (has_final::value) {
                    final_(op, reduced, output, isr, iso, indices...);
                } else if constexpr (nt::tuple_of_accessor<Output>) { // turn off if no outputs
                    default_final_<Op>(reduced, output, isr, iso, tl, indices...);
                }
            }(nt::index_list_t<Reduced>{}, nt::index_list_t<Output>{}, nt::type_list_t<Output>{});
        }

    private:
        template<typename Op, typename Reduced, typename Output, size_t... R, size_t... O, typename... Indices>
        NOA_FHD static constexpr void final_(
                Op& op, Reduced& reduced, Output& output,
                std::index_sequence<R...>, std::index_sequence<O...>,
                Indices... indices
        ) {
            if constexpr (ZipReduced and ZipOutput) {
                auto pr = forward_as_tuple(reduced[Tag<R>{}].deref()...);
                auto po = forward_as_tuple(output[Tag<O>{}](indices...)...);
                op.final(pr, po);

            } else if constexpr (ZipReduced and not ZipOutput) {
                auto pr = forward_as_tuple(reduced[Tag<R>{}].deref()...);
                op.final(pr, output[Tag<O>{}](indices...)...);

            } else if constexpr (not ZipReduced and ZipOutput) {
                auto po = forward_as_tuple(output[Tag<O>{}](indices...)...);
                op.final(reduced[Tag<R>{}].deref()..., po);

            } else {
                op.final(reduced[Tag<R>{}].deref()..., output[Tag<O>{}](indices...)...);
            }
        }

        template<typename Op, typename Reduced, typename Output,
                 size_t... R, size_t... O,
                 typename...T, typename... Indices>
        NOA_FHD static constexpr void default_final_(
                Reduced& reduced, Output& output,
                std::index_sequence<R...>, std::index_sequence<O...>,
                nt::TypeList<T...>,
                Indices... indices
        ) {
            if constexpr (nt::has_remove_defaulted_final_v<Op>) {
                static_assert(nt::always_false<Op>,
                              "Defaulted .final() was removed using the remove_defaulted_final flag, but no explicit .final() was detected");
            } else {
                // Default copy assignment, with explicit cast.
                // TODO Here we could perfectly forward the reduced values into the outputs.
                ((output[Tag<O>{}](indices...) = static_cast<typename T::mutable_value_type>(reduced[Tag<R>{}].deref())), ...);
            }
        }
    };

    template<bool ZipReduced, bool ZipOutput>
    struct ReduceIwiseInterface : ReduceJoinInterface<ZipReduced>, ReduceFinalInterface<ZipReduced, ZipOutput> {
        template<typename Op, nt::tuple_of_accessor_value Reduced, nt::integer... Indices>
        NOA_FHD static constexpr void init(Op& op, Reduced& reduced, Indices... indices) {
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
        template<typename Op, typename R, size_t... R0, typename... Indices>
        NOA_FHD static constexpr void init_or_call_(Op& op, R&& reduced, std::index_sequence<R0...>, Indices... indices) {
            using packed_indices_t = Vec<nt::first_t<Indices...>, sizeof...(indices)>;
            if constexpr (std::is_invocable_v<ReduceIwiseChecker, Tag<0>, Op, R, std::index_sequence<R0...>, Indices...>) {
                auto packed = packed_indices_t{indices...};
                op.init(packed, reduced[Tag<R0>{}]...);
            } else if constexpr (std::is_invocable_v<ReduceIwiseChecker, Tag<1>, Op, R, std::index_sequence<R0...>, Indices...>) {
                op.init(indices..., reduced[Tag<R0>{}]...);
            } else if constexpr (std::is_invocable_v<ReduceIwiseChecker, Tag<2>, Op, R, std::index_sequence<R0...>, Indices...>) {
                auto packed = packed_indices_t{indices...};
                op(packed, reduced[Tag<R0>{}]...);
            } else if constexpr (std::is_invocable_v<ReduceIwiseChecker, Tag<3>, Op, R, std::index_sequence<R0...>, Indices...>) {
                op(indices..., reduced[Tag<R0>{}]...);
            } else {
                static_assert(nt::always_false<Op>);
            }
        }
    };

    template<bool ZipInput, bool ZipReduced, bool ZipOutput>
    struct ReduceEwiseInterface : ReduceJoinInterface<ZipReduced>, ReduceFinalInterface<ZipReduced, ZipOutput> {
        template<typename Op, nt::tuple_of_accessor Input, nt::tuple_of_accessor_value Reduced, nt::integer... Indices>
        NOA_FHD static constexpr void init(Op& op, Input& input, Reduced& reduced, Indices... indices) {
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
        template<typename Op, typename Input, typename Reduced, size_t... I, size_t... R>
        NOA_FHD static constexpr void init_or_call_(
                Op& op, Input&& input, Reduced&& reduced,
                std::index_sequence<I...>, std::index_sequence<R...>
        ) {
            if constexpr (std::is_invocable_v<ReduceEwiseChecker, Op, Input, Reduced, std::index_sequence<I...>, std::index_sequence<R...>>) {
                op.init(input[Tag<I>{}]..., reduced[Tag<R>{}]...);
            } else {
                op(input[Tag<I>{}]..., reduced[Tag<R>{}]...);
            }
        }
    };
}
