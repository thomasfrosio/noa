#pragma once

#include "noa/base/Tuple.hpp"
#include "noa/base/Vec.hpp"
#include "noa/runtime/core/Traits.hpp"

namespace noa::details {
    template<typename... T>
    struct AdaptorZip {
        static constexpr bool ZIP = true;
        Tuple<T...> tuple;
    };

    template<typename... T>
    struct AdaptorUnzip {
        static constexpr bool ZIP = false;
        Tuple<T...> tuple;
    };

    NOA_GENERATE_PROCLAIM_FULL(adaptor);
    template<typename... T> struct proclaim_is_adaptor<AdaptorZip<T...>> : std::true_type {};
    template<typename... T> struct proclaim_is_adaptor<AdaptorUnzip<T...>> : std::true_type {};

    template<typename... T>
    concept adaptor_decay = adaptor<std::decay_t<T>...>;
}

namespace noa {
    /// Core functions utility used to fuse arguments.
    /// - Wrapped arguments are fused into a Tuple, which is then passed to the operator.
    ///   The core functions handle the details of this operation, the adaptor is just a thin wrapper
    ///   used to store and forward the arguments to these core functions.
    /// - Temporaries (rvalue references) are moved (if their type is movable, otherwise they are copied) and
    ///   stored by value. Lvalue references are stored as such, i.e. no move/copy are involved.
    ///   See forward_as_final_tuple for more details.
    template<typename... T>
    constexpr auto fuse(T&&... a) noexcept {
        return nd::AdaptorZip{.tuple = noa::forward_as_final_tuple(std::forward<T>(a)...)};
    }

    /// Core functions utility used to wrap arguments.
    /// - Wrapped arguments are passed directly to the operator.
    ///   The core functions handle the details of this operation, the adaptor is just a thin wrapper
    ///   used to store and forward the arguments to these core functions.
    /// - Temporaries (rvalue references) are moved (if their type is movable, otherwise they are copied) and
    ///   stored by value. Lvalue references are stored as such, i.e. no move/copy are involved.
    ///   See forward_as_final_tuple for more details.
    template<typename... T>
    constexpr auto wrap(T&&... a) noexcept {
        return nd::AdaptorUnzip{.tuple=noa::forward_as_final_tuple(std::forward<T>(a)...)};
    }
}

/// These define the expected interface for each type of operators, and provide the common functionalities
/// of zip/unzip, optional member functions, and function redirections.

namespace noa::traits {
    template<typename T, typename = void>
    struct enable_vectorization : std::false_type {};
    template<typename T>
    struct enable_vectorization<T, std::void_t<typename T::enable_vectorization>> : std::true_type {};

    /// By default, the element-wise interfaces (ewise, reduce_ewise and reduce_axes_ewise) allow the operators
    /// to write from the inputs and read from the outputs. While this can be useful for some operations, it may also
    /// constrain some backends when it comes to vectorization (e.g. vectorized load/write in CUDA). Operators can
    /// define the optional type alias "enable_vectorization" to indicate that the input values are read-only and
    /// that output values must be initialized by the operator.
    template<typename T>
    constexpr bool enable_vectorization_v = enable_vectorization<std::decay_t<T>>::value;

    template<typename T, typename = void> struct remove_default_init : std::false_type {};
    template<typename T, typename = void> struct remove_default_deinit : std::false_type {};
    template<typename T, typename = void> struct remove_default_post : std::false_type {};
    template<typename T, typename = void> struct remove_compute_handle : std::false_type {};

    template<typename T> struct remove_default_init<T, std::void_t<typename T::remove_default_init>> : std::true_type {};
    template<typename T> struct remove_default_deinit<T, std::void_t<typename T::remove_default_deinit>> : std::true_type {};
    template<typename T> struct remove_default_post<T, std::void_t<typename T::remove_default_post>> : std::true_type {};
    template<typename T> struct remove_compute_handle<T, std::void_t<typename T::remove_compute_handle>> : std::true_type {};

    /// One issue with the default functions interface is that if the user provides a definition of (de)init but
    /// makes an error in the input arguments (e.g. by specifying forgetting const/ref to a type), instead of giving
    /// a compile-time error, the interface will skip it and fall back to the default implementation. This is a case
    /// of "it's not a bug, it's a feature" because the user can use this to their advantage and provide specializations
    /// for specific types. However, it is probably best to use function overloading in this case and explicitly write
    /// the desired default case. Regardless, we need a way to make sure the user definitions get used. To do so,
    /// operators can define type-aliases named "remove_default_{(de)init}" (e.g. using remove_default_init=bool) which
    /// the interface detects. Then, if the operator doesn't implement the corresponding function or is not valid,
    /// a compile-time error will be raised.
    template<typename T> constexpr bool remove_default_init_v = remove_default_init<std::decay_t<T>>::value;
    template<typename T> constexpr bool remove_default_deinit_v = remove_default_deinit<std::decay_t<T>>::value;
    template<typename T> constexpr bool remove_default_post_v = remove_default_post<std::decay_t<T>>::value;
    template<typename T> constexpr bool remove_compute_handle_v = remove_compute_handle<std::decay_t<T>>::value;
}

namespace noa::details {
    // nvcc (12.0-12.5) easily breaks (or worse, give wrong results) with `if constexpr (requires...)` statements,
    // so we have to walk on eggshells and define "friendlier" structures to check these type substitutions...
    // std::declval seems necessary, as using the name of variadics seem to freak out nvcc...
}

// Suppress spurious visibility warning with NVCC-GCC when a lambda captures anonymous types.
#ifdef __CUDACC__
#   if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#       pragma GCC diagnostic push
#       pragma GCC diagnostic ignored "-Wattributes"
#   elif defined(NOA_COMPILER_MSVC)
#       pragma warning(push, 0)
#   endif
#endif

namespace noa::traits {
    /// Hook to bypass the interface detection.
    template<typename> struct proclaim_is_compute_handle : std::false_type {};

    /// Compute handle.
    ///
    /// \note
    /// The API description below is for documentation only. Use the proclaim_is_compute_handle to make a type actually
    /// conform to the compute_handle concept. We unfortunately don't check for the API explicitly mostly because
    /// 1) C++20 concepts and require expressions have a rather limited ability to exhaustively detect APIs (how to say
    /// that a certain template function exists and test that, for instance, it can only accept integers?) and
    /// 2) because some of the compute_handle might be device-only code and nvcc freaks out.
    ///
    /// \details Grid of thread-blocks. Grids are 1d, 2d, or 3d.
    /// compute_handle.grid() -> {
    ///     Integer type used for indexing.
    ///     { T::index_type } -> integer;
    ///
    ///     Number of dimensions of the grid.
    ///     { grid.ndim() } -> same_as<typename T::index_type>;
    ///     { grid.template ndim<i32>() } -> same_as<i32>;
    ///
    ///     Grid size, i.e., number of blocks in the grid.
    ///     { grid.size() } -> same_as<typename T::index_type>;
    ///     { grid.template size<i32>() } -> same_as<i32>;
    ///
    ///     Grid shape, i.e., the number of blocks along the specified dimensions of the grid.
    ///     If a shape with a higher dimension than the grid is requested, these higher dimensions have a size of 1.
    ///     If a shape with a lower dimension than the grid is requested, the additional dimensions are simply ignored.
    ///     { grid.template shape<integer, 3>() } -> same_as<Shape<I, 3>>;
    ///
    ///     Atomically adds a value.
    ///     This function guarantees no data-races between the threads of the grid.
    ///     { grid.atomic_add(value, accessor, indices...)
    ///
    ///     Whether the grid is part of a two-part reduction
    ///     (and thus whether this is called from the first kernel).
    ///     { grid.is_two_part_reduction() } -> same_as<bool>;
    /// }
    ///
    /// \details Thread-blocks. Blocks are 1d, 2d, or 3d.
    /// compute_handle.block() -> {
    ///     Integer type used for indexing.
    ///     { T::index_type } -> integer;
    ///
    ///     Number of dimensions of the block.
    ///     { block.ndim() } -> same_as<typename T::index_type>;
    ///     { block.template ndim<I>() } -> same_as<I>;
    ///
    ///     Block size, i.e., number of threads in the block.
    ///     { block.size() } -> same_as<typename T::index_type>;
    ///     { block.template size<I>() } -> same_as<I>;
    ///
    ///     Block shape, i.e., the number of threads along the specified dimensions of the block.
    ///     If a shape with a higher dimension than the block is requested, these higher dimensions have a size of 1.
    ///     If a shape with a lower dimension than the block is requested, the additional dimensions are simply ignored.
    ///     { block.template shape<I, N>() } -> same_as<Shape<I, N>>;
    ///
    ///     Linear index of the block within the grid.
    ///     { block.lid() } -> same_as<typename T::index_type>;
    ///     { block.template lid<I>() } -> same_as<I>;
    ///
    ///     Returns the block nd-indices within the grid.
    ///     If the grid has fewer dimensions than the requested indices, the extra indices are set to 0.
    ///     If the grid has more dimensions than the requested indices, these higher dimensions are simply ignored.
    ///     { block.template id<I, N>() } -> same_as<Vec<I, N>>;
    ///
    ///     Whether the block has some scratch memory available.
    ///     In CUDA, the scratch is dynamic shared memory.
    ///     { block.has_scratch() } -> same_as<bool>;
    ///
    ///     Returns the scratch span, which can be empty if the block doesn't have any scratch memory available
    ///     { block.scratch() } -> details::scratch_span;
    ///     { block.template scratch<T, I>() } -> details::scratch_span<T, I>;
    ///
    ///     Returns the scratch pointer, which can be null if the block doesn't have any scratch memory available
    ///     { block.scratch_pointer() } -> same_as<std::byte*>;
    ///     { block.template scratch_pointer<T>() } -> same_as<T*>;
    ///
    ///     Sets all bytes of the available scratch to zero and returns the scratch span.
    ///     If the block doesn't have any scratch memory available, this does nothing and returns an empty span.
    ///     { block.zeroed_scratch() } -> details::scratch_span;
    ///     { block.template zeroed_scratch<T, I>() } -> details::scratch_span<T, I>;
    ///
    ///     Waits for all threads in the block to reach this point.
    ///     After this point, memory writes are visible to other threads in the block.
    ///     { block.synchronize() } -> same_as<void>;
    ///
    ///     Atomically adds a value, guarantees no data-races between the threads of the block.
    ///     { block.atomic_add(value, accessor, indices...) }
    /// }
    ///
    /// \details Thread
    /// compute_handle.threads() -> {
    ///     /// Integer type used for indexing.
    ///     { T::index_type } -> integer;
    ///
    ///     Linear index of the thread within the block.
    ///     { thread.lid() } -> same_as<typename T::index_type>;
    ///     { thread.template lid<I>() } -> same_as<I>;
    ///
    ///     Returns the thread nd-indices within the block.
    ///     If the block has fewer dimensions than the requested indices, the extra indices are set to 0.
    ///     If the block has more dimensions than the requested indices, these higher dimensions are simply ignored.
    ///     { thread.template id<I, N>() } -> same_as<Vec<I, N>>;
    ///
    ///     Returns the thread global index, i.e., the index within the grid of threads.
    ///     This 1d-index is unique to each thread.
    ///     { thread.gid() } -> same_as<typename T::index_type>;
    ///     { thread.template gid<I>() } -> same_as<I>;
    /// }
    ///
    /// \warning
    /// Note that certain implementations (CUDA) may launch multi-grid kernels. At the moment, the compute handle
    /// doesn't support this except thread().gid() which correctly computes the global multi-grid index thereby
    /// ensuring a unique ID per-thread. The other member functions query the current grid. We could correct this
    /// by tracking the grid shape (we already track the offset of the block indices), but the usefullness/cost ratio
    /// seems quite low.
    template<typename T>
    concept compute_handle = proclaim_is_compute_handle<std::decay_t<T>>::value;

    template<typename T> concept compute_handle_cpu = compute_handle<T> and T::is_cpu();
    template<typename T> concept compute_handle_gpu = compute_handle<T> and T::is_gpu();
}

namespace noa::details {
    struct InitChecker {
        template<typename CH, typename Op>
        static consteval bool check_0() {
            return requires { std::declval<Op&>().init(std::declval<const CH&>()); };
        }

        template<typename Op>
        static consteval bool check_1() {
            return requires { std::declval<Op&>().init(); };
        }

        template<typename CH, typename Op>
        static consteval bool check_10() {
            return requires { std::declval<Op&>().deinit(std::declval<const CH&>()); };
        }

        template<typename Op>
        static consteval bool check_11() {
            return requires { std::declval<Op&>().deinit(); };
        }
    };

    struct InitInterface {
        template<nt::compute_handle CH, typename Op>
        static constexpr void init(const CH& ci, Op& op) {
            if constexpr (not nt::remove_compute_handle_v<Op> and InitChecker::check_0<CH, Op>()) {
                op.init(ci);
            } else if constexpr (InitChecker::check_1<Op>()) {
                op.init();
            } else {
                static_assert(
                    not nt::remove_default_init_v<Op>,
                    "Defaulted no-op .init() was removed using the remove_default_init type flag, but no explicit .init() was detected"
                );
            }
        }

        template<nt::compute_handle CH, typename Op>
        static constexpr void deinit(const CH& ci, Op& op) {
            if constexpr (not nt::remove_compute_handle_v<Op> and InitChecker::check_10<CH, Op>()) {
                op.deinit(ci);
            } else if constexpr (InitChecker::check_11<Op>()) {
                op.deinit();
            } else {
                static_assert(
                    not nt::remove_default_deinit_v<Op>,
                    "Defaulted no-op .deinit() was removed using the remove_default_deinit type flag, but no explicit .deinit() was detected"
                );
            }
        }
    };

    struct IwiseChecker{
        template<typename CH, typename Op, typename... I>
        static consteval bool check_10() {
            return requires { std::declval<Op&>()(std::declval<const CH&>(), std::declval<decltype(Vec{std::declval<I>()...})&>()); };
        }

        template<typename CH, typename Op, typename... I>
        static consteval bool check_11() {
            return requires { std::declval<Op&>()(std::declval<const CH&>(), std::declval<I>()...); };
        }

        template<typename Op, typename... I>
        static consteval bool check_12() {
            return requires { std::declval<Op&>()(std::declval<decltype(Vec{std::declval<I>()...})&>()); };
        }

        template<typename Op, typename... I>
        static consteval bool check_13() {
            return requires { std::declval<Op&>()(std::declval<I>()...); };
        }
    };

    struct IwiseInterface : InitInterface {
        template<nt::compute_handle CH, typename Op, nt::integer... Indices>
        static constexpr void call(const CH& ci, Op& op, Indices... indices) {
            if constexpr (not nt::remove_compute_handle_v<Op> and IwiseChecker::check_10<CH, Op, Indices...>()) {
                auto v = Vec{indices...};
                op(ci, v);
            } else if constexpr (not nt::remove_compute_handle_v<Op> and IwiseChecker::check_11<CH, Op, Indices...>()) {
                op(ci, indices...);
            } else if constexpr (IwiseChecker::check_12<Op, Indices...>()) {
                auto v = Vec{indices...};
                op(v);
            } else if constexpr (IwiseChecker::check_13<Op, Indices...>()) {
                op(indices...);
            } else {
                static_assert(nt::always_false<Op>, "op.operator(...) is not defined or is invalid");
            }
        }
    };

    struct EwiseChecker {
        template<typename CH, typename Op, typename... IO>
        static consteval bool check_0 () {
            return requires { std::declval<Op&>()(std::declval<const CH&>(), std::declval<IO&>()...); };
        }

        template<typename Op, typename... IO>
        static consteval bool check_1 () {
            return requires { std::declval<Op&>()(std::declval<IO&>()...); };
        }
    };

    template<bool ZipInput, bool ZipOutput>
    struct EwiseInterface : InitInterface {
        template<nt::compute_handle CH, typename Op, nt::tuple Input, nt::tuple Output, nt::integer... Indices>
        static constexpr void call(const CH& ci, Op& op, Input& input, Output& output, Indices... indices) {
            EwiseInterface::call_(ci, op, input, output, nt::index_list_t<Input>{}, nt::index_list_t<Output>{}, indices...);
        }

    private:
        template<typename CH, typename Op, typename Input, typename Output, usize... I, usize... O, typename... Indices>
        static constexpr void call_(const CH& ci, Op& op, Input& input, Output& output, std::index_sequence<I...>, std::index_sequence<O...>, Indices... indices) {
            // "input" and "output" are accessors, so we know that the operator() returns a lvalue reference.
            // forward_as_tuple will create a tuple of these lvalue references; there's nothing being taken by value.
            // Also, the zipped parameters should not be passed as rvalues.
            if constexpr (ZipInput and ZipOutput) {
                auto pi = noa::forward_as_tuple(input[Tag<I>{}](indices...)...);
                auto po = noa::forward_as_tuple(output[Tag<O>{}](indices...)...);
                if constexpr (not nt::remove_compute_handle_v<Op> and EwiseChecker::check_0<CH, Op, decltype(pi), decltype(po)>()) {
                    op(ci, pi, po);
                } else if constexpr (EwiseChecker::check_1<Op, decltype(pi), decltype(po)>()) {
                    op(pi, po);
                } else {
                    static_assert(nt::always_false<Op>, "op(...) is not defined or is invalid");
                }
            } else if constexpr (ZipInput) {
                auto pi = noa::forward_as_tuple(input[Tag<I>{}](indices...)...);
                if constexpr (not nt::remove_compute_handle_v<Op> and EwiseChecker::check_0<CH, Op, decltype(pi), decltype(output[Tag<O>{}](indices...))...>()) {
                    op(ci, pi, output[Tag<O>{}](indices...)...);
                } else if constexpr (EwiseChecker::check_1<Op, decltype(pi), decltype(output[Tag<O>{}](indices...))...>()) {
                    op(pi, output[Tag<O>{}](indices...)...);
                } else {
                    static_assert(nt::always_false<Op>, "op(...) is not defined or is invalid");
                }
            } else if constexpr (ZipOutput) {
                auto po = noa::forward_as_tuple(output[Tag<O>{}](indices...)...);
                if constexpr (not nt::remove_compute_handle_v<Op> and EwiseChecker::check_0<CH, Op, decltype(input[Tag<I>{}](indices...))..., decltype(po)>()) {
                    op(ci, input[Tag<I>{}](indices...)..., po);
                } else if constexpr (EwiseChecker::check_1<Op, decltype(input[Tag<I>{}](indices...))..., decltype(po)>()) {
                    op(input[Tag<I>{}](indices...)..., po);
                } else {
                    static_assert(nt::always_false<Op>, "op(...) is not defined or is invalid");
                }
            } else {
                if constexpr (not nt::remove_compute_handle_v<Op> and EwiseChecker::check_0<CH, Op, decltype(input[Tag<I>{}](indices...))..., decltype(output[Tag<O>{}](indices...))...>()) {
                    op(ci, input[Tag<I>{}](indices...)..., output[Tag<O>{}](indices...)...);
                } else if constexpr (EwiseChecker::check_1<Op, decltype(input[Tag<I>{}](indices...))..., decltype(output[Tag<O>{}](indices...))...>()) {
                    op(input[Tag<I>{}](indices...)..., output[Tag<O>{}](indices...)...);
                } else {
                    static_assert(nt::always_false<Op>, "op(...) is not defined or is invalid");
                }
            }
        }
    };

    struct ReduceInitChecker {
        template<typename CH, typename Op, typename... I>
        static consteval bool check_0() {
            return requires { std::declval<Op&>().init(std::declval<const CH&>(), std::declval<decltype(Vec{std::declval<I>()...})&>()); };
        }

        template<typename CH, typename Op, typename... I>
        static consteval bool check_1() {
            return requires { std::declval<Op&>().init(std::declval<const CH&>(), std::declval<I>()...); };
        }

        template<typename CH, typename Op>
        static consteval bool check_2() {
            return requires { std::declval<Op&>().init(std::declval<const CH&>()); };
        }

        template<typename Op, typename... I>
        static consteval bool check_3() {
            return requires { std::declval<Op&>().init(std::declval<decltype(Vec{std::declval<I>()...})&>()); };
        }

        template<typename Op, typename... I>
        static consteval bool check_4() {
            return requires { std::declval<Op&>().init(std::declval<I>()...); };
        }

        template<typename Op>
        static consteval bool check_5() {
            return requires { std::declval<Op&>().init(); };
        }

        template<typename CH, typename Op, typename... I>
        static consteval bool check_10() {
            return requires { std::declval<Op&>().deinit(std::declval<const CH&>(), std::declval<decltype(Vec{std::declval<I>()...})&>()); };
        }

        template<typename CH, typename Op, typename... I>
        static consteval bool check_11() {
            return requires { std::declval<Op&>().deinit(std::declval<const CH&>(), std::declval<I>()...); };
        }

        template<typename CH, typename Op>
        static consteval bool check_12() {
            return requires { std::declval<Op&>().deinit(std::declval<const CH&>()); };
        }

        template<typename Op, typename... I>
        static consteval bool check_13() {
            return requires { std::declval<Op&>().deinit(std::declval<decltype(Vec{std::declval<I>()...})&>()); };
        }

        template<typename Op, typename... I>
        static consteval bool check_14() {
            return requires { std::declval<Op&>().deinit(std::declval<I>()...); };
        }

        template<typename Op>
        static consteval bool check_15() {
            return requires { std::declval<Op&>().deinit(); };
        }
    };

    struct ReduceInitInterface {
        template<nt::compute_handle CH, typename Op, typename... Indices>
        static constexpr void init(const CH& ci, Op& op, Indices... indices) {
            if constexpr (not nt::remove_compute_handle_v<Op> and ReduceInitChecker::check_0<CH, Op, Indices...>()) {
                auto v = Vec{indices...};
                op.init(ci, v);
            } else if constexpr (not nt::remove_compute_handle_v<Op> and ReduceInitChecker::check_1<CH, Op, Indices...>()) {
                op.init(ci, indices...);
            } else if constexpr (not nt::remove_compute_handle_v<Op> and ReduceInitChecker::check_2<CH, Op>()) {
                op.init(ci);
            } else if constexpr (ReduceInitChecker::check_3<Op, Indices...>()) {
                auto v = Vec{indices...};
                op.init(v);
            } else if constexpr (ReduceInitChecker::check_4<Op, Indices...>()) {
                op.init(indices...);
            } else if constexpr (ReduceInitChecker::check_5<Op>()) {
                op.init();
            } else {
                static_assert(
                    not nt::remove_default_init_v<Op>,
                    "Defaulted no-op .init() was removed using the remove_default_init type flag, but no explicit .init() was detected"
                );
            }
        }

        template<nt::compute_handle CH, typename Op, typename... Indices>
        static constexpr void deinit(const CH& ci, Op& op, Indices... indices) {
            if constexpr (not nt::remove_compute_handle_v<Op> and ReduceInitChecker::check_10<CH, Op, Indices...>()) {
                auto v = Vec{indices...};
                op.deinit(ci, v);
            } else if constexpr (not nt::remove_compute_handle_v<Op> and ReduceInitChecker::check_11<CH, Op, Indices...>()) {
                op.deinit(ci, indices...);
            } else if constexpr (not nt::remove_compute_handle_v<Op> and ReduceInitChecker::check_12<CH, Op>()) {
                op.deinit(ci);
            } else if constexpr (ReduceInitChecker::check_13<Op, Indices...>()) {
                auto v = Vec{indices...};
                op.deinit(v);
            } else if constexpr (ReduceInitChecker::check_14<Op, Indices...>()) {
                op.deinit(indices...);
            } else if constexpr (ReduceInitChecker::check_15<Op>()) {
                op.deinit();
            } else {
                static_assert(
                    not nt::remove_default_deinit_v<Op>,
                    "Defaulted no-op .deinit() was removed using the remove_default_deinit type flag, but no explicit .deinit() was detected"
                );
            }
        }
    };

    struct ReduceJoinChecker {
        template<typename Op, typename... IO>
        static consteval bool check_0() {
            return requires { std::declval<Op&>().join(std::declval<IO&>()...); };
        }
        template<typename Op, typename... IO>
        static consteval bool check_1() {
            return requires { std::declval<Op&>()(std::declval<IO&>()...); };
        }
    };

    template<bool ZipReduced, bool AllowCall>
    struct ReduceJoinInterface {
        template<typename Op, nt::tuple_of_accessor_value_or_empty Reduced>
        static constexpr void join(Op& op, Reduced& to_reduce, Reduced& reduced) {
            join_(op, to_reduce, reduced, nt::index_list_t<Reduced>{});
        }

    private:
        template<typename Op, typename Reduced, usize... R>
        static constexpr void join_(Op& op, Reduced& to_reduce, Reduced& reduced, std::index_sequence<R...>) {
            if constexpr (ZipReduced) {
                auto pi = noa::forward_as_tuple(to_reduce[Tag<R>{}].ref()...);
                auto po = noa::forward_as_tuple(reduced[Tag<R>{}].ref()...);
                if constexpr (not nt::remove_compute_handle_v<Op> and ReduceJoinChecker::check_0<Op, decltype(pi), decltype(po)>()) {
                    op.join(pi, po);
                } else if constexpr (AllowCall and ReduceJoinChecker::check_1<Op, decltype(pi), decltype(po)>()) {
                    op(pi, po);
                } else {
                    static_assert(nt::empty_tuple<Reduced>, "op.join(...) is not defined or is invalid");
                }
            } else {
                if constexpr (not nt::remove_compute_handle_v<Op> and ReduceJoinChecker::check_0<Op, decltype(to_reduce[Tag<R>{}].ref())..., decltype(reduced[Tag<R>{}].ref())...>()) {
                    op.join(to_reduce[Tag<R>{}].ref()..., reduced[Tag<R>{}].ref()...);
                } else if constexpr (AllowCall and ReduceJoinChecker::check_1<Op, decltype(to_reduce[Tag<R>{}].ref())..., decltype(reduced[Tag<R>{}].ref())...>()) {
                    op(to_reduce[Tag<R>{}].ref()..., reduced[Tag<R>{}].ref()...);
                } else {
                    static_assert(nt::empty_tuple<Reduced>, "op.join(...) is not defined or is invalid");
                }
            }
        }
    };

    struct ReducePostChecker {
        template<typename Op, typename... IO>
        static consteval bool check_0 () {
            return requires { std::declval<Op&>().post(std::declval<IO&>()...); };
        }
    };

    template<bool ZipReduced, bool ZipOutput>
    struct ReducePostInterface {
        template<typename Op, nt::tuple_of_accessor_value_or_empty Reduced, nt::tuple_of_accessor_or_empty Output, nt::integer... Indices>
        static constexpr void post(Op& op, Reduced& reduced, Output& output, Indices... indices) {
            post_(op, reduced, output, nt::index_list_t<Reduced>{}, nt::index_list_t<Output>{}, nt::type_list_t<Output>{}, indices...);
        }

    private:
        template<typename Op, typename Reduced, typename Output, usize... R, usize... O, typename... T, typename... Indices>
        static constexpr void post_(
            Op& op, Reduced& reduced, Output& output,
            std::index_sequence<R...>, std::index_sequence<O...>, nt::TypeList<T...>,
            Indices... indices
        ) {
            using packed_indices_t = Vec<nt::first_t<Indices...>, sizeof...(Indices)>;
            if constexpr (ZipReduced and ZipOutput) {
                auto pr = noa::forward_as_tuple(reduced[Tag<R>{}].ref()...);
                auto po = noa::forward_as_tuple(output[Tag<O>{}](indices...)...);
                if constexpr (ReducePostChecker::check_0<Op, decltype(pr), decltype(po), packed_indices_t>()) {
                    auto packed = Vec{indices...};
                    op.post(pr, po, packed);
                } else if constexpr (ReducePostChecker::check_0<Op, decltype(pr), decltype(po), Indices...>()) {
                    op.post(pr, po, indices...);
                } else if constexpr (ReducePostChecker::check_0<Op, decltype(pr), decltype(po)>()) {
                    op.post(pr, po);
                } else if constexpr (nt::remove_default_post_v<Op>) {
                    static_assert(nt::always_false<Op>, "Defaulted copy of op.post(...) was removed using the remove_default_post type flag, but no explicit op.post() was detected");
                } else if constexpr (not nt::empty_tuple<Output> and not nt::empty_tuple<Reduced>) {
                    ((output[Tag<O>{}](indices...) = static_cast<T::mutable_value_type>(reduced[Tag<R>{}].ref())), ...);
                }
            } else if constexpr (ZipReduced) {
                auto pr = noa::forward_as_tuple(reduced[Tag<R>{}].ref()...);
                if constexpr (ReducePostChecker::check_0<Op, decltype(pr), decltype(output[Tag<O>{}](indices...))..., packed_indices_t>()) {
                    auto packed = Vec{indices...};
                    op.post(pr, output[Tag<O>{}](indices...)..., packed);
                } else if constexpr (ReducePostChecker::check_0<Op, decltype(pr), decltype(output[Tag<O>{}](indices...))..., Indices...>()) {
                    op.post(pr, output[Tag<O>{}](indices...)..., indices...);
                } else if constexpr (ReducePostChecker::check_0<Op, decltype(pr), decltype(output[Tag<O>{}](indices...))...>()) {
                    op.post(pr, output[Tag<O>{}](indices...)...);
                } else if constexpr (nt::remove_default_post_v<Op>) {
                    static_assert(nt::always_false<Op>, "Defaulted copy of op.post(...) was removed using the remove_default_post type flag, but no explicit op.post() was detected");
                } else if constexpr (not nt::empty_tuple<Output> and not nt::empty_tuple<Reduced>) {
                    ((output[Tag<O>{}](indices...) = static_cast<T::mutable_value_type>(reduced[Tag<R>{}].ref())), ...);
                }
            } else if constexpr (ZipOutput) {
                auto po = noa::forward_as_tuple(output[Tag<O>{}](indices...)...);
                if constexpr (ReducePostChecker::check_0<Op, decltype(reduced[Tag<R>{}].ref())..., decltype(po), packed_indices_t>()) {
                    auto packed = Vec{indices...};
                    op.post(reduced[Tag<R>{}].ref()..., po, packed);
                } else if constexpr (ReducePostChecker::check_0<Op, decltype(reduced[Tag<R>{}].ref())..., decltype(po), Indices...>()) {
                    op.post(reduced[Tag<R>{}].ref()..., po, indices...);
                } else if constexpr (ReducePostChecker::check_0<Op, decltype(reduced[Tag<R>{}].ref())..., decltype(po)>()) {
                    op.post(reduced[Tag<R>{}].ref()..., po);
                } else if constexpr (nt::remove_default_post_v<Op>) {
                    static_assert(nt::always_false<Op>, "Defaulted copy of op.post(...) was removed using the remove_default_post type flag, but no explicit op.post() was detected");
                } else if constexpr (not nt::empty_tuple<Output> and not nt::empty_tuple<Reduced>) {
                    ((output[Tag<O>{}](indices...) = static_cast<T::mutable_value_type>(reduced[Tag<R>{}].ref())), ...);
                }
            } else {
                if constexpr (ReducePostChecker::check_0<Op, decltype(reduced[Tag<R>{}].ref())..., decltype(output[Tag<O>{}](indices...))..., packed_indices_t>()) {
                    auto packed = Vec{indices...};
                    op.post(reduced[Tag<R>{}].ref()..., output[Tag<O>{}](indices...)..., packed);
                } else if constexpr (ReducePostChecker::check_0<Op, decltype(reduced[Tag<R>{}].ref())..., decltype(output[Tag<O>{}](indices...))..., Indices...>()) {
                    op.post(reduced[Tag<R>{}].ref()..., output[Tag<O>{}](indices...)..., indices...);
                } else if constexpr (ReducePostChecker::check_0<Op, decltype(reduced[Tag<R>{}].ref())..., decltype(output[Tag<O>{}](indices...))...>()) {
                    op.post(reduced[Tag<R>{}].ref()..., output[Tag<O>{}](indices...)...);
                } else if constexpr (nt::remove_default_post_v<Op>) {
                    static_assert(nt::always_false<Op>, "Defaulted copy of op.post(...) was removed using the remove_default_post type flag, but no explicit op.post() was detected");
                } else if constexpr (not nt::empty_tuple<Output> and not nt::empty_tuple<Reduced>) {
                    ((output[Tag<O>{}](indices...) = static_cast<T::mutable_value_type>(reduced[Tag<R>{}].ref())), ...);
                }
            }
        }
    };

    struct ReduceCallChecker {
        template<typename CH, typename Op, typename... IO>
        static consteval bool check_0() {
            return requires { std::declval<Op&>()(std::declval<const CH&>(), std::declval<IO&>()...); };
        }

        template<typename Op, typename... IO>
        static consteval bool check_1() {
            return requires { std::declval<Op&>()(std::declval<IO&>()...); };
        }
    };

    template<bool ZipReduced, bool ZipOutput>
    struct ReduceIwiseInterface : ReduceInitInterface, ReduceJoinInterface<ZipReduced, false>, ReducePostInterface<ZipReduced, ZipOutput> {
        template<nt::compute_handle CH, typename Op, nt::tuple_of_accessor_value_or_empty Reduced, nt::integer... Indices>
        static constexpr void call(const CH& ci, Op& op, Reduced& reduced, Indices... indices) {
            call_(ci, op, reduced, nt::index_list_t<Reduced>{}, indices...);
        }

    private:
        template<typename CH, typename Op, typename Reduced, usize... R, typename... Indices>
        static constexpr void call_(const CH& ci, Op& op, Reduced& reduced, std::index_sequence<R...>, Indices... indices) {
            using packed_indices_t = Vec<nt::first_t<Indices...>, sizeof...(indices)>;
            if constexpr (ZipReduced) {
                auto pr = noa::forward_as_tuple(reduced[Tag<R>{}].ref()...);
                if constexpr (ReduceCallChecker::check_0<CH, Op, packed_indices_t, decltype(pr)>()) {
                    auto packed = packed_indices_t{indices...};
                    op(ci, packed, pr);
                } else if constexpr (ReduceCallChecker::check_0<CH, Op, Indices..., decltype(pr)>()) {
                    op(ci, indices..., pr);
                } else if constexpr (ReduceCallChecker::check_1<Op, packed_indices_t, decltype(pr)>()) {
                    auto packed = packed_indices_t{indices...};
                    op(packed, pr);
                } else if constexpr (ReduceCallChecker::check_1<Op, Indices..., decltype(pr)>()) {
                    op(indices..., pr);
                } else {
                    static_assert(nt::always_false<Op>, "op(...) is not defined or is invalid");
                }
            } else {
                if constexpr (ReduceCallChecker::check_0<CH, Op, packed_indices_t, decltype(reduced[Tag<R>{}].ref())...>()) {
                    auto packed = packed_indices_t{indices...};
                    op(ci, packed, reduced[Tag<R>{}].ref()...);
                } else if constexpr (ReduceCallChecker::check_0<CH, Op, Indices..., decltype(reduced[Tag<R>{}].ref())...>()) {
                    op(ci, indices..., reduced[Tag<R>{}].ref()...);
                } else if constexpr (ReduceCallChecker::check_1<Op, packed_indices_t, decltype(reduced[Tag<R>{}].ref())...>()) {
                    auto packed = packed_indices_t{indices...};
                    op(packed, reduced[Tag<R>{}].ref()...);
                } else if constexpr (ReduceCallChecker::check_1<Op, Indices..., decltype(reduced[Tag<R>{}].ref())...>()) {
                    op(indices..., reduced[Tag<R>{}].ref()...);
                } else {
                    static_assert(nt::always_false<Op>, "op(...) is not defined or is invalid");
                }
            }
        }
    };

    template<bool ZipInput, bool ZipReduced, bool ZipOutput>
    struct ReduceEwiseInterface : ReduceInitInterface, ReduceJoinInterface<ZipReduced, true>, ReducePostInterface<ZipReduced, ZipOutput> {
        template<nt::compute_handle CH, typename Op, nt::tuple_of_accessor Input, nt::tuple_of_accessor_value_or_empty Reduced, nt::integer... Indices>
        static constexpr void call(const CH& ci, Op& op, Input& input, Reduced& reduced, Indices... indices) {
            call_(ci, op, input, reduced, nt::index_list_t<Input>{}, nt::index_list_t<Reduced>{}, indices...);
        }

    private:
        template<typename CH, typename Op, typename Input, typename Reduced, usize... I, usize... R, typename... Indices>
        static constexpr void call_(
            const CH& ci, Op& op, Input& input, Reduced& reduced,
            std::index_sequence<I...>, std::index_sequence<R...>,
            Indices... indices
        ) {
            if constexpr (ZipInput and ZipReduced) {
                auto pi = noa::forward_as_tuple(input[Tag<I>{}](indices...)...);
                auto pr = noa::forward_as_tuple(reduced[Tag<R>{}].ref()...);
                if constexpr (not nt::remove_compute_handle_v<Op> and ReduceCallChecker::check_0<CH, Op, decltype(pi), decltype(pr)>()) {
                    op(ci, pi, pr);
                } else if constexpr (ReduceCallChecker::check_1<Op, decltype(pi), decltype(pr)>()) {
                    op(pi, pr);
                } else {
                    static_assert(nt::always_false<Op>, "op(...) is not defined or is invalid");
                }
            } else if constexpr (ZipInput) {
                auto pi = noa::forward_as_tuple(input[Tag<I>{}](indices...)...);
                if constexpr (not nt::remove_compute_handle_v<Op> and ReduceCallChecker::check_0<CH, Op, decltype(pi), decltype(reduced[Tag<R>{}].ref())...>()) {
                    op(ci, pi, reduced[Tag<R>{}].ref()...);
                } else if constexpr (ReduceCallChecker::check_1<Op, decltype(pi), decltype(reduced[Tag<R>{}].ref())...>()) {
                    op(pi, reduced[Tag<R>{}].ref()...);
                } else {
                    static_assert(nt::always_false<Op>, "op(...) is not defined or is invalid");
                }
            } else if constexpr (ZipReduced) {
                auto pr = noa::forward_as_tuple(reduced[Tag<R>{}].ref()...);
                if constexpr (not nt::remove_compute_handle_v<Op> and ReduceCallChecker::check_0<CH, Op, decltype(input[Tag<I>{}](indices...))..., decltype(pr)>()) {
                    op(ci, input[Tag<I>{}](indices...)..., pr);
                } else if constexpr (ReduceCallChecker::check_1<Op, decltype(input[Tag<I>{}](indices...))..., decltype(pr)>()) {
                    op(input[Tag<I>{}](indices...)..., pr);
                } else {
                    static_assert(nt::always_false<Op>, "op(...) is not defined or is invalid");
                }
            } else {
                if constexpr (not nt::remove_compute_handle_v<Op> and ReduceCallChecker::check_0<CH, Op, decltype(input[Tag<I>{}](indices...))..., decltype(reduced[Tag<R>{}].ref())...>()) {
                    op(ci, input[Tag<I>{}](indices...)..., reduced[Tag<R>{}].ref()...);
                } else if constexpr (ReduceCallChecker::check_1<Op, decltype(input[Tag<I>{}](indices...))..., decltype(reduced[Tag<R>{}].ref())...>()) {
                    op(input[Tag<I>{}](indices...)..., reduced[Tag<R>{}].ref()...);
                } else {
                    static_assert(nt::always_false<Op>, "op(...) is not defined or is invalid");
                }
            }
        }
    };
}

#ifdef __CUDACC__
#   if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
#       pragma GCC diagnostic pop
#   elif defined(NOA_COMPILER_MSVC)
#       pragma warning(pop)
#   endif
#endif
