#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"

#if defined(NOA_IS_OFFLINE)
#include <cstddef>
#include <utility>
#include <functional> // reference_wrapper
#else
#include <cuda/std/cstddef>
#include <cuda/std/utility>
#include <cuda/std/functional> // reference_wrapper
#endif

/// Original implementation: https://github.com/codeinred/tuplet
/// - Type names, namespaces and code-format are changed, just to have this code fully integrated into noa.
/// - Comparisons, concepts, and MSVC-support are removed to simplify the code (because we don't need that).
/// - tuple_base_t implementation was changed to support nvrtc (weird bug?).

namespace noa {
    template<typename... T>
    struct TypeList {};

    template<typename... Ls, typename... Rs>
    inline constexpr auto operator+(TypeList<Ls...>, TypeList<Rs...>) {
        return TypeList<Ls..., Rs...>{};
    }
}

namespace noa::traits {
    template<class T>
    struct is_reference_wrapper : std::false_type {};
    template<class U>
    struct is_reference_wrapper<std::reference_wrapper<U>> : std::true_type {};

    template<typename T>
    struct unwrap_reference { using type = T; };

    template<typename U>
    struct unwrap_reference<std::reference_wrapper<U>> { using type = U&; };

    template<typename T>
    struct unwrap_ref_decay : unwrap_reference<std::decay_t<T>> {};

    template<typename T>
    using unwrap_ref_decay_t = typename unwrap_ref_decay<T>::type;

    template<typename Tup>
    using base_list_t = typename std::decay_t<Tup>::base_list;

    template<typename Tup>
    using element_list_t = typename std::decay_t<Tup>::element_list;

    template<class Tup, class = typename Tup::base_list>
    constexpr bool has_base_list(int) {
        return true;
    }

    template<class Tup>
    constexpr bool has_base_list(long long) {
        return false;
    }

    template <class Tuple>
    constexpr auto has_base_list_v = has_base_list<Tuple>(0);
}

namespace noa::guts {
    template<size_t I>
    using tag = std::integral_constant<size_t, I>;

    template<size_t I, typename T>
    struct TupleElement {
        using type = T;
        NOA_NO_UNIQUE_ADDRESS T value;

        inline constexpr decltype(auto) operator[](tag<I>) & {
            return (value);
        }
        inline constexpr decltype(auto) operator[](tag<I>) const& {
            return (value);
        }
        inline constexpr decltype(auto) operator[](tag<I>) && {
            return (static_cast<TupleElement&&>(*this).value);
        }

        // Used to extract the type for the Ith element.
        static auto declval(tag<I>) -> T;
    };

    template<typename... Bases>
    struct TypeMap : Bases... {
        using base_list = TypeList<Bases...>;
        using Bases::operator[]...;
        using Bases::declval...;
    };

    // The original implementation from "codeinred" uses a template specialization to extract the index sequence.
    // This should be completely fine, but somehow nvrtc doesn't like it and doesn't seem to see the specialization,
    // resulting in an "incomplete type is not allowed" error.
    //
    //    template <class IndexSequence, class... T>
    //    struct _get_tuple_base;
    //
    //    template <size_t... I, class... T>
    //    struct _get_tuple_base<std::index_sequence<I...>, T...> {
    //        using type = TypeMap<TupleElement<I, T>...>;
    //    };
    //
    // Instead, use a variadic template function (no specialization involved) and retrieve the return type.
    // Actually, wrap this in a non-template type with a template operator() to easily retrieve the return type
    // while using type deduction. Note that there's no need to define this function; the declaration is all we need.
    struct GetTupleType {
        template<size_t... I, class... T>
        auto operator()(std::index_sequence<I...>, T...) -> TypeMap<TupleElement<I, T>...>;
    };
}

namespace noa {
    template<typename... T>
    using TupleBase = typename std::invoke_result_t<guts::GetTupleType, std::make_index_sequence<sizeof...(T)>, T...>;

    template<class... T>
    struct Tuple;

    template<typename Tup, typename F, typename... B>
    inline constexpr void tuple_for_each(Tup&& tup, F&& func, TypeList<B...>) {
        (void(func(tup.::noa::traits::identity_t<B>::value)), ...);
    }

    template<typename Tup, typename F, typename... B>
    inline constexpr bool tuple_any(Tup&& tup, F&& func, TypeList<B...>) {
        return (bool(func(tup.::noa::traits::identity_t<B>::value)) || ...);
    }

    template<typename Tup, typename F, typename... B>
    inline constexpr bool tuple_all(Tup&& tup, F&& func, TypeList<B...>) {
        return (bool(func(tup.::noa::traits::identity_t<B>::value)) && ...);
    }

    template<typename Tup, typename F, typename... B>
    inline constexpr auto tuple_map(Tup&& tup, F&& func, TypeList<B...>)
    -> Tuple<decltype(func(tup.::noa::traits::identity_t<B>::value))...> {
        return {func(tup.::noa::traits::identity_t<B>::value)...};
    }

    template<typename Tup, typename F, typename... B>
    inline constexpr decltype(auto) tuple_apply(Tup&& t, F&& f, TypeList<B...>) {
        return static_cast<F&&>(f)(t.::noa::traits::identity_t<B>::value...);
    }

    template<typename U, typename Tup, typename... B>
    inline constexpr U tuple_convert(Tup&& t, TypeList<B...>) {
        return U{t.::noa::traits::identity_t<B>::value...};
    }

    template<class... T>
    struct Tuple : TupleBase<T...> {
        constexpr static size_t N = sizeof...(T);
        constexpr static bool nothrow_swappable = (std::is_nothrow_swappable_v<T> && ...);

        using super = TupleBase<T...>;
        using super::operator[];
        using base_list = typename super::base_list;
        using element_list = TypeList<T...>;
        using super::declval;

        template<size_t I>
        inline constexpr decltype(auto) get() & {
            return ((*this)[guts::tag<I>{}]);
        }

        template<size_t I>
        inline constexpr decltype(auto) get() const& {
            return ((*this)[guts::tag<I>{}]);
        }

        template<size_t I>
        inline constexpr decltype(auto) get() && {
            return (static_cast<Tuple&&>(*this)[guts::tag<I>{}]);
        }

        template<typename U, typename = std::enable_if_t<!std::is_same_v<std::decay_t<Tuple>, std::decay_t<U>>>> // Preserves default assignments
        inline constexpr auto& operator=(U&& tup) {
            using tuple_t = std::decay_t<U>;
            if constexpr (nt::has_base_list_v<tuple_t>) {
                assign_tup_(static_cast<U&&>(tup), base_list{}, typename tuple_t::base_list{});
            } else {
                assign_index_tup_(static_cast<U&&>(tup), std::make_index_sequence<N>());
            }
            return *this;
        }

        template<class... U>
        constexpr auto& assign(U&&... values) {
            _assign(base_list {}, static_cast<U&&>(values)...);
            return *this;
        }

        inline constexpr void swap(Tuple& other) noexcept(nothrow_swappable) {
            _swap(other, base_list {});
        }

        /// Applies a function to every element of the tuple. The order is the
        /// declaration order, so first the function will be applied to element
        /// 0, then element 1, then element 2, and so on, where element N is
        /// identified by get<N>
        template<class F>
        inline constexpr void for_each(F&& func)& {
            tuple_for_each(*this, static_cast<F&&>(func), base_list{});
        }
        template<class F>
        inline constexpr void for_each(F&& func) const& {
            tuple_for_each(*this, static_cast<F&&>(func), base_list{});
        }
        template<class F>
        inline constexpr void for_each(F&& func)&& {
            tuple_for_each(static_cast<Tuple&&>(*this), static_cast<F&&>(func), base_list{});
        }

        /// Applies a function to each element successively, until one returns true.
        /// Returns true if any application returned true, otherwise returns false.
        template<typename F>
        inline constexpr bool any(F&& func)& {
            return tuple_any(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr bool any(F&& func) const& {
            return tuple_any(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr bool any(F&& func)&& {
            return tuple_any(static_cast<Tuple&&>(*this), static_cast<F&&>(func), base_list{});
        }

        /// Applies a function to each element successively, until one returns false.
        /// Returns true if every application returned true, otherwise returns false.
        template<typename F>
        inline constexpr bool all(F&& func)& {
            return tuple_all(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr bool all(F&& func) const& {
            return tuple_all(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr bool all(F&& func)&& {
            return tuple_all(static_cast<Tuple&&>(*this), static_cast<F&&>(func), base_list{});
        }

        /// Map a function over every element in the tuple, using the returned values to construct a new tuple.
        template<typename F>
        inline constexpr auto map(F&& func)& {
            return tuple_map(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr auto map(F&& func) const& {
            return tuple_map(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr auto map(F&& func)&& {
            return tuple_map(static_cast<Tuple&&>(*this), static_cast<F&&>(func), base_list{});
        }

        template<typename F>
        inline constexpr decltype(auto) apply(F&& func)& {
            return tuple_apply(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr decltype(auto) apply(F&& func) const& {
            return tuple_apply(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr decltype(auto) apply(F&& func)&& {
            return tuple_apply(static_cast<Tuple&&>(*this), static_cast<F&&>(func), base_list{});
        }

        template<typename... U>
        constexpr explicit operator Tuple<U...>()& {
            static_assert(sizeof...(U) == N, "Can only convert to tuples with the same number of items");
            return tuple_convert<Tuple<U...>>(*this, base_list{});
        }
        template<typename... U>
        constexpr explicit operator Tuple<U...>() const& {
            static_assert(sizeof...(U) == N, "Can only convert to tuples with the same number of items");
            return tuple_convert<Tuple<U...>>(*this, base_list{});
        }
        template<typename... U>
        constexpr explicit operator Tuple<U...>()&& {
            static_assert(sizeof...(U) == N, "Can only convert to tuples with the same number of items");
            return tuple_convert<Tuple<U...>>(static_cast<Tuple&&>(*this), base_list{});
        }

        /// Instantiate the given type using list initialization
        template<typename U>
        inline constexpr U as()& {
            return tuple_convert<U>(*this, base_list{});
        }
        template<typename U>
        inline constexpr U as() const& {
            return tuple_convert<U>(*this, base_list{});
        }
        template<typename U>
        inline constexpr U as()&& {
            return tuple_convert<U>(static_cast<Tuple&&>(*this), base_list{});
        }

    private:
        template<class... B>
        inline constexpr void _swap(Tuple& other, TypeList<B...>) noexcept(nothrow_swappable) {
            using std::swap;
            (swap(B::value, other.::noa::traits::identity_t<B>::value), ...);
        }

        template <class U, class... B1, class... B2>
        inline constexpr void assign_tup_(U&& u,TypeList<B1...>,TypeList<B2...>) {
            (void(B1::value = static_cast<U&&>(u).::noa::traits::identity_t<B2>::value), ...);
        }
        template<class U, size_t... I>
        inline constexpr void assign_index_tup_(U&& u, std::index_sequence<I...>) {
            using std::get;
            (void(guts::TupleElement<I, T>::value = get<I>(static_cast<U&&>(u))), ...);
        }
        template <class... U, class... B>
        inline constexpr void _assign(TypeList<B...>, U&&... u) {
            (void(B::value = static_cast<U&&>(u)), ...);
        }
    };

    template<>
    struct Tuple<> : TupleBase<> {
        constexpr static size_t N = 0;
        constexpr static bool nothrow_swappable = true;
        using super = TupleBase<>;
        using base_list = TypeList<>;
        using element_list = TypeList<>;

        template<size_t>
        inline constexpr void get() {}

        template<typename U, typename = std::enable_if_t<!std::is_same_v<std::decay_t<Tuple>, std::decay_t<U>>>>
        constexpr auto& operator=(U&&) noexcept {
            return *this;
        }

        constexpr void swap(Tuple) noexcept {}
        constexpr auto& assign() noexcept { return *this; }

        template <class F>
        constexpr void for_each(F&&) const noexcept {}

        template <class F>
        constexpr bool any(F&&) const noexcept { return false; }

        template <class F>
        constexpr bool all(F&&) const noexcept { return true; }

        template <class F>
        constexpr auto map(F&&) const noexcept { return Tuple{}; }

        template <class F>
        constexpr decltype(auto) apply(F&& func) const noexcept { return func(); }

        template <class U>
        constexpr U as() const noexcept { return U {}; }
    };

    // Deduction guide.
    template <class... Ts>
    Tuple(Ts...) -> Tuple<nt::unwrap_ref_decay_t<Ts>...>;
}

namespace noa {
    template<typename First, typename Second>
    struct Pair {
        constexpr static size_t N = 2;
        constexpr static bool nothrow_swappable =
                std::is_nothrow_swappable_v<First>
                && std::is_nothrow_swappable_v<Second>;

        NOA_NO_UNIQUE_ADDRESS First first;
        NOA_NO_UNIQUE_ADDRESS Second second;

        template<size_t I>
        NOA_HD constexpr decltype(auto) get()& {
            if constexpr (I == 0)
                return (first);
            else if constexpr (I == 1)
                return (second);
            else
                static_assert(I < 2);
        }

        template<size_t I>
        NOA_HD constexpr decltype(auto) get() const& {
            if constexpr (I == 0)
                return (first);
            else if constexpr (I == 1)
                return (second);
            else
                static_assert(I < 2);
        }

        template<size_t I>
        NOA_HD constexpr decltype(auto) get()&& {
            if constexpr (I == 0)
                return (static_cast<Pair&&>(*this).first);
            else if constexpr (I == 1)
                return (static_cast<Pair&&>(*this).second);
            else
                static_assert(I < 2);
        }

        NOA_HD void swap(Pair& other) noexcept(nothrow_swappable) {
            using std::swap;
            swap(first, other.first);
            swap(second, other.second);
        }

        template<typename T, typename = std::enable_if_t<!std::is_same_v<std::decay_t<Pair>, std::decay_t<T>>>>
        NOA_HD constexpr auto& operator=(T&& tup) {
            auto&& [a, b] = static_cast<T&&>(tup);
            first = static_cast<decltype(a)&&>(a);
            second = static_cast<decltype(b)&&>(b);
            return *this;
        }

        template<typename F2, typename S2>
        NOA_HD constexpr auto& assign(F2&& f, S2&& s) {
            first = static_cast<F2&&>(f);
            second = static_cast<S2&&>(s);
            return *this;
        }
    };

    template<typename A, typename B>
    Pair(A, B) -> Pair<nt::unwrap_ref_decay_t<A>, nt::unwrap_ref_decay_t<B>>;
}

namespace noa {
    /// Converts from one tuple type to any other tuple or U
    template<typename Tuple>
    struct Convert {
        using base_list = typename std::decay_t<Tuple>::base_list;
        Tuple tuple;

        template<typename U>
        constexpr operator U()&& {
            return tuple_convert<U>(static_cast<Tuple&&>(tuple), base_list{});
        }
    };

    template<typename Tuple>
    Convert(Tuple&) -> Convert<Tuple&>;
    template<typename Tuple>
    Convert(Tuple const&) -> Convert<Tuple const&>;
    template<typename Tuple>
    Convert(Tuple&&) -> Convert<Tuple>;
}

namespace noa {
    template<size_t I, typename Tup>
    inline constexpr decltype(auto) get(Tup&& tup) {
        return static_cast<Tup&&>(tup).template get<I>();
    }

    template<typename... T>
    inline constexpr Tuple<T& ...> tie(T& ... t) { return {t...}; }

    template<typename F, typename Tup>
    inline constexpr decltype(auto) apply(F&& func, Tup&& tup) {
        return tuple_apply(static_cast<Tup&&>(tup), static_cast<F&&>(func), typename std::decay_t<Tup>::base_list{});
    }
    template<typename F, typename A, typename B>
    inline constexpr decltype(auto) apply(F&& func, Pair<A, B>& pair) {
        return static_cast<F&&>(func)(pair.first, pair.second);
    }
    template<typename F, typename A, typename B>
    inline constexpr decltype(auto) apply(F&& func, Pair<A, B> const& pair) {
        return static_cast<F&&>(func)(pair.first, pair.second);
    }
    template<typename F, typename A, typename B>
    inline constexpr decltype(auto) apply(F&& func, Pair<A, B>&& pair) {
        using P = Pair<A, B>&&;
        return static_cast<F&&>(func)(static_cast<P>(pair).first, static_cast<P>(pair).second);
    }

    template<typename... T>
    inline void swap(Tuple<T...>& a, Tuple<T...>& b) noexcept(Tuple<T...>::nothrow_swappable) {
        a.swap(b);
    }

    template<typename A, typename B>
    inline void swap(Pair<A, B>& a, Pair<A, B>& b) noexcept(Pair<A, B>::nothrow_swappable) {
        a.swap(b);
    }

    template<typename... Ts>
    inline constexpr auto make_tuple(Ts&& ... args) {
        return Tuple<nt::unwrap_ref_decay_t<Ts>...>{static_cast<Ts&&>(args)...};
    }

    template<typename... T>
    inline constexpr auto forward_as_tuple(T&& ... a) noexcept {
        return Tuple<T&& ...>{static_cast<T&&>(a)...};
    }
}

namespace noa {
    namespace guts {
        template<typename T, typename... Q>
        inline constexpr auto tuple_repeat_type(TypeList<Q...>) {
            return TypeList<nt::first_t<T, Q>...>{};
        }
        template<class... Outer>
        inline constexpr auto tuple_get_outer_bases(TypeList<Outer...>) {
            return (tuple_repeat_type<Outer>(nt::base_list_t<nt::type_t<Outer>>{}) + ...);
        }
        template<class... Outer>
        inline constexpr auto tuple_get_inner_bases(TypeList<Outer...>) {
            return (nt::base_list_t<nt::type_t<Outer>>{} + ...);
        }

        // This takes a forwarding tuple as a parameter. The forwarding tuple only
        // contains references, so it should just be taken by value.
        template<class T, class... Outer, class... Inner>
        inline constexpr auto tuple_cat(
                T tup,
                TypeList<Outer...>,
                TypeList<Inner...>
        ) -> Tuple<nt::type_t<Inner>...> {
            return {static_cast<nt::type_t<Outer>&&>(
                            tup.::noa::traits::identity_t<Outer>::value
                    ).::noa::traits::identity_t<Inner>::value...};
        }
    }

    template <typename... T>
    constexpr auto tuple_cat(T&&... ts) {
        if constexpr (sizeof...(T) == 0) {
            return Tuple<>();
        } else {
            // It appears that Clang produces better assembly when
            // TUPLET_CAT_BY_FORWARDING_TUPLE == 0, while GCC produces better assembly when
            // TUPLET_CAT_BY_FORWARDING_TUPLE == 1. MSVC always produces terrible assembly
            // in either case. This will set TUPLET_CAT_BY_FORWARDING_TUPLE to the correct
            // value (0 for clang, 1 for everyone else)
            // See: https://github.com/codeinred/tuplet/discussions/14
            #if defined(NOA_COMPILER_CLANG)
            using big_tuple = Tuple<std::decay_t<T>...>;
            #else
            using big_tuple = Tuple<T&& ...>;
            #endif

            using outer_bases = nt::base_list_t<big_tuple>;
            constexpr auto outer = guts::tuple_get_outer_bases(outer_bases{});
            constexpr auto inner = guts::tuple_get_inner_bases(outer_bases{});
            return guts::tuple_cat(big_tuple{static_cast<T&&>(ts)...}, outer, inner);
        }
    }
}

namespace std {
    template<class... T>
    struct tuple_size<noa::Tuple<T...>> : std::integral_constant<size_t, sizeof...(T)> {};

    template<class... T>
    struct tuple_size<const noa::Tuple<T...>> : std::integral_constant<size_t, sizeof...(T)> {};

    template<typename A, typename B>
    struct tuple_size<noa::Pair<A, B>> : std::integral_constant<size_t, 2> {};

    template<typename A, typename B>
    struct tuple_size<const noa::Pair<A, B>> : std::integral_constant<size_t, 2> {};

    template<size_t I, typename... T>
    struct tuple_element<I, noa::Tuple<T...>> {
        using type = decltype(noa::Tuple<T...>::declval(noa::guts::tag<I>()));
    };

    template<size_t I, typename... T>
    struct tuple_element<I, const noa::Tuple<T...>> {
        using type = const decltype(noa::Tuple<T...>::declval(noa::guts::tag<I>()));
    };

    template<size_t I, typename A, typename B>
    struct tuple_element<I, noa::Pair<A, B>> {
        static_assert(I < 2);
        using type = std::conditional_t<I == 0, A, B>;
    };

    template<size_t I, typename A, typename B>
    struct tuple_element<I, const noa::Pair<A, B>> {
        static_assert(I < 2);
        using type = std::conditional_t<I == 0, const A, const B>;
    };
}
