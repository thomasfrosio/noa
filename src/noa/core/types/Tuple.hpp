#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"

#if defined(NOA_IS_OFFLINE)
#include <cstddef>
#include <utility>
#include <functional> // reference_wrapper
#include "noa/core/string/Format.hpp"
#else
#include <cuda/std/cstddef>
#include <cuda/std/utility>
#include <cuda/std/functional> // reference_wrapper
#endif

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wmissing-braces"
#elif defined(NOA_COMPILER_MSVC)
    #pragma warning(push, 0)
#endif

/// Original implementation: https://github.com/codeinred/tuplet
/// - Type names, namespaces and code-format are changed, just to have this code fully integrated into noa.
/// - Comparisons, concepts, and MSVC-support are removed to simplify the code (because we don't need that).
/// - tuple_base_t implementation was changed to support nvrtc (weird bug?).
///
/// Why not std::tuple?
/// - std::tuple (or cuda::std::tuple) could absolutely be used, but this Tuple is significantly more efficient,
///   likely because Tuple is an aggregate and aggregate are (apparently) easier to optimize.
/// - Controlling the layout of the tuple and being able to easily manipulate the internals is very useful to
///   add efficient functions to manipulate the tuple elements.
/// - One issue with the Tuple implementation is that aggregate initialization triggers the pedantic warning
///   -Wmissing-braces. Braces can be added to acknowledge the fact that each tuple element is from a separate
///   base class, but this is a bit annoying. This warning could be useful in other places so it's probably
///   best to keep it. If adding the extra braces is unacceptable, the make_tuple() template function can
///   be an alternative solution.

namespace noa::inline types {
    template<size_t I>
    using Tag = std::integral_constant<size_t, I>;

    template<class... T>
    struct Tuple;
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

    template<typename Tup>
    using decay_list_t = typename std::decay_t<Tup>::decay_list;

    template<typename Tup, typename = typename Tup::base_list>
    constexpr bool has_base_list(int) {
        return true;
    }

    template<typename Tup>
    constexpr bool has_base_list(long long) {
        return false;
    }

    template<typename Tup>
    constexpr auto has_base_list_v = has_base_list<Tup>(0);
}

namespace noa::guts {
    // Stores an element from the tuple.
    // The index is stored in the type, so that each tuple element is unique.
    template<size_t I, typename T>
    struct TupleElement {
        using type = T;
        static constexpr size_t INDEX = I;
        [[no_unique_address]] T value;

        inline constexpr decltype(auto) operator[](Tag<I>) & {
            return (value);
        }
        inline constexpr decltype(auto) operator[](Tag<I>) const& {
            return (value);
        }
        inline constexpr decltype(auto) operator[](Tag<I>) && {
            return (static_cast<TupleElement&&>(*this).value);
        }

        // Used to extract the type for the Ith element.
        static auto declval(Tag<I>) -> T;
    };

    // The nested type that stores the TupleElements.
    // The Tuple directly (and only) inherits from this type.
    template<typename... Bases>
    struct TypeMap : Bases... {
        using base_list = nt::TypeList<Bases...>;
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
    template<typename... T>
    struct GetTypeMap {
        template<size_t... I>
        auto operator()(std::index_sequence<I...>) -> TypeMap<TupleElement<I, T>...>;
    };

    // Constructs the tuple base type.
    // This alias is simply here to create the index sequence.
    template<typename... T>
    using TupleBase = typename std::invoke_result_t<guts::GetTypeMap<T...>, std::make_index_sequence<sizeof...(T)>>;
}

namespace noa::guts {
    template<typename Tup, typename F, typename... B>
    inline constexpr void tuple_for_each(Tup&& tup, F&& func, nt::TypeList<B...>) {
        (void(func(tup.::std::type_identity_t<B>::value)), ...);
    }

    template<typename Tup, typename F, typename... B>
    inline constexpr void tuple_for_each_enumerate(Tup&& tup, F&& func, nt::TypeList<B...>) {
        (void(func<B::SIZE>(tup.::std::type_identity_t<B>::value)), ...);
    }

    template<typename Tup, typename F, typename... B>
    inline constexpr bool tuple_any(Tup&& tup, F&& func, nt::TypeList<B...>) {
        return (bool(func(tup.::std::type_identity_t<B>::value)) || ...);
    }

    template<typename Tup, typename F, typename... B>
    inline constexpr bool tuple_any_enumerate(Tup&& tup, F&& func, nt::TypeList<B...>) {
        return (bool(func<B::SIZE>(tup.::std::type_identity_t<B>::value)) || ...);
    }

    template<typename Tup, typename F, typename... B>
    inline constexpr bool tuple_all(Tup&& tup, F&& func, nt::TypeList<B...>) {
        return (bool(func(tup.::std::type_identity_t<B>::value)) && ...);
    }

    template<typename Tup, typename F, typename... B>
    inline constexpr bool tuple_all_enumerate(Tup&& tup, F&& func, nt::TypeList<B...>) {
        return (bool(func<B::SIZE>(tup.::std::type_identity_t<B>::value)) && ...);
    }

    template<typename Tup, typename F, typename... B>
    inline constexpr auto tuple_map(Tup&& tup, F&& func, nt::TypeList<B...>)
    -> Tuple<decltype(func(tup.::std::type_identity_t<B>::value))...> {
        return {func(tup.::std::type_identity_t<B>::value)...};
    }

    template<typename Tup, typename F, typename... B>
    inline constexpr auto tuple_map_enumerate(Tup&& tup, F&& func, nt::TypeList<B...>)
    -> Tuple<decltype(func<B::SIZE>(tup.::std::type_identity_t<B>::value))...> {
        return {func<B::SIZE>(tup.::std::type_identity_t<B>::value)...};
    }

    template<typename Tup, typename F, typename... B>
    inline constexpr decltype(auto) tuple_apply(Tup&& t, F&& f, nt::TypeList<B...>) {
        return static_cast<F&&>(f)(t.::std::type_identity_t<B>::value...);
    }

    template<typename U, typename Tup, typename... B>
    inline constexpr U tuple_convert(Tup&& t, nt::TypeList<B...>) {
        return U{t.::std::type_identity_t<B>::value...};
    }
}

namespace noa::inline types {
    /// Efficient tuple aggregate-type.
    template<class... T>
    struct Tuple : guts::TupleBase<T...> {
        constexpr static size_t SIZE = sizeof...(T);
        constexpr static bool nothrow_swappable = (std::is_nothrow_swappable_v<T> && ...);

        using super = guts::TupleBase<T...>;
        using super::operator[];
        using base_list = typename super::base_list;
        using element_list = nt::TypeList<T...>;
        using decay_list = nt::TypeList<std::decay_t<T>...>;
        using decayed_tuple = Tuple<std::decay_t<T>...>;
        using super::declval;

    public: // Additional element access
        template<size_t I>
        inline constexpr decltype(auto) get() & {
            return ((*this)[Tag<I>{}]);
        }

        template<size_t I>
        inline constexpr decltype(auto) get() const& {
            return ((*this)[Tag<I>{}]);
        }

        template<size_t I>
        inline constexpr decltype(auto) get() && {
            return (static_cast<Tuple&&>(*this)[Tag<I>{}]);
        }

    public: // Whole tuple assignment
        template<typename U, typename = std::enable_if_t<
                 !std::is_same_v<std::decay_t<Tuple>, std::decay_t<U>>>> // Preserves default assignments
        inline constexpr auto& operator=(U&& tup) {
            using tuple_t = std::decay_t<U>;
            if constexpr (nt::has_base_list_v<tuple_t>) {
                assign_tup_(static_cast<U&&>(tup), base_list{}, typename tuple_t::base_list{});
            } else {
                assign_index_tup_(static_cast<U&&>(tup), std::make_index_sequence<SIZE>());
            }
            return *this;
        }

        template<class... U>
        constexpr auto& assign(U&& ... values) {
            assign_(base_list{}, static_cast<U&&>(values)...);
            return *this;
        }

        inline constexpr void swap(Tuple& other) noexcept(nothrow_swappable) {
            swap_(other, base_list {});
        }

    public: // Utility functions
        /// Applies a function to every element of the tuple. The order is the
        /// declaration order, so first the function will be applied to element
        /// 0, then element 1, then element 2, and so on, where element N is
        /// identified by get<N>
        template<class F>
        inline constexpr void for_each(F&& func)& {
            guts::tuple_for_each(*this, static_cast<F&&>(func), base_list{});
        }
        template<class F>
        inline constexpr void for_each(F&& func) const& {
            guts::tuple_for_each(*this, static_cast<F&&>(func), base_list{});
        }
        template<class F>
        inline constexpr void for_each(F&& func)&& {
            guts::tuple_for_each(static_cast<Tuple&&>(*this), static_cast<F&&>(func), base_list{});
        }

        /// Same as for_each, but passes the index of the current tuple element to func as its first template parameter.
        template<class F>
        inline constexpr void for_each_enumerate(F&& func)& {
            guts::tuple_for_each_enumerate(*this, static_cast<F&&>(func), base_list{});
        }
        template<class F>
        inline constexpr void for_each_enumerate(F&& func) const& {
            guts::tuple_for_each_enumerate(*this, static_cast<F&&>(func), base_list{});
        }
        template<class F>
        inline constexpr void for_each_enumerate(F&& func)&& {
            guts::tuple_for_each_enumerate(static_cast<Tuple&&>(*this), static_cast<F&&>(func), base_list{});
        }

        /// Applies a function to each element successively, until one returns true.
        /// Returns true if any application returned true, otherwise returns false.
        template<typename F>
        inline constexpr bool any(F&& func)& {
            return guts::tuple_any(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr bool any(F&& func) const& {
            return guts::tuple_any(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr bool any(F&& func)&& {
            return guts::tuple_any(static_cast<Tuple&&>(*this), static_cast<F&&>(func), base_list{});
        }

        /// Same as any, but passes the index of the current tuple element to func as its first template parameter.
        template<typename F>
        inline constexpr bool any_enumerate(F&& func)& {
            return guts::tuple_any_enumerate(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr bool any_enumerate(F&& func) const& {
            return guts::tuple_any_enumerate(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr bool any_enumerate(F&& func)&& {
            return guts::tuple_any_enumerate(static_cast<Tuple&&>(*this), static_cast<F&&>(func), base_list{});
        }

        /// Applies a function to each element successively, until one returns false.
        /// Returns true if every application returned true, otherwise returns false.
        template<typename F>
        inline constexpr bool all(F&& func)& {
            return guts::tuple_all(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr bool all(F&& func) const& {
            return guts::tuple_all(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr bool all(F&& func)&& {
            return guts::tuple_all(static_cast<Tuple&&>(*this), static_cast<F&&>(func), base_list{});
        }

        /// Same as all, but passes the index of the current tuple element to func as its first template parameter.
        template<typename F>
        inline constexpr bool all_enumerate(F&& func)& {
            return guts::tuple_all_enumerate(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr bool all_enumerate(F&& func) const& {
            return guts::tuple_all_enumerate(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr bool all_enumerate(F&& func)&& {
            return guts::tuple_all_enumerate(static_cast<Tuple&&>(*this), static_cast<F&&>(func), base_list{});
        }

        /// Map a function over every element in the tuple, using the returned values to construct a new tuple.
        template<typename F>
        inline constexpr auto map(F&& func)& {
            return guts::tuple_map(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr auto map(F&& func) const& {
            return guts::tuple_map(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr auto map(F&& func)&& {
            return guts::tuple_map(static_cast<Tuple&&>(*this), static_cast<F&&>(func), base_list{});
        }

        /// Same as map, but passes the index of the current tuple element to func as its first template parameter.
        template<typename F>
        inline constexpr auto map_enumerate(F&& func)& {
            return guts::tuple_map_enumerate(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr auto map_enumerate(F&& func) const& {
            return guts::tuple_map_enumerate(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr auto map_enumerate(F&& func)&& {
            return guts::tuple_map_enumerate(static_cast<Tuple&&>(*this), static_cast<F&&>(func), base_list{});
        }

        /// See apply.
        template<typename F>
        inline constexpr decltype(auto) apply(F&& func)& {
            return guts::tuple_apply(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr decltype(auto) apply(F&& func) const& {
            return guts::tuple_apply(*this, static_cast<F&&>(func), base_list{});
        }
        template<typename F>
        inline constexpr decltype(auto) apply(F&& func)&& {
            return guts::tuple_apply(static_cast<Tuple&&>(*this), static_cast<F&&>(func), base_list{});
        }

    public: // Type conversion
        template<typename... U>
        constexpr explicit operator Tuple<U...>()& {
            static_assert(sizeof...(U) == SIZE, "Can only convert to tuples with the same number of items");
            return guts::tuple_convert<Tuple<U...>>(*this, base_list{});
        }
        template<typename... U>
        constexpr explicit operator Tuple<U...>() const& {
            static_assert(sizeof...(U) == SIZE, "Can only convert to tuples with the same number of items");
            return guts::tuple_convert<Tuple<U...>>(*this, base_list{});
        }
        template<typename... U>
        constexpr explicit operator Tuple<U...>()&& {
            static_assert(sizeof...(U) == SIZE, "Can only convert to tuples with the same number of items");
            return guts::tuple_convert<Tuple<U...>>(static_cast<Tuple&&>(*this), base_list{});
        }

        /// Instantiate the given type using list initialization
        template<typename U>
        inline constexpr U as()& {
            return guts::tuple_convert<U>(*this, base_list{});
        }
        template<typename U>
        inline constexpr U as() const& {
            return guts::tuple_convert<U>(*this, base_list{});
        }
        template<typename U>
        inline constexpr U as()&& {
            return guts::tuple_convert<U>(static_cast<Tuple&&>(*this), base_list{});
        }

        inline constexpr auto decay()& {
            return guts::tuple_convert<decayed_tuple>(*this, base_list{});
        }
        inline constexpr auto decay() const& {
            return guts::tuple_convert<decayed_tuple>(*this, base_list{});
        }
        inline constexpr auto decay()&& {
            return guts::tuple_convert<decayed_tuple>(static_cast<Tuple&&>(*this), base_list{});
        }

    private:
        template<class... B>
        inline constexpr void swap_(Tuple& other, nt::TypeList<B...>) noexcept(nothrow_swappable) {
            using std::swap;
            (swap(B::value, other.::std::type_identity_t<B>::value), ...);
        }

        template<class U, class... B1, class... B2>
        inline constexpr void assign_tup_(U&& u, nt::TypeList<B1...>, nt::TypeList<B2...>) {
            (void(B1::value = static_cast<U&&>(u).::std::type_identity_t<B2>::value), ...);
        }
        template<class U, size_t... I>
        inline constexpr void assign_index_tup_(U&& u, std::index_sequence<I...>) {
            using std::get;
            (void(guts::TupleElement<I, T>::value = get<I>(static_cast<U&&>(u))), ...);
        }
        template<class... U, class... B>
        inline constexpr void assign_(nt::TypeList<B...>, U&& ... u) {
            (void(B::value = static_cast<U&&>(u)), ...);
        }
    };

    template<>
    struct Tuple<> : guts::TupleBase<> {
        constexpr static size_t SIZE = 0;
        constexpr static bool nothrow_swappable = true;
        using super = guts::TupleBase<>;
        using base_list = nt::TypeList<>;
        using element_list = nt::TypeList<>;
        using decay_list = nt::TypeList<>;
        using decayed_tuple = Tuple<>;

        template<size_t>
        inline constexpr void get() {}

        template<typename U, typename = std::enable_if_t<!std::is_same_v<std::decay_t<Tuple>, std::decay_t<U>>>>
        constexpr auto& operator=(U&&) noexcept {
            return *this;
        }

        constexpr void swap(Tuple) noexcept {}
        constexpr auto& assign() noexcept { return *this; }

        constexpr void for_each(auto&&) const noexcept {}
        constexpr bool any(auto&&) const noexcept { return false; }
        constexpr bool all(auto&&) const noexcept { return true; }
        constexpr auto map(auto&&) const noexcept { return Tuple{}; }
        constexpr void for_each_enumerate(auto&&) const noexcept {}
        constexpr bool any_enumerate(auto&&) const noexcept { return false; }
        constexpr bool all_enumerate(auto&&) const noexcept { return true; }
        constexpr auto map_enumerate(auto&&) const noexcept { return Tuple{}; }
        constexpr decltype(auto) apply(auto&& func) const noexcept { return func(); }

        template<class U>
        constexpr U as() const noexcept { return U{}; }

        [[nodiscard]] constexpr Tuple decay() const noexcept {
            return {};
        }
    };

    // Deduction guide.
    template<typename... Ts>
    Tuple(Ts...) -> Tuple<nt::unwrap_ref_decay_t<Ts>...>;
}

namespace noa::inline types {
    template<typename First, typename Second>
    struct Pair {
        constexpr static size_t N = 2;
        constexpr static bool nothrow_swappable =
                std::is_nothrow_swappable_v<First>
                && std::is_nothrow_swappable_v<Second>;

        [[no_unique_address]] First first;
        [[no_unique_address]] Second second;

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
            return guts::tuple_convert<U>(static_cast<Tuple&&>(tuple), base_list{});
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

    /// Copies the arguments in a tuple.
    /// Use move(arg) to move it into the tuple.
    /// Use ref(arg) to save a reference instead.
    template<typename... Ts>
    inline constexpr auto make_tuple(Ts&&... args) {
        return Tuple<nt::unwrap_ref_decay_t<Ts>...>{static_cast<Ts&&>(args)...};
    }

    /// Saves references of arguments in a tuple.
    /// Use rvalues have their lifetime extended.
    template<typename... T>
    inline constexpr auto forward_as_tuple(T&&... a) noexcept {
        return Tuple<T&&...>{static_cast<T&&>(a)...};
    }
}

namespace noa {
    namespace guts {
        template<typename T, typename... Q>
        inline constexpr auto tuple_repeat_type(nt::TypeList<Q...>) {
            return nt::TypeList<nt::first_t<T, Q>...>{};
        }
        template<class... Outer>
        inline constexpr auto tuple_get_outer_bases(nt::TypeList<Outer...>) {
            return (tuple_repeat_type<Outer>(nt::base_list_t<nt::type_type_t<Outer>>{}) + ...);
        }
        template<class... Outer>
        inline constexpr auto tuple_get_inner_bases(nt::TypeList<Outer...>) {
            return (nt::base_list_t<nt::type_type_t<Outer>>{} + ...);
        }

        // This takes a forwarding tuple as a parameter. The forwarding tuple only
        // contains references, so it should just be taken by value.
        template<class T, class... Outer, class... Inner>
        inline constexpr auto tuple_cat(
                T tup,
                nt::TypeList<Outer...>,
                nt::TypeList<Inner...>
        ) -> Tuple<nt::type_type_t<Inner>...> {
            return {static_cast<nt::type_type_t<Outer>&&>(
                            tup.::std::type_identity_t<Outer>::value
                    ).::std::type_identity_t<Inner>::value...};
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

    /// Filters the tuple based on a predicate.
    /// The predicate should be a constant expression; if it returns true, keep the tuple argument, otherwise discard.
    /// Transform is an optional operator used to transform the argument before saving it into the new tuple.
    /// FIXME T will be S if tuple is moved and contains S, otherwise T is S&
    /// FIXME https://godbolt.org/z/fx3Kh3rs8
    template<typename Predicate, typename Transform, typename T> requires nt::is_tuple_v<T>
    auto tuple_filter(T&& tuple) {
        return apply(
                []<typename F, typename...R>(F&& first, R&& ... rest) {
                    auto filtered_rest = [&] {
                        if constexpr (sizeof...(rest)) {
                            return tuple_filter<Predicate, Transform>(forward_as_tuple(std::forward<R>(rest)...));
                        } else {
                            return Tuple{};
                        }
                    }();

                    if constexpr (Predicate{}.template operator()<T>()) { // FIXME loop through TupleElement to pass exact type
                        return tuple_cat(Transform{}(std::forward<T>(first)), std::move(filtered_rest));
                    } else {
                        return filtered_rest;
                    }
                }, std::forward<T>(tuple));
    }
}

namespace std {
    template<class... T> struct tuple_size<noa::Tuple<T...>> : std::integral_constant<size_t, sizeof...(T)> {};
    template<class... T> struct tuple_size<const noa::Tuple<T...>> : std::integral_constant<size_t, sizeof...(T)> {};
    template<typename A, typename B> struct tuple_size<noa::Pair<A, B>> : std::integral_constant<size_t, 2> {};
    template<typename A, typename B> struct tuple_size<const noa::Pair<A, B>> : std::integral_constant<size_t, 2> {};

    template<size_t I, typename... T>
    struct tuple_element<I, noa::Tuple<T...>> {
        using type = decltype(noa::Tuple<T...>::declval(noa::Tag<I>()));
    };

    template<size_t I, typename... T>
    struct tuple_element<I, const noa::Tuple<T...>> {
        using type = const decltype(noa::Tuple<T...>::declval(noa::Tag<I>()));
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

namespace noa::traits {
    template<typename... Ts>
    struct proclaim_is_tuple<noa::Tuple<Ts...>> : std::true_type {};

    template<typename... Ts>
    struct proclaim_is_tuple_of_accessor<noa::Tuple<Ts...>> : std::bool_constant<nt::are_accessor<Ts...>::value> {};
}

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
    #pragma warning(pop)
#endif

#if defined(NOA_IS_OFFLINE)
namespace fmt {
    template<typename... T>
    struct fmt::formatter<noa::Tuple<T...>> {
        char open_char = '(';
        char separator = ',';
        char close_char = ')';
        constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
            constexpr auto npos = std::string_view::npos;
            std::string_view view(ctx.begin(), static_cast<size_t>(ctx.end() - ctx.begin()));
            if (view.empty()) {
                return ctx.begin();
            } else if (view.size() == 3) {
                open_char = view[0];
                close_char = view[1];
            } else if (view.size() == 4) {
                open_char = view[0];
                separator = view[1];
                close_char = view[2];
            } else {
                throw fmt::format_error(fmt::format(
                        "Format specification {} is currently unsupported",
                        view));
            }
            if (std::string_view("<{[(").find(open_char) == npos) {
                throw fmt::format_error(
                        fmt::format("Enable to interpret open char {}", open_char));
            }
            if (std::string_view(">}])").find(close_char) == npos) {
                throw fmt::format_error(
                        fmt::format("Unable to interpret close char {}", close_char));
            }
            return ctx.begin() + view.size() - 1;
        }
        template<typename FormatContext>
        constexpr auto format(const noa::Tuple<T...>& p, FormatContext& ctx) -> decltype(ctx.out()) {
            if constexpr (sizeof...(T) >= 1) {
                auto print_elems = [&](auto const& first, auto const& ... rest) {
                    auto out = fmt::format_to(ctx.out(), "{}{}", open_char, first);
                    ((out = fmt::format_to(out, "{} {}", separator, rest)), ...);
                    return fmt::format_to(out, "{}", close_char);
                };
                return p.apply(print_elems);
            } else {
                return fmt::format_to(ctx.out(), "{}{}", open_char, close_char);
            }
        }
    };
}
#endif
