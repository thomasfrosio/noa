#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"
#include "noa/core/types/Pair.hpp"

#ifdef NOA_IS_OFFLINE
#include "noa/core/string/Format.hpp"
#endif

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wmissing-braces"
#elif defined(NOA_COMPILER_MSVC)
    #pragma warning(push, 0)
#endif

/// Original implementation: https://github.com/codeinred/tuplet
/// - Change the way the elements are accessed to properly match std::tuple.
/// - Use std::forward/move, instead of a direct cast.
/// - Type names, namespaces and code-format are changed, just to have this code fully integrated into noa.
/// - Comparisons, concepts, and MSVC-support are removed to simplify the code (because we don't need that).
/// - TupleBase implementation was changed to support nvrtc (weird bug?).
/// - Added a few utility functions, e.g. *_enumerate() functions.
/// - Add support for c-arrays
///
/// Why not std::tuple?
/// - std::tuple (or cuda::std::tuple) could be used, but this Tuple is more efficient due to being an aggregate
///   (aggregates are (apparently) easier to optimize).
/// - Controlling the layout of the tuple and being able to easily manipulate the internals is very useful to
///   add efficient functions to manipulate the tuple elements.
/// - One issue with the Tuple implementation is that aggregate initialization triggers the pedantic warning
///   -Wmissing-braces. Braces can be added to acknowledge the fact that each tuple element is from a separate
///   base class, but this is a bit annoying as it exposes an implementation detail. This warning could be useful
///   in other places so it's probably best to keep it. If adding the extra braces is unacceptable, using the
///   make_tuple() or forward_as_tuple() template functions is the recommended solution.

namespace noa::inline types {
    template<class... T>
    struct Tuple;
}

namespace noa::guts {
    template<size_t I, typename T>
    struct TupleElement {
        using type = T;
        static constexpr size_t INDEX = I;
        NOA_NO_UNIQUE_ADDRESS T value;

        // std::get<I>(...) equivalent.
        constexpr auto operator[](Tag<I>) & -> T& { return value; }
        constexpr auto operator[](Tag<I>) const& -> const T& { return value; }
        constexpr auto operator[](Tag<I>) && -> T&& { return std::forward<T>(value); }
        constexpr auto operator[](Tag<I>) const && -> const T&& { return std::forward<const T>(value); }

        // Used to extract the type for the Ith element.
        static constexpr auto declval(Tag<I>) -> std::type_identity<T>;
    };

    // The nested type that stores the TupleElements.
    // The Tuple directly (and only) inherits from this type.
    template<typename... Elements>
    struct TypeMap : Elements... {
        using element_list = nt::TypeList<Elements...>;
        using Elements::operator[]...;
        using Elements::declval...;
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
        constexpr auto operator()(std::index_sequence<I...>) -> TypeMap<TupleElement<I, T>...>;
    };

    // Constructs the tuple base type.
    // This alias is simply here to create the index sequence.
    // FIXME don't think we need the type_identity here, but better safe than sorry.
    template<typename... T>
    using TupleBase = typename std::invoke_result_t<
            guts::GetTypeMap<std::type_identity_t<T>...>,
            std::index_sequence_for<T...>>;
}

namespace noa::guts {
    template<typename Tup, typename F, size_t... B>
    constexpr void tuple_for_each(Tup&& t, F&& f, std::index_sequence<B...>) {
        ((std::forward<F>(f)(std::forward<Tup>(t)[Tag<B>{}])), ...);
    }

    template<typename Tup, typename F, size_t... B>
    constexpr void tuple_for_each_enumerate(Tup&& t, F&& f, std::index_sequence<B...>) {
        ((std::forward<F>(f).template operator()<B>(std::forward<Tup>(t)[Tag<B>{}])), ...);
    }

    template<typename Tup, typename F, size_t... B>
    constexpr bool tuple_any(Tup&& t, F&& f, std::index_sequence<B...>) {
        return (static_cast<bool>(std::forward<F>(f)(std::forward<Tup>(t)[Tag<B>{}])) || ...);
    }

    template<typename Tup, typename F, size_t... B>
    constexpr bool tuple_any_enumerate(Tup&& t, F&& f, std::index_sequence<B...>) {
        return (static_cast<bool>(std::forward<F>(f).template operator()<B>(std::forward<Tup>(t)[Tag<B>{}])) || ...);
    }

    template<typename Tup, typename F, size_t... B>
    constexpr bool tuple_all(Tup&& t, F&& f, std::index_sequence<B...>) {
        return (static_cast<bool>(std::forward<F>(f)(std::forward<Tup>(t)[Tag<B>{}])) && ...);
    }

    template<typename Tup, typename F, size_t... B>
    constexpr bool tuple_all_enumerate(Tup&& t, F&& f, std::index_sequence<B...>) {
        return (static_cast<bool>(std::forward<F>(f).template operator()<B>(std::forward<Tup>(t)[Tag<B>{}])) && ...);
    }

    template<typename Tup, typename F, size_t... B>
    constexpr auto tuple_map(Tup&& t, F&& f, std::index_sequence<B...>) {
        return Tuple{std::forward<F>(f)(std::forward<Tup>(t)[Tag<B>{}])...};
    }

    template<typename Tup, typename F, size_t... B>
    constexpr auto tuple_map_enumerate(Tup&& t, F&& f, std::index_sequence<B...>) {
        return Tuple{std::forward<F>(f).template operator()<B>(std::forward<Tup>(t)[Tag<B>{}])...};
    }

    template<typename Tup, typename F, size_t... B>
    constexpr decltype(auto) tuple_apply(Tup&& t, F&& f, std::index_sequence<B...>) {
        return std::forward<F>(f)(std::forward<Tup>(t)[Tag<B>{}]...);
    }

    template<typename U, typename Tup, size_t... B>
    constexpr U tuple_convert(Tup&& t, std::index_sequence<B...>) {
        return U{std::forward<Tup>(t)[Tag<B>{}]...};
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
        using super::declval;
        using element_list = typename super::element_list;
        using type_list = nt::TypeList<T...>;
        using decayed_type_list = nt::TypeList<std::decay_t<T>...>;
        using decayed_tuple = Tuple<std::decay_t<T>...>;
        using index_list = std::make_index_sequence<SIZE>;

    public:
        template<typename... U>
        constexpr auto& assign(U&&... values) {
            assign_value_(index_list{}, std::forward<U>(values)...);
            return *this;
        }

        constexpr void swap(Tuple& other) noexcept(nothrow_swappable) {
            swap_(other, index_list{});
        }

    public: // Utility functions
        #define NOA_MAKE_TUPLE_UTILITY_FUNCS(name) \
        template<typename F> constexpr decltype(auto) name(F&& f) &        { return guts::tuple_##name(*this, std::forward<F>(f), index_list{}); }             \
        template<typename F> constexpr decltype(auto) name(F&& f) const&   { return guts::tuple_##name(*this, std::forward<F>(f), index_list{}); }             \
        template<typename F> constexpr decltype(auto) name(F&& f) &&       { return guts::tuple_##name(std::move(*this), std::forward<F>(f), index_list{}); }  \
        template<typename F> constexpr decltype(auto) name(F&& f) const && { return guts::tuple_##name(std::move(*this), std::forward<F>(f), index_list{}); }

        /// Applies a function to every element of the tuple. The order is the declaration order, so first the
        /// function will be applied to element 0, then element 1, then element 2, and so on, where element N
        /// is identified by get<N>.
        NOA_MAKE_TUPLE_UTILITY_FUNCS(for_each)

        /// Same as for_each, but passes the index of the current tuple element to func as its first template parameter.
        NOA_MAKE_TUPLE_UTILITY_FUNCS(for_each_enumerate)

        /// Applies a function to each element successively, until one returns true.
        /// Returns true if any application returned true, otherwise returns false.
        NOA_MAKE_TUPLE_UTILITY_FUNCS(any)
        NOA_MAKE_TUPLE_UTILITY_FUNCS(any_enumerate)

        /// Applies a function to each element successively, until one returns false.
        /// Returns true if every application returned true, otherwise returns false.
        NOA_MAKE_TUPLE_UTILITY_FUNCS(all)
        NOA_MAKE_TUPLE_UTILITY_FUNCS(all_enumerate)

        /// Map a function over every element in the tuple, using the returned values to construct a new tuple.
        NOA_MAKE_TUPLE_UTILITY_FUNCS(map)
        NOA_MAKE_TUPLE_UTILITY_FUNCS(map_enumerate)

        /// See apply.
        NOA_MAKE_TUPLE_UTILITY_FUNCS(apply)

        #undef NOA_MAKE_TUPLE_UTILITY_FUNCS

    public: // Type conversion
        template<typename... U>
        constexpr explicit operator Tuple<U...>() & {
            static_assert(sizeof...(U) == SIZE, "Can only convert to tuples with the same number of items");
            return guts::tuple_convert<Tuple<U...>>(*this, index_list{});
        }
        template<typename... U>
        constexpr explicit operator Tuple<U...>() const& {
            static_assert(sizeof...(U) == SIZE, "Can only convert to tuples with the same number of items");
            return guts::tuple_convert<Tuple<U...>>(*this, index_list{});
        }
        template<typename... U>
        constexpr explicit operator Tuple<U...>() && {
            static_assert(sizeof...(U) == SIZE, "Can only convert to tuples with the same number of items");
            return guts::tuple_convert<Tuple<U...>>(std::move(*this), index_list{});
        }

        /// Instantiate the given type using list initialization
        constexpr auto decay() & {
            return guts::tuple_convert<decayed_tuple>(*this, index_list{});
        }
        constexpr auto decay() const& {
            return guts::tuple_convert<decayed_tuple>(*this, index_list{});
        }
        constexpr auto decay() && {
            return guts::tuple_convert<decayed_tuple>(std::move(*this), index_list{});
        }

    private:
        template<size_t... I>
        constexpr void swap_(Tuple& other, std::index_sequence<I...>) noexcept(nothrow_swappable) {
            using std::swap;
            (swap((*this)[Tag<I>{}], other[Tag<I>{}]), ...);
        }

        template<size_t... I, class... U>
        constexpr void assign_value_(std::index_sequence<I...>, U&&... values) {
            (((*this)[Tag<I>{}] = std::forward<U>(values)), ...);
        }
    };

    template<>
    struct Tuple<> : guts::TupleBase<> {
        constexpr static size_t SIZE = 0;
        constexpr static bool nothrow_swappable = true;
        using super = guts::TupleBase<>;
        using element_list = nt::TypeList<>;
        using type_list = nt::TypeList<>;
        using decayed_type_list = nt::TypeList<>;
        using decayed_tuple = Tuple<>;
        using index_list = std::make_index_sequence<SIZE>;

        template<typename U>
        requires (!std::is_same_v<std::decay_t<Tuple>, std::decay_t<U>>)
        constexpr auto& operator=(U&&) noexcept {
            return *this;
        }

        constexpr void swap(Tuple) noexcept {}
        constexpr Tuple& assign() noexcept { return *this; }

        constexpr void for_each(auto&&) const noexcept {}
        constexpr bool any(auto&&) const noexcept { return false; }
        constexpr bool all(auto&&) const noexcept { return true; }
        constexpr auto map(auto&&) const noexcept { return Tuple{}; }
        constexpr void for_each_enumerate(auto&&) const noexcept {}
        constexpr bool any_enumerate(auto&&) const noexcept { return false; }
        constexpr bool all_enumerate(auto&&) const noexcept { return true; }
        constexpr auto map_enumerate(auto&&) const noexcept { return Tuple{}; }
        constexpr decltype(auto) apply(auto&& func) const noexcept { return func(); }
        [[nodiscard]] constexpr Tuple decay() const noexcept { return {}; }
    };

    /// Deduction guide.
    /// As opposed to std::tuple, we unwrap the std::reference_wrapper during CTAD.
    /// As a result, Tuple{...} and noa::make_tuple(...) are equivalent.
    template<typename... Ts>
    Tuple(Ts...) -> Tuple<nt::unwrap_reference_decay_t<Ts>...>;
}

namespace noa {
    /// Converts from one tuple type to any other tuple or U.
    /// U output = noa::Convert{tuple}; // which is equivalent to:
    /// U output = static_cast<U>(tuple);
    template<typename Tuple>
    struct Convert {
        Tuple tuple;

        template<typename U>
        constexpr /* implicit */ operator U()&& {
            using index_list = typename std::decay_t<Tuple>::index_list;
            return guts::tuple_convert<U>(std::forward<Tuple>(tuple), index_list{});
        }
    };

    // Deduction guides...
    template<typename Tuple>
    Convert(Tuple&) -> Convert<Tuple&>;
    template<typename Tuple>
    Convert(Tuple const&) -> Convert<Tuple const&>;
    template<typename Tuple>
    Convert(Tuple&&) -> Convert<Tuple>;
}

namespace noa {
    template<typename... T>
    constexpr Tuple<T& ...> tie(T&... t) { return {t...}; }

    template<typename F, typename Tup>
    constexpr decltype(auto) apply(F&& func, Tup&& tup) { // works for Pair too
        return guts::tuple_apply(std::forward<Tup>(tup), std::forward<F>(func), nt::index_list_t<Tup>{});
    }

    template<typename... T>
    constexpr void swap(Tuple<T...>& a, Tuple<T...>& b) noexcept(Tuple<T...>::nothrow_swappable) {
        a.swap(b);
    }

    /// Copies the arguments in a tuple.
    /// Use std::move(arg) to move it into the tuple.
    /// Use std::ref(arg) to save a reference instead.
    template<typename... Ts>
    constexpr auto make_tuple(Ts&&... args) {
        return Tuple<nt::unwrap_reference_decay_t<Ts>...>{std::forward<Ts>(args)...};
    }

    /// Saves references of arguments in a tuple.
    /// rvalues have their lifetime extended.
    template<typename... Ts>
    constexpr auto forward_as_tuple(Ts&&... args) noexcept {
        return Tuple<Ts&&...>{std::forward<Ts>(args)...};
    }
}

namespace noa {
    namespace guts {
        template<typename T, typename... Q>
        constexpr auto tuple_repeat_type(nt::TypeList<Q...>) {
            return nt::TypeList<nt::first_t<T, Q>...>{};
        }
        template<class... Outer>
        constexpr auto tuple_get_outer_elements(nt::TypeList<Outer...>) {
            return (tuple_repeat_type<Outer>(nt::element_list_t<nt::type_type_t<Outer>>{}) + ...);
        }
        template<class... Outer>
        constexpr auto tuple_get_inner_elements(nt::TypeList<Outer...>) {
            return (nt::element_list_t<nt::type_type_t<Outer>>{} + ...);
        }

        // This takes a forwarding tuple as a parameter. The forwarding tuple only
        // contains references, so it should just be taken by value.
        template<class T, class... Outer, class... Inner>
        constexpr auto tuple_cat(
                T tup,
                nt::TypeList<Outer...>,
                nt::TypeList<Inner...>
        ) -> Tuple<nt::type_type_t<Inner>...> {
            return {std::forward<nt::type_type_t<Outer>>(tup[Tag<std::decay_t<Outer>::INDEX>{}])
                    [Tag<std::decay_t<Inner>::INDEX>{}]...};
        }
    }

    template <typename... T>
    constexpr auto tuple_cat(T&&... ts) {
        if constexpr (sizeof...(T) == 0) {
            return Tuple<>();
        } else {
            using big_tuple = Tuple<T&&...>;
            using outer_elements = nt::element_list_t<big_tuple>;
            constexpr auto outer = guts::tuple_get_outer_elements(outer_elements{});
            constexpr auto inner = guts::tuple_get_inner_elements(outer_elements{});
            return guts::tuple_cat(big_tuple{std::forward<T>(ts)...}, outer, inner);
        }
    }

    /// Filters the tuple based on a predicate.
    /// The predicate should be a constant expression; if it returns true, keep the tuple argument, otherwise discard.
    /// Transform is an optional operator used to transform the argument before saving it into the new tuple.
    /// FIXME T will be S if tuple is moved and contains S, otherwise T is S&
    /// FIXME https://godbolt.org/z/fx3Kh3rs8
    template<typename Predicate, typename Transform, typename T> requires nt::is_tuple_v<T>
    constexpr auto tuple_filter(T&& tuple) {
        return apply(
                []<typename F, typename...R>(F&& first, R&& ... rest) {
                    auto filtered_rest = [&rest...] {
                        if constexpr (sizeof...(rest)) {
                            return tuple_filter<Predicate, Transform>(forward_as_tuple(std::forward<R>(rest)..., 0));
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

    template<size_t I, typename... T>
    struct tuple_element<I, noa::Tuple<T...>> {
        using type = decltype(noa::Tuple<T...>::declval(noa::Tag<I>()))::type;
    };

    template<size_t I, typename... T>
    struct tuple_element<I, const noa::Tuple<T...>> {
        using type = const decltype(noa::Tuple<T...>::declval(noa::Tag<I>()))::type;
    };
}

namespace noa::traits {
    template<>               struct proclaim_is_empty_tuple<noa::Tuple<>> : std::true_type {};
    template<typename... Ts> struct proclaim_is_tuple<noa::Tuple<Ts...>> : std::true_type {};
    template<typename... Ts> struct proclaim_is_tuple_of_accessor<noa::Tuple<Ts...>> : std::bool_constant<nt::are_accessor<Ts...>::value> {};
    template<typename... Ts> struct proclaim_is_tuple_of_accessor_pure<noa::Tuple<Ts...>> : std::bool_constant<nt::are_accessor_pure<Ts...>::value> {};
    template<typename... Ts> struct proclaim_is_tuple_of_accessor_reference<noa::Tuple<Ts...>> : std::bool_constant<nt::are_accessor_reference<Ts...>::value> {};
    template<typename... Ts> struct proclaim_is_tuple_of_accessor_value<noa::Tuple<Ts...>> : std::bool_constant<nt::are_accessor_value<Ts...>::value> {};
    template<typename... Ts> struct proclaim_is_tuple_of_accessor_or_empty<noa::Tuple<Ts...>> : std::bool_constant<nt::are_accessor<Ts...>::value> {};
    template<>               struct proclaim_is_tuple_of_accessor_or_empty<noa::Tuple<>> : std::true_type {};
    template<size_t N, typename... Ts> struct proclaim_is_tuple_of_accessor_ndim<N, noa::Tuple<Ts...>> : std::bool_constant<nt::are_accessor_nd<N, Ts...>::value> {};
}

#if defined(NOA_COMPILER_GCC) || defined(NOA_COMPILER_CLANG)
    #pragma GCC diagnostic pop
#elif defined(NOA_COMPILER_MSVC)
    #pragma warning(pop)
#endif

#ifdef NOA_IS_OFFLINE
namespace fmt {
    template<typename... T>
    struct formatter<noa::Tuple<T...>> {
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
