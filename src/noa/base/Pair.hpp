#pragma once

#include <cstddef>
#include <utility>
#include <functional> // reference_wrapper

#include "noa/base/Config.hpp"
#include "noa/base/Traits.hpp"

namespace noa::inline types {
    template<usize I>
    using Tag = std::integral_constant<usize, I>;
}

namespace noa::traits {
    template<typename T> struct unwrap_reference { using type = T; };
    template<typename U> struct unwrap_reference<std::reference_wrapper<U>> { using type = U&; };
    template<typename T> struct unwrap_reference_decay : unwrap_reference<std::decay_t<T>> {};
    template<typename T> using unwrap_reference_decay_t = unwrap_reference_decay<T>::type;

    template<typename Tup> using element_list_t = std::decay_t<Tup>::element_list;
    template<typename Tup> using type_list_t = std::decay_t<Tup>::type_list;
    template<typename Tup> using index_list_t = std::decay_t<Tup>::index_list;
}

namespace noa::inline types {
    template<typename First, typename Second>
    struct Pair {
        constexpr static usize SIZE = 2;
        constexpr static isize SSIZE = 2;
        constexpr static bool nothrow_swappable =
            std::is_nothrow_swappable_v<First> and
            std::is_nothrow_swappable_v<Second>;
        using type_list = nt::TypeList<First, Second>;
        using decayed_type_list = nt::TypeList<std::decay_t<First>, std::decay_t<Second>>;
        using index_list = std::index_sequence<0, 1>;
        using first_type = First;
        using second_type = Second;

        NOA_NO_UNIQUE_ADDRESS First first;
        NOA_NO_UNIQUE_ADDRESS Second second;

        template<usize I>
        constexpr auto operator[](Tag<I>) & noexcept -> std::conditional_t<I == 0, First, Second>& {
            if constexpr (I == 0)
                return first;
            else if constexpr (I == 1)
                return second;
            else
                static_assert(I < 2);
        }

        template<usize I>
        constexpr auto operator[](Tag<I>) const& noexcept -> std::conditional_t<I == 0, First, Second> const& {
            if constexpr (I == 0)
                return first;
            else if constexpr (I == 1)
                return second;
            else
                static_assert(I < 2);
        }

        template<usize I>
        constexpr auto operator[](Tag<I>) && noexcept -> std::conditional_t<I == 0, First, Second>&& {
            if constexpr (I == 0)
                return std::forward<Pair>(*this).first;
            else if constexpr (I == 1)
                return std::forward<Pair>(*this).second;
            else
                static_assert(I < 2);
        }

        template<usize I>
        constexpr auto operator[](Tag<I>) const && noexcept -> std::conditional_t<I == 0, First, Second> const && {
            if constexpr (I == 0)
                return std::forward<const Pair>(*this).first;
            else if constexpr (I == 1)
                return std::forward<const Pair>(*this).second;
            else
                static_assert(I < 2);
        }

        void swap(Pair& other) noexcept(nothrow_swappable) {
            using std::swap;
            swap(first, other.first);
            swap(second, other.second);
        }

        template<typename T> requires (not nt::almost_same_as<Pair, T>)
        constexpr auto& operator=(T&& tup) { // TODO noexcept
            static_assert(std::decay_t<T>::SIZE == 2);
            first = std::forward<T>(tup)[Tag<0>{}];
            second = std::forward<T>(tup)[Tag<1>{}];
            return *this;
        }

        template<typename F2, typename S2>
        constexpr auto& assign(F2&& f, S2&& s) {
            first = std::forward<F2>(f);
            second = std::forward<S2>(s);
            return *this;
        }
    };

    template<typename A, typename B>
    Pair(A, B) -> Pair<nt::unwrap_reference_decay_t<A>, nt::unwrap_reference_decay_t<B>>;
}

namespace noa {
    template<usize I, typename Tup>
    requires (nt::tuple<std::decay_t<Tup>> or nt::pair<std::decay_t<Tup>>)
    constexpr auto get(Tup&& tup) -> decltype(auto) {
        return std::forward<Tup>(tup)[Tag<I>{}];
    }

    template<typename A, typename B>
    constexpr void swap(Pair<A, B>& a, Pair<A, B>& b) noexcept(Pair<A, B>::nothrow_swappable) {
        a.swap(b);
    }
}

namespace noa::traits {
    template<typename T, typename U>
    struct proclaim_is_pair<noa::Pair<T, U>> : std::true_type {};
}
