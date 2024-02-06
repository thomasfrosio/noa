#pragma once

#include "noa/core/types/Tuple.hpp"

namespace noa::guts {
    template<typename... Ts>
    struct AdaptorZip {
        static constexpr bool ZIP = true;
        Tuple<Ts&&...> tuple;
    };

    template<typename... Ts>
    struct AdaptorUnzip {
        static constexpr bool ZIP = false;
        Tuple<Ts&&...> tuple;
    };

    template<typename> struct proclaim_is_adaptor : std::false_type {};
    template<typename T> struct proclaim_is_adaptor<AdaptorZip<T>> : std::true_type {};
    template<typename T> struct proclaim_is_adaptor<AdaptorUnzip<T>> : std::true_type {};

    template<typename T> using is_adaptor = std::bool_constant<proclaim_is_adaptor<T>::value>;
    template<typename T> constexpr bool is_adaptor_v = is_adaptor<std::decay_t<T>>::value;
    template<typename... Ts> using are_adaptor = nt::bool_and<is_adaptor<Ts>::value...>::value;
    template<typename... Ts> constexpr bool are_adaptor_v = are_adaptor<std::decay_t<Ts>...>::value;
}

namespace noa {
    template<typename... T>
    constexpr auto zip(T&&... a) noexcept {
        return guts::AdaptorZip{.tuple=forward_as_tuple(std::forward<T>(a)...)};
    }

    template<typename... T>
    constexpr auto wrap(T&&... a) noexcept {
        return guts::AdaptorUnzip{.tuple=forward_as_tuple(std::forward<T>(a)...)};
    }
}
