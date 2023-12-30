#pragma once

#include "noa/core/types/Tuple.hpp"

namespace noa::guts {
    template<typename... Ts>
    struct EwiseAdaptorUnzip {
        static constexpr bool ZIP = false;
        Tuple<Ts&& ...> tuple;
    };

    template<typename... Ts>
    struct EwiseAdaptorZip {
        static constexpr bool ZIP = true;
        Tuple<Ts&& ...> tuple;
    };

    struct EwiseAdaptorTags {
        bool zip_inputs{};
        bool zip_outputs{};
    };

    template<typename Input, typename Output>
    constexpr auto ewise_adaptor_tags() -> EwiseAdaptorTags { return {std::decay_t<Input>::ZIP, std::decay_t<Output>::ZIP}; }

    template<typename> struct proclaim_is_ewise_adaptor : std::false_type {};
    template<typename T> struct proclaim_is_ewise_adaptor<EwiseAdaptorUnzip<T>> : std::true_type {};
    template<typename T> struct proclaim_is_ewise_adaptor<EwiseAdaptorZip<T>> : std::true_type {};

    template<typename T> using is_ewise_adaptor = std::bool_constant<proclaim_is_ewise_adaptor<T>::value>;
    template<typename T> constexpr bool is_ewise_adaptor_v = is_ewise_adaptor<std::decay_t<T>>::value;
    template<typename... Ts> using are_ewise_adaptor = nt::bool_and<is_ewise_adaptor<Ts>::value...>::value;
    template<typename... Ts> constexpr bool are_ewise_adaptor_v = are_ewise_adaptor<std::decay_t<Ts>...>::value;
}

namespace noa {
    /// Similar to zip(...) but return a Wrapper instead of a Tuple.
    /// This is used to track that the underlying Tuple was made from the wrap(...) function,
    /// so that other functions can trigger a specific behavior. For instance, the core
    /// ewise functions will unwrap the elements from Wrapper, but will leave Tuple as is.
    template<typename... T>
    constexpr auto wrap(T&& ... a) noexcept {
        return guts::EwiseAdaptorUnzip<T&&...>{std::forward<T>(a)...};
    }
    template<typename... T>
    constexpr auto zip(T&& ... a) noexcept {
        return guts::EwiseAdaptorZip<T&&...>{std::forward<T>(a)...};
    }
}
