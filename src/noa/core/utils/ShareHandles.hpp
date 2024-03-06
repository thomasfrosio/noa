#pragma once

#include "noa/core/Config.hpp"

#ifdef NOA_IS_OFFLINE
#include <memory>

namespace noa::traits {
    template<typename T> struct proclaim_is_unique_ptr : std::false_type {};
    template<typename T> struct proclaim_is_unique_ptr<std::unique_ptr<T>> : std::true_type {};
    template<typename T> constexpr bool is_unique_ptr_v = proclaim_is_unique_ptr<std::decay_t<T>>::value;

    template<typename T>
    constexpr bool is_shareable_v =
            requires(T t) { std::shared_ptr<const void>(t); } and
            not std::is_pointer_v<std::decay_t<T>> and
            not is_unique_ptr_v<T>;

    template<typename T>
    constexpr bool has_share_v = []() {
        if constexpr (requires(T t) { std::shared_ptr<const void>(t.share()); }) {
            using shared_t = std::decay_t<decltype(std::declval<T>().share())>;
            return not std::is_pointer_v<shared_t> and not is_unique_ptr_v<T>;
        } else {
            return false;
        }
    }();
}
#endif
