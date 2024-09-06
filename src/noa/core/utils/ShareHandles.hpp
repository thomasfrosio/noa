#pragma once

#include "noa/core/Config.hpp"
#include "noa/core/Traits.hpp"

#ifdef NOA_IS_OFFLINE
#include <memory>

namespace noa::traits {
    NOA_GENERATE_PROCLAIM_FULL(unique_ptr);
    template<typename T, typename D> struct proclaim_is_unique_ptr<std::unique_ptr<T, D>> : std::true_type {};

    NOA_GENERATE_PROCLAIM_FULL(shared_ptr);
    template<typename T> struct proclaim_is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

    template<typename T> using is_smart_ptr = std::disjunction<is_unique_ptr<T>, is_shared_ptr<T>>;
    template<typename T> constexpr bool is_smart_ptr_v = is_smart_ptr<T>::value;
    template<typename... T> constexpr bool are_smart_ptr_v = std::conjunction_v<is_smart_ptr<T>...>;
    template<typename... T> concept smart_ptr = are_smart_ptr_v<T...>;

    template<typename T> using is_smart_ptr_decay = is_smart_ptr<std::decay_t<T>>;
    template<typename T> constexpr bool is_smart_ptr_decay_v = is_smart_ptr_decay<T>::value;
    template<typename... T> constexpr bool are_smart_ptr_decay_v = std::conjunction_v<is_smart_ptr_decay<T>...>;
    template<typename... T> concept smart_ptr_decay = are_smart_ptr_decay_v<T...>;

    template<typename T>
    concept shareable =
        requires(T t) { std::shared_ptr<const void>(t); } and
        not pointer<std::decay_t<T>> and
        not unique_ptr<std::decay_t<T>>;

    template<typename T>
    concept shareable_using_share = shareable<decltype(std::declval<T>().share())>;
}
#endif
