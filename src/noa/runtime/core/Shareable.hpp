#pragma once

#include <memory>
#include "noa/base/Traits.hpp"

namespace noa::traits {
    NOA_GENERATE_PROCLAIM_FULL(unique_ptr);
    template<typename T, typename D> struct proclaim_is_unique_ptr<std::unique_ptr<T, D>> : std::true_type {};

    NOA_GENERATE_PROCLAIM_FULL(shared_ptr);
    template<typename T> struct proclaim_is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

    template<typename T>
    concept shareable =
        requires(T t) { std::shared_ptr<const void>(t); } and
        not pointer<std::decay_t<T>> and
        not unique_ptr<std::decay_t<T>>;

    template<typename T>
    concept shareable_using_share = shareable<decltype(std::declval<T>().share())>;
}
