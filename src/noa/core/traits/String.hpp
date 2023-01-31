#pragma once

#include <type_traits>
#include <string>
#include <string_view>

#include "noa/core/traits/Utilities.hpp"

namespace noa::traits {
    // std::string(_view)
    template<typename> struct proclaim_is_string : std::false_type {};
    template<> struct proclaim_is_string<std::string> : std::true_type {};
    template<> struct proclaim_is_string<std::string_view> : std::true_type {};
    template<typename T> using is_string = std::bool_constant<proclaim_is_string<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_string_v = is_string<T>::value;
}
