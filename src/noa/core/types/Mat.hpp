#pragma once

#include "noa/core/types/Mat22.hpp"
#include "noa/core/types/Mat23.hpp"
#include "noa/core/types/Mat33.hpp"
#include "noa/core/types/Mat34.hpp"
#include "noa/core/types/Mat44.hpp"

namespace noa::traits {
    template<typename T> using is_matXX = std::bool_constant<is_mat22_v<T> || is_mat23_v<T> || is_mat33_v<T> || is_mat34_v<T> || is_mat44_v<T>>;
    template<typename T> constexpr bool is_matXX_v = is_matXX<T>::value;
}
