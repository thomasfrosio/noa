#pragma once

#include <type_traits>
#include "noa/common/traits/Utilities.h"
#include "noa/common/traits/Numerics.h"

namespace noa::traits {
    template<typename T> struct proclaim_is_shape : std::false_type {};
    template<typename T> using is_shape = std::bool_constant<proclaim_is_shape<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_shape_v = is_shape<T>::value;
    template<typename T> constexpr bool is_shapeX_v = is_shape<T>::value;

    template<typename, typename> struct proclaim_is_shape_of_type : std::false_type {};
    template<typename T, typename V> using is_shape_of_type = std::bool_constant<proclaim_is_shape_of_type<remove_ref_cv_t<T>, V>::value>;
    template<typename T, typename V> constexpr bool is_shape_of_type_v = is_shape_of_type<T, V>::value;
    template<typename T, typename V> constexpr bool is_shapeT_v = is_shape_of_type<T, V>::value;

    template<typename, size_t> struct proclaim_is_shape_of_size : std::false_type {};
    template<typename T, size_t N> using is_shape_of_size = std::bool_constant<proclaim_is_shape_of_size<remove_ref_cv_t<T>, N>::value>;
    template<typename T, size_t N> constexpr bool is_shape_of_size_v = is_shape_of_size<T, N>::value;
    template<typename T, size_t N> constexpr bool is_shapeN_v = is_shape_of_size<T, N>::value;

    template<typename T> constexpr bool is_shape1_v = is_shapeN_v<T, 1>;
    template<typename T> constexpr bool is_shape2_v = is_shapeN_v<T, 2>;
    template<typename T> constexpr bool is_shape3_v = is_shapeN_v<T, 3>;
    template<typename T> constexpr bool is_shape4_v = is_shapeN_v<T, 4>;
}

namespace noa::traits {
    template<typename T> struct proclaim_is_strides : std::false_type {};
    template<typename T> using is_strides = std::bool_constant<proclaim_is_strides<remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_strides_v = is_strides<T>::value;
    template<typename T> constexpr bool is_stridesX_v = is_strides<T>::value;

    template<typename, typename> struct proclaim_is_strides_of_type : std::false_type {};
    template<typename T, typename V> using is_strides_of_type = std::bool_constant<proclaim_is_strides_of_type<remove_ref_cv_t<T>, V>::value>;
    template<typename T, typename V> constexpr bool is_strides_of_type_v = is_strides_of_type<T, V>::value;
    template<typename T, typename V> constexpr bool is_stridesT_v = is_strides_of_type<T, V>::value;

    template<typename, size_t> struct proclaim_is_strides_of_size : std::false_type {};
    template<typename T, size_t N> using is_strides_of_size = std::bool_constant<proclaim_is_strides_of_size<remove_ref_cv_t<T>, N>::value>;
    template<typename T, size_t N> constexpr bool is_strides_of_size_v = is_strides_of_size<T, N>::value;
    template<typename T, size_t N> constexpr bool is_stridesN_v = is_strides_of_size<T, N>::value;

    template<typename T> constexpr bool is_strides1_v = is_stridesN_v<T, 1>;
    template<typename T> constexpr bool is_strides2_v = is_stridesN_v<T, 2>;
    template<typename T> constexpr bool is_strides3_v = is_stridesN_v<T, 3>;
    template<typename T> constexpr bool is_strides4_v = is_stridesN_v<T, 4>;
}

namespace noa::traits {
    template<typename T> constexpr bool is_shape_or_strides_v = is_strides_v<T> || is_shape_v<T>;
    template<typename T, size_t N> constexpr bool is_shapeN_or_stridesN_v = is_stridesN_v<T, N> || is_shapeN_v<T, N>;
}
