#pragma once

#include <type_traits>
#include "noa/common/traits/Utilities.h"

namespace noa::traits {
    template<typename T> struct proclaim_is_bool2 : std::false_type {};
    template<typename T> using is_bool2 = std::bool_constant<proclaim_is_bool2<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_bool2_v = is_bool2<T>::value;

    template<typename T> struct proclaim_is_bool3 : std::false_type {};
    template<typename T> using is_bool3 = std::bool_constant<proclaim_is_bool3<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_bool3_v = is_bool3<T>::value;

    template<typename T> struct proclaim_is_bool4 : std::false_type {};
    template<typename T> using is_bool4 = std::bool_constant<proclaim_is_bool4<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_bool4_v = is_bool4<T>::value;

    template<typename T> using is_boolX = std::bool_constant<is_bool2_v<T> || is_bool3_v<T> || is_bool4_v<T>>;
    /// One of: bool2_t, bool3_t, bool4_t. \c remove_ref_cv_t is applied to T.
    template<typename T> constexpr bool is_boolX_v = is_boolX<T>::value;
}

namespace noa::traits {
    template<typename T> struct proclaim_is_int2 : std::false_type {};
    template<typename T> using is_int2 = std::bool_constant<proclaim_is_int2<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_int2_v = is_int2<T>::value;

    template<typename T> struct proclaim_is_int3 : std::false_type {};
    template<typename T> using is_int3 = std::bool_constant<proclaim_is_int3<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_int3_v = is_int3<T>::value;

    template<typename T> struct proclaim_is_int4 : std::false_type {};
    template<typename T> using is_int4 = std::bool_constant<proclaim_is_int4<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_int4_v = is_int4<T>::value;

    template<typename T> using is_intX = std::bool_constant<is_int2_v<T> || is_int3_v<T> || is_int4_v<T>>;
    /// One of: int2_t, int3_t, int4_t, long2_t, long3_t, long4_t. \c remove_ref_cv_t is applied to T.
    template<typename T> constexpr bool is_intX_v = is_intX<T>::value;
}

namespace noa::traits {
    template<typename T> struct proclaim_is_uint2 : std::false_type {};
    template<typename T> using is_uint2 = std::bool_constant<proclaim_is_uint2<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_uint2_v = is_uint2<T>::value;

    template<typename T> struct proclaim_is_uint3 : std::false_type {};
    template<typename T> using is_uint3 = std::bool_constant<proclaim_is_uint3<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_uint3_v = is_uint3<T>::value;

    template<typename T> struct proclaim_is_uint4 : std::false_type {};
    template<typename T> using is_uint4 = std::bool_constant<proclaim_is_uint4<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_uint4_v = is_uint4<T>::value;

    template<typename T> using is_uintX = std::bool_constant<is_uint2_v<T> || is_uint3_v<T> || is_uint4_v<T>>;
    /// One of: uint2_t, uint3_t, uint4_t, ulong2_t, ulong3_t, ulong4_t. \c remove_ref_cv_t is applied to T.
    template<typename T> constexpr bool is_uintX_v = is_uintX<T>::value;
}

namespace noa::traits {
    template<typename T> struct proclaim_is_float2 : std::false_type {};
    template<typename T> using is_float2 = std::bool_constant<proclaim_is_float2<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_float2_v = is_float2<T>::value;

    template<typename T> struct proclaim_is_float3 : std::false_type {};
    template<typename T> using is_float3 = std::bool_constant<proclaim_is_float3<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_float3_v = is_float3<T>::value;

    template<typename T> struct proclaim_is_float4 : std::false_type {};
    template<typename T> using is_float4 = std::bool_constant<proclaim_is_float4<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_float4_v = is_float4<T>::value;

    template<typename T> using is_floatX = std::bool_constant<is_float2_v<T> || is_float3_v<T> || is_float4_v<T>>;
    /// One of: float2_t, float3_t, float4_t. \c remove_ref_cv_t is applied to T.
    template<typename T> constexpr bool is_floatX_v = is_floatX<T>::value;
}

namespace noa::traits {
    template<typename> struct proclaim_is_float22 : std::false_type {};
    template<typename T> using is_float22 = std::bool_constant<proclaim_is_float22<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_float22_v = is_float22<T>::value;

    template<typename> struct proclaim_is_float23 : std::false_type {};
    template<typename T> using is_float23 = std::bool_constant<proclaim_is_float23<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_float23_v = is_float23<T>::value;

    template<typename> struct proclaim_is_float33 : std::false_type {};
    template<typename T> using is_float33 = std::bool_constant<proclaim_is_float33<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_float33_v = is_float33<T>::value;

    template<typename> struct proclaim_is_float34 : std::false_type {};
    template<typename T> using is_float34 = std::bool_constant<proclaim_is_float34<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_float34_v = is_float34<T>::value;

    template<typename> struct proclaim_is_float44 : std::false_type {};
    template<typename T> using is_float44 = std::bool_constant<proclaim_is_float44<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_float44_v = is_float44<T>::value;

    template<typename T> using is_floatXX = std::bool_constant<is_float22_v<T> || is_float23_v<T> || is_float33_v<T> || is_float34_v<T> || is_float44_v<T>>;
    /// One of: float22_t, float23_t, float33_t, float34_t, float44_t. \c remove_ref_cv_t is applied to T.
    template<typename T> constexpr bool is_floatXX_v = is_floatXX<T>::value;

    // TODO Replace floatXX_t to matXX_t
    template<typename> struct proclaim_is_mat22 : std::false_type {};
    template<typename T> using is_mat22 = std::bool_constant<proclaim_is_mat22<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_mat22_v = is_mat22<T>::value;

    template<typename> struct proclaim_is_mat23 : std::false_type {};
    template<typename T> using is_mat23 = std::bool_constant<proclaim_is_mat23<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_mat23_v = is_mat23<T>::value;

    template<typename> struct proclaim_is_mat33 : std::false_type {};
    template<typename T> using is_mat33 = std::bool_constant<proclaim_is_mat33<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_mat33_v = is_mat33<T>::value;

    template<typename> struct proclaim_is_mat34 : std::false_type {};
    template<typename T> using is_mat34 = std::bool_constant<proclaim_is_mat34<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_mat34_v = is_mat34<T>::value;

    template<typename> struct proclaim_is_mat44 : std::false_type {};
    template<typename T> using is_mat44 = std::bool_constant<proclaim_is_mat44<traits::remove_ref_cv_t<T>>::value>;
    template<typename T> constexpr bool is_mat44_v = is_mat44<T>::value;

    template<typename T> using is_matXX = std::bool_constant<is_mat22_v<T> || is_mat23_v<T> || is_mat33_v<T> || is_mat34_v<T> || is_mat44_v<T>>;
    /// One of: mat22_t, mat23_t, mat33_t, mat34_t, mat44_t. \c remove_ref_cv_t is applied to T.
    template<typename T> constexpr bool is_matXX_v = is_matXX<T>::value;
}
