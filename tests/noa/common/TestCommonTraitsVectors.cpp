#include <noa/common/types/BoolX.h>
#include <noa/common/types/IntX.h>
#include <noa/common/types/FloatX.h>

#include <catch2/catch.hpp>

using namespace ::noa;

#define REQUIRE_FOR_ALL_TYPES(type_trait, type)               \
REQUIRE((type_trait<type>));                                  \
REQUIRE((type_trait<std::add_const_t<type>>));                \
REQUIRE((type_trait<std::add_volatile_t<type>>));             \
REQUIRE((type_trait<std::add_cv_t<type>>));                   \
REQUIRE((type_trait<std::add_lvalue_reference_t<type>>));     \
REQUIRE((type_trait<std::add_rvalue_reference_t<type>>))

#define REQUIRE_FALSE_FOR_ALL_TYPES(type_trait, type)               \
REQUIRE_FALSE((type_trait<type>));                                  \
REQUIRE_FALSE((type_trait<std::add_const_t<type>>));                \
REQUIRE_FALSE((type_trait<std::add_volatile_t<type>>));             \
REQUIRE_FALSE((type_trait<std::add_cv_t<type>>));                   \
REQUIRE_FALSE((type_trait<std::add_lvalue_reference_t<type>>));     \
REQUIRE_FALSE((type_trait<std::add_rvalue_reference_t<type>>))

#define REQUIRE_FOR_ALL_TYPES_BOOL(type_traits)              \
REQUIRE_FOR_ALL_TYPES(type_traits, ::noa::Bool2);  \
REQUIRE_FOR_ALL_TYPES(type_traits, ::noa::Bool3);  \
REQUIRE_FOR_ALL_TYPES(type_traits, ::noa::Bool4)

#define REQUIRE_FALSE_FOR_ALL_TYPES_BOOL(type_traits)                \
REQUIRE_FALSE_FOR_ALL_TYPES(type_traits, ::noa::Bool2);    \
REQUIRE_FALSE_FOR_ALL_TYPES(type_traits, ::noa::Bool3);    \
REQUIRE_FALSE_FOR_ALL_TYPES(type_traits, ::noa::Bool4)

#define REQUIRE_FOR_ALL_TYPES_INT(type_traits)              \
REQUIRE_FOR_ALL_TYPES(type_traits, ::noa::Int2<TestType>);  \
REQUIRE_FOR_ALL_TYPES(type_traits, ::noa::Int3<TestType>);  \
REQUIRE_FOR_ALL_TYPES(type_traits, ::noa::Int4<TestType>)

#define REQUIRE_FALSE_FOR_ALL_TYPES_INT(type_traits)                \
REQUIRE_FALSE_FOR_ALL_TYPES(type_traits, ::noa::Int2<TestType>);    \
REQUIRE_FALSE_FOR_ALL_TYPES(type_traits, ::noa::Int3<TestType>);    \
REQUIRE_FALSE_FOR_ALL_TYPES(type_traits, ::noa::Int4<TestType>)

#define REQUIRE_FOR_ALL_TYPES_FLOAT(type_traits)                \
REQUIRE_FOR_ALL_TYPES(type_traits, ::noa::Float2<TestType>);    \
REQUIRE_FOR_ALL_TYPES(type_traits, ::noa::Float3<TestType>);    \
REQUIRE_FOR_ALL_TYPES(type_traits, ::noa::Float4<TestType>)

#define REQUIRE_FALSE_FOR_ALL_TYPES_FLOAT(type_traits)              \
REQUIRE_FALSE_FOR_ALL_TYPES(type_traits, ::noa::Float2<TestType>);  \
REQUIRE_FALSE_FOR_ALL_TYPES(type_traits, ::noa::Float3<TestType>);  \
REQUIRE_FALSE_FOR_ALL_TYPES(type_traits, ::noa::Float4<TestType>)

TEMPLATE_TEST_CASE("traits:: BoolX, IntX, FloatX", "[noa][common][traits]",
                   uint8_t, unsigned short, unsigned int, unsigned long, unsigned long long,
                   int8_t, short, int, long, long long,
                   float, double,
                   bool) {
    using namespace ::noa::traits;

    if constexpr (std::is_same_v<TestType, float> || std::is_same_v<TestType, double>) {
        REQUIRE_FOR_ALL_TYPES_FLOAT(is_floatX_v);
        REQUIRE_FOR_ALL_TYPES(is_float2_v, Float2<TestType>);
        REQUIRE_FOR_ALL_TYPES(is_float3_v, Float3<TestType>);
        REQUIRE_FOR_ALL_TYPES(is_float4_v, Float4<TestType>);

        REQUIRE_FALSE(is_float3_v<Float2<TestType>>);
        REQUIRE_FALSE(is_float4_v<Float2<TestType>>);
        REQUIRE_FALSE(is_float2_v<Float3<TestType>>);
        REQUIRE_FALSE(is_float4_v<Float3<TestType>>);
        REQUIRE_FALSE(is_float2_v<Float4<TestType>>);
        REQUIRE_FALSE(is_float3_v<Float4<TestType>>);

        REQUIRE_FALSE_FOR_ALL_TYPES_FLOAT(is_intX_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_FLOAT(is_boolX_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_FLOAT(is_int2_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_FLOAT(is_int3_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_FLOAT(is_int4_v);

    } else if constexpr (std::is_same_v<TestType, bool>) {
        REQUIRE_FOR_ALL_TYPES_BOOL(noa::traits::is_boolX_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_BOOL(noa::traits::is_floatX_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_BOOL(noa::traits::is_intX_v);

    } else if constexpr (std::is_unsigned_v<TestType>) {
        REQUIRE_FOR_ALL_TYPES_INT(is_intX_v);
        REQUIRE_FOR_ALL_TYPES_INT(is_uintX_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_INT(is_boolX_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_INT(is_floatX_v);

        REQUIRE_FALSE(is_uint3_v<Int2<TestType>>);
        REQUIRE_FALSE(is_uint4_v<Int2<TestType>>);
        REQUIRE_FALSE(is_uint2_v<Int3<TestType>>);
        REQUIRE_FALSE(is_uint4_v<Int3<TestType>>);
        REQUIRE_FALSE(is_uint2_v<Int4<TestType>>);
        REQUIRE_FALSE(is_uint3_v<Int4<TestType>>);

        REQUIRE_FALSE_FOR_ALL_TYPES_INT(is_float2_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_INT(is_float3_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_INT(is_float4_v);

    } else {
        REQUIRE_FOR_ALL_TYPES_INT(is_intX_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_INT(is_uintX_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_INT(is_boolX_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_INT(is_floatX_v);

        REQUIRE_FALSE(is_int3_v<Int2<TestType>>);
        REQUIRE_FALSE(is_int4_v<Int2<TestType>>);
        REQUIRE_FALSE(is_int2_v<Int3<TestType>>);
        REQUIRE_FALSE(is_int4_v<Int3<TestType>>);
        REQUIRE_FALSE(is_int2_v<Int4<TestType>>);
        REQUIRE_FALSE(is_int3_v<Int4<TestType>>);

        REQUIRE_FALSE_FOR_ALL_TYPES_INT(is_float2_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_INT(is_float3_v);
        REQUIRE_FALSE_FOR_ALL_TYPES_INT(is_float4_v);
    }
}
