#include <noa/common/Types.h>

#include <catch2/catch.hpp>

using namespace ::noa;

TEMPLATE_TEST_CASE("core::traits:: Vec", "[noa][core]",
                   u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, bool) {
    using namespace ::noa::traits;

    static_assert(is_vec_v<Vec1<TestType>>);
    static_assert(is_vec_v<const Vec2<TestType>>);
    static_assert(is_vec_v<Vec3<TestType>&>);
    static_assert(is_vec_v<Vec4<TestType>&&>);
    static_assert(!is_vec_v<Vec2<TestType>*>);

    static_assert(is_vecN_v<Vec2<TestType>, 2>);
    static_assert(is_vecN_v<const Vec<TestType, 5>, 5>);
    static_assert(is_vecN_v<Vec1<TestType>&, 1>);

    static_assert(is_vecT_v<Vec2<TestType>, TestType>);
    static_assert(is_vecT_v<const Vec<TestType, 5>, TestType>);
    static_assert(is_vecT_v<Vec1<TestType>&, TestType>);
    static_assert(!is_vecT_v<Vec2<TestType>, const TestType>);
    static_assert(!is_vecT_v<const Vec<TestType, 5>, TestType&>);
    static_assert(!is_vecT_v<Vec1<TestType>&, TestType*>);

    if constexpr (std::is_same_v<TestType, f32> || std::is_same_v<TestType, f64>) {
        static_assert(is_realX_v<Vec1<TestType>>);
        static_assert(is_real2_v<Vec2<TestType>>);
        static_assert(is_real4_v<Vec4<TestType>>);
        static_assert(is_realN_v<Vec4<TestType>, 4>);

        static_assert(!is_real3_v<Vec2<TestType>>);
        static_assert(!is_real4_v<Vec2<TestType>>);
        static_assert(!is_real2_v<Vec3<TestType>>);
        static_assert(!is_real4_v<Vec3<TestType>>);
        static_assert(!is_int4_v<Vec4<TestType>>);
        static_assert(!is_real3_v<Vec4<TestType>>);

    } else if constexpr (std::is_same_v<TestType, bool>) {
        static_assert(is_boolX_v<Vec1<TestType>>);
        static_assert(is_bool2_v<Vec2<TestType>>);
        static_assert(is_bool4_v<Vec4<TestType>>);
        static_assert(is_boolN_v<Vec4<TestType>, 4>);
        static_assert(is_int3_v<Vec3<TestType>>);
        static_assert(is_uint3_v<Vec3<TestType>>);
        static_assert(!is_sint3_v<Vec3<TestType>>);

        static_assert(!is_bool4_v<Vec2<TestType>>);
        static_assert(!is_bool2_v<Vec3<TestType>>);
        static_assert(!is_bool4_v<Vec3<TestType>>);

    } else if constexpr (std::is_unsigned_v<TestType>) {
        static_assert(is_uintX_v<Vec1<TestType>>);
        static_assert(is_uint2_v<Vec2<TestType>>);
        static_assert(is_uint4_v<Vec4<TestType>>);
        static_assert(is_uintN_v<Vec4<TestType>, 4>);

        static_assert(is_intX_v<Vec1<TestType>>);
        static_assert(is_int2_v<Vec2<TestType>>);
        static_assert(is_int4_v<Vec4<TestType>>);
        static_assert(is_intN_v<Vec4<TestType>, 4>);

        static_assert(!is_sintX_v<Vec1<TestType>>);
        static_assert(!is_sint2_v<Vec2<TestType>>);
        static_assert(!is_sint4_v<Vec4<TestType>>);
        static_assert(!is_sintN_v<Vec4<TestType>, 4>);

        static_assert(!is_int3_v<Vec2<TestType>>);
        static_assert(!is_real4_v<Vec4<TestType>>);
        static_assert(!is_int2_v<Vec3<TestType>>);
        static_assert(!is_int4_v<Vec3<TestType>>);
        static_assert(!is_bool4_v<Vec4<TestType>>);
        static_assert(!is_int3_v<Vec4<TestType>>);

    } else {
        static_assert(is_sintX_v<Vec1<TestType>>);
        static_assert(is_sint2_v<Vec2<TestType>>);
        static_assert(is_sint4_v<Vec4<TestType>>);
        static_assert(is_sintN_v<Vec4<TestType>, 4>);

        static_assert(is_intX_v<Vec1<TestType>>);
        static_assert(is_int2_v<Vec2<TestType>>);
        static_assert(is_int4_v<Vec4<TestType>>);
        static_assert(is_intN_v<Vec4<TestType>, 4>);

        static_assert(!is_uintX_v<Vec1<TestType>>);
        static_assert(!is_uint2_v<Vec2<TestType>>);
        static_assert(!is_uint4_v<Vec4<TestType>>);
        static_assert(!is_uintN_v<Vec4<TestType>, 4>);

        static_assert(!is_int3_v<Vec2<TestType>>);
        static_assert(!is_real4_v<Vec4<TestType>>);
        static_assert(!is_int2_v<Vec3<TestType>>);
        static_assert(!is_int4_v<Vec3<TestType>>);
        static_assert(!is_bool4_v<Vec4<TestType>>);
        static_assert(!is_int3_v<Vec4<TestType>>);
    }
}
