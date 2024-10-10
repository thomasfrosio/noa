#include <noa/core/types/Vec.hpp>

#include <catch2/catch.hpp>

using namespace ::noa::types;

TEMPLATE_TEST_CASE("core::traits:: Vec", "[noa][core]", u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, bool) {
    namespace nt = ::noa::traits;

    static_assert(nt::is_vec_v<Vec1<TestType>>);
    static_assert(nt::is_vec_v<const Vec2<TestType>>);
    static_assert(not nt::is_vec_v<Vec3<TestType>&>);
    static_assert(not nt::is_vec_v<Vec4<TestType>&&>);
    static_assert(not nt::is_vec_v<Vec2<TestType>*>);

    static_assert(nt::is_vec_of_size_v<Vec2<TestType>, 2>);
    static_assert(nt::is_vec_of_size_v<const Vec<TestType, 5>, 5>);
    static_assert(not nt::is_vec_of_size_v<Vec1<TestType>&, 1>);

    static_assert(nt::is_vec_of_type_v<Vec2<TestType>, TestType>);
    static_assert(nt::is_vec_of_type_v<const Vec<TestType, 5>, TestType>);
    static_assert(not nt::is_vec_of_type_v<Vec1<TestType>&, TestType>);
    static_assert(not nt::is_vec_of_type_v<Vec2<TestType>, const TestType>);
    static_assert(not nt::is_vec_of_type_v<const Vec<TestType, 5>, TestType&>);
    static_assert(not nt::is_vec_of_type_v<Vec1<TestType>, TestType*>);

    if constexpr (std::is_same_v<TestType, f32> or std::is_same_v<TestType, f64>) {
        static_assert(nt::is_vec_real_v<Vec1<TestType>>);
        static_assert(nt::is_vec_real_size_v<Vec2<TestType>, 2>);
        static_assert(nt::is_vec_real_size_v<Vec4<TestType>, 4>);

        static_assert(not nt::is_vec_real_size_v<Vec2<TestType>, 3>);
        static_assert(not nt::is_vec_real_size_v<Vec2<TestType>, 4>);
        static_assert(not nt::is_vec_real_size_v<Vec3<TestType>, 2>);
        static_assert(not nt::is_vec_real_size_v<Vec3<TestType>, 4>);
        static_assert(not nt::is_vec_integer_size_v<Vec4<TestType>, 4>);
        static_assert(not nt::is_vec_real_size_v<Vec4<TestType>, 3>);

    } else if constexpr (std::is_same_v<TestType, bool>) {
        static_assert(nt::is_vec_boolean_v<Vec1<TestType>>);
        static_assert(nt::is_vec_boolean_size_v<Vec2<TestType>, 2>);
        static_assert(nt::is_vec_boolean_size_v<Vec4<TestType>, 4>);
        static_assert(nt::is_vec_boolean_size_v<Vec4<TestType>, 4>);
        static_assert(nt::is_vec_integer_size_v<Vec3<TestType>, 3>);
        static_assert(nt::is_vec_uinteger_size_v<Vec3<TestType>, 3>);
        static_assert(not nt::is_vec_sinteger_size_v<Vec3<TestType>, 3>);

        static_assert(not nt::is_vec_boolean_size_v<Vec2<TestType>, 4>);
        static_assert(not nt::is_vec_boolean_size_v<Vec3<TestType>, 2>);
        static_assert(not nt::is_vec_boolean_size_v<Vec3<TestType>, 4>);

    } else if constexpr (std::is_unsigned_v<TestType>) {
        static_assert(nt::is_vec_uinteger_v<Vec1<TestType>>);
        static_assert(nt::is_vec_uinteger_size_v<Vec2<TestType>, 2>);
        static_assert(nt::is_vec_uinteger_size_v<Vec4<TestType>, 4>);
        static_assert(nt::is_vec_uinteger_size_v<Vec4<TestType>, 4>);

        static_assert(nt::is_vec_integer_v<Vec1<TestType>>);
        static_assert(nt::is_vec_integer_size_v<Vec2<TestType>, 2>);
        static_assert(nt::is_vec_integer_size_v<Vec4<TestType>, 4>);
        static_assert(nt::is_vec_integer_size_v<Vec4<TestType>, 4>);

        static_assert(not nt::is_vec_sinteger_v<Vec1<TestType>>);
        static_assert(not nt::is_vec_sinteger_size_v<Vec2<TestType>, 2>);
        static_assert(not nt::is_vec_sinteger_size_v<Vec4<TestType>, 4>);
        static_assert(not nt::is_vec_sinteger_size_v<Vec4<TestType>, 4>);

        static_assert(not nt::is_vec_integer_size_v<Vec2<TestType>, 3>);
        static_assert(not nt::is_vec_real_size_v<Vec4<TestType>, 4>);
        static_assert(not nt::is_vec_integer_size_v<Vec3<TestType>, 2>);
        static_assert(not nt::is_vec_integer_size_v<Vec3<TestType>, 4>);
        static_assert(not nt::is_vec_boolean_size_v<Vec4<TestType>, 4>);
        static_assert(not nt::is_vec_integer_size_v<Vec4<TestType>, 3>);

    } else {
        static_assert(nt::is_vec_sinteger_v<Vec1<TestType>>);
        static_assert(nt::is_vec_sinteger_size_v<Vec2<TestType>, 2>);
        static_assert(nt::is_vec_sinteger_size_v<Vec4<TestType>, 4>);
        static_assert(nt::is_vec_sinteger_size_v<Vec4<TestType>, 4>);

        static_assert(nt::is_vec_integer_v<Vec1<TestType>>);
        static_assert(nt::is_vec_integer_size_v<Vec2<TestType>, 2>);
        static_assert(nt::is_vec_integer_size_v<Vec4<TestType>, 4>);
        static_assert(nt::is_vec_integer_size_v<Vec4<TestType>, 4>);

        static_assert(not nt::is_vec_uinteger_v<Vec1<TestType>>);
        static_assert(not nt::is_vec_uinteger_size_v<Vec2<TestType>, 2>);
        static_assert(not nt::is_vec_uinteger_size_v<Vec4<TestType>, 4>);
        static_assert(not nt::is_vec_uinteger_size_v<Vec4<TestType>, 4>);

        static_assert(not nt::is_vec_integer_size_v<Vec2<TestType>, 3>);
        static_assert(not nt::is_vec_real_size_v<Vec4<TestType>, 4>);
        static_assert(not nt::is_vec_integer_size_v<Vec3<TestType>, 2>);
        static_assert(not nt::is_vec_integer_size_v<Vec3<TestType>, 4>);
        static_assert(not nt::is_vec_boolean_size_v<Vec4<TestType>, 4>);
        static_assert(not nt::is_vec_integer_size_v<Vec4<TestType>, 3>);
    }
}
