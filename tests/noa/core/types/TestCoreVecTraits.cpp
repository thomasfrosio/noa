#include <noa/core/Types.hpp>

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

    static_assert(is_vec_of_size_v<Vec2<TestType>, 2>);
    static_assert(is_vec_of_size_v<const Vec<TestType, 5>, 5>);
    static_assert(is_vec_of_size_v<Vec1<TestType>&, 1>);

    static_assert(is_vec_of_type_v<Vec2<TestType>, TestType>);
    static_assert(is_vec_of_type_v<const Vec<TestType, 5>, TestType>);
    static_assert(is_vec_of_type_v<Vec1<TestType>&, TestType>);
    static_assert(!is_vec_of_type_v<Vec2<TestType>, const TestType>);
    static_assert(!is_vec_of_type_v<const Vec<TestType, 5>, TestType&>);
    static_assert(!is_vec_of_type_v<Vec1<TestType>&, TestType*>);

    if constexpr (std::is_same_v<TestType, f32> || std::is_same_v<TestType, f64>) {
        static_assert(is_vec_real_v<Vec1<TestType>>);
        static_assert(is_vec_real_size_v<Vec2<TestType>, 2>);
        static_assert(is_vec_real_size_v<Vec4<TestType>, 4>);

        static_assert(!is_vec_real_size_v<Vec2<TestType>, 3>);
        static_assert(!is_vec_real_size_v<Vec2<TestType>, 4>);
        static_assert(!is_vec_real_size_v<Vec3<TestType>, 2>);
        static_assert(!is_vec_real_size_v<Vec3<TestType>, 4>);
        static_assert(!is_vec_int_size_v<Vec4<TestType>, 4>);
        static_assert(!is_vec_real_size_v<Vec4<TestType>, 3>);

    } else if constexpr (std::is_same_v<TestType, bool>) {
        static_assert(is_vec_bool_v<Vec1<TestType>>);
        static_assert(is_vec_bool_size_v<Vec2<TestType>, 2>);
        static_assert(is_vec_bool_size_v<Vec4<TestType>, 4>);
        static_assert(is_vec_bool_size_v<Vec4<TestType>, 4>);
        static_assert(is_vec_int_size_v<Vec3<TestType>, 3>);
        static_assert(is_vec_uint_size_v<Vec3<TestType>, 3>);
        static_assert(!is_vec_sint_size_v<Vec3<TestType>, 3>);

        static_assert(!is_vec_bool_size_v<Vec2<TestType>, 4>);
        static_assert(!is_vec_bool_size_v<Vec3<TestType>, 2>);
        static_assert(!is_vec_bool_size_v<Vec3<TestType>, 4>);

    } else if constexpr (std::is_unsigned_v<TestType>) {
        static_assert(is_vec_uint_v<Vec1<TestType>>);
        static_assert(is_vec_uint_size_v<Vec2<TestType>, 2>);
        static_assert(is_vec_uint_size_v<Vec4<TestType>, 4>);
        static_assert(is_vec_uint_size_v<Vec4<TestType>, 4>);

        static_assert(is_vec_int_v<Vec1<TestType>>);
        static_assert(is_vec_int_size_v<Vec2<TestType>, 2>);
        static_assert(is_vec_int_size_v<Vec4<TestType>, 4>);
        static_assert(is_vec_int_size_v<Vec4<TestType>, 4>);

        static_assert(!is_vec_sint_v<Vec1<TestType>>);
        static_assert(!is_vec_sint_size_v<Vec2<TestType>, 2>);
        static_assert(!is_vec_sint_size_v<Vec4<TestType>, 4>);
        static_assert(!is_vec_sint_size_v<Vec4<TestType>, 4>);

        static_assert(!is_vec_int_size_v<Vec2<TestType>, 3>);
        static_assert(!is_vec_real_size_v<Vec4<TestType>, 4>);
        static_assert(!is_vec_int_size_v<Vec3<TestType>, 2>);
        static_assert(!is_vec_int_size_v<Vec3<TestType>, 4>);
        static_assert(!is_vec_bool_size_v<Vec4<TestType>, 4>);
        static_assert(!is_vec_int_size_v<Vec4<TestType>, 3>);

    } else {
        static_assert(is_vec_sint_v<Vec1<TestType>>);
        static_assert(is_vec_sint_size_v<Vec2<TestType>, 2>);
        static_assert(is_vec_sint_size_v<Vec4<TestType>, 4>);
        static_assert(is_vec_sint_size_v<Vec4<TestType>, 4>);

        static_assert(is_vec_int_v<Vec1<TestType>>);
        static_assert(is_vec_int_size_v<Vec2<TestType>, 2>);
        static_assert(is_vec_int_size_v<Vec4<TestType>, 4>);
        static_assert(is_vec_int_size_v<Vec4<TestType>, 4>);

        static_assert(!is_vec_uint_v<Vec1<TestType>>);
        static_assert(!is_vec_uint_size_v<Vec2<TestType>, 2>);
        static_assert(!is_vec_uint_size_v<Vec4<TestType>, 4>);
        static_assert(!is_vec_uint_size_v<Vec4<TestType>, 4>);

        static_assert(!is_vec_int_size_v<Vec2<TestType>, 3>);
        static_assert(!is_vec_real_size_v<Vec4<TestType>, 4>);
        static_assert(!is_vec_int_size_v<Vec3<TestType>, 2>);
        static_assert(!is_vec_int_size_v<Vec3<TestType>, 4>);
        static_assert(!is_vec_bool_size_v<Vec4<TestType>, 4>);
        static_assert(!is_vec_int_size_v<Vec4<TestType>, 3>);
    }
}
