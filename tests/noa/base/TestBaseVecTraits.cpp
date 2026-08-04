#include <noa/base/Vec.hpp>

#include "Catch.hpp"

using namespace ::noa::types;

TEMPLATE_TEST_CASE("base::traits:: Vec", "", u8, u16, u32, u64, i8, i16, i32, i64, f32, f64, bool) {
    namespace nt = ::noa::traits;

    static_assert(nt::is_vec_v<Vec<TestType, 1>>);
    static_assert(nt::is_vec_v<const Vec<TestType, 2>>);
    static_assert(not nt::is_vec_v<Vec<TestType, 3>&>);
    static_assert(not nt::is_vec_v<Vec<TestType, 4>&&>);
    static_assert(not nt::is_vec_v<Vec<TestType, 2>*>);

    static_assert(nt::is_vec_of_size_v<Vec<TestType, 2>, 2>);
    static_assert(nt::is_vec_of_size_v<const Vec<TestType, 5>, 5>);
    static_assert(not nt::is_vec_of_size_v<Vec<TestType, 1>&, 1>);

    static_assert(nt::is_vec_of_type_v<Vec<TestType, 2>, TestType>);
    static_assert(nt::is_vec_of_type_v<const Vec<TestType, 5>, TestType>);
    static_assert(not nt::is_vec_of_type_v<Vec<TestType, 1>&, TestType>);
    static_assert(not nt::is_vec_of_type_v<Vec<TestType, 2>, const TestType>);
    static_assert(not nt::is_vec_of_type_v<const Vec<TestType, 5>, TestType&>);
    static_assert(not nt::is_vec_of_type_v<Vec<TestType, 1>, TestType*>);

    if constexpr (std::is_same_v<TestType, f32> or std::is_same_v<TestType, f64>) {
        static_assert(nt::is_vec_real_v<Vec<TestType, 1>>);
        static_assert(nt::is_vec_real_size_v<Vec<TestType, 2>, 2>);
        static_assert(nt::is_vec_real_size_v<Vec<TestType, 4>, 4>);

        static_assert(not nt::is_vec_real_size_v<Vec<TestType, 2>, 3>);
        static_assert(not nt::is_vec_real_size_v<Vec<TestType, 2>, 4>);
        static_assert(not nt::is_vec_real_size_v<Vec<TestType, 3>, 2>);
        static_assert(not nt::is_vec_real_size_v<Vec<TestType, 3>, 4>);
        static_assert(not nt::is_vec_integer_size_v<Vec<TestType, 4>, 4>);
        static_assert(not nt::is_vec_real_size_v<Vec<TestType, 4>, 3>);

    } else if constexpr (std::is_same_v<TestType, bool>) {
        static_assert(nt::is_vec_boolean_v<Vec<TestType, 1>>);
        static_assert(nt::is_vec_boolean_size_v<Vec<TestType, 2>, 2>);
        static_assert(nt::is_vec_boolean_size_v<Vec<TestType, 4>, 4>);
        static_assert(nt::is_vec_boolean_size_v<Vec<TestType, 4>, 4>);
        static_assert(nt::is_vec_integer_size_v<Vec<TestType, 3>, 3>);
        static_assert(nt::is_vec_uinteger_size_v<Vec<TestType, 3>, 3>);
        static_assert(not nt::is_vec_sinteger_size_v<Vec<TestType, 3>, 3>);

        static_assert(not nt::is_vec_boolean_size_v<Vec<TestType, 2>, 4>);
        static_assert(not nt::is_vec_boolean_size_v<Vec<TestType, 3>, 2>);
        static_assert(not nt::is_vec_boolean_size_v<Vec<TestType, 3>, 4>);

    } else if constexpr (std::is_unsigned_v<TestType>) {
        static_assert(nt::is_vec_uinteger_v<Vec<TestType, 1>>);
        static_assert(nt::is_vec_uinteger_size_v<Vec<TestType, 2>, 2>);
        static_assert(nt::is_vec_uinteger_size_v<Vec<TestType, 4>, 4>);
        static_assert(nt::is_vec_uinteger_size_v<Vec<TestType, 4>, 4>);

        static_assert(nt::is_vec_integer_v<Vec<TestType, 1>>);
        static_assert(nt::is_vec_integer_size_v<Vec<TestType, 2>, 2>);
        static_assert(nt::is_vec_integer_size_v<Vec<TestType, 4>, 4>);
        static_assert(nt::is_vec_integer_size_v<Vec<TestType, 4>, 4>);

        static_assert(not nt::is_vec_sinteger_v<Vec<TestType, 1>>);
        static_assert(not nt::is_vec_sinteger_size_v<Vec<TestType, 2>, 2>);
        static_assert(not nt::is_vec_sinteger_size_v<Vec<TestType, 4>, 4>);
        static_assert(not nt::is_vec_sinteger_size_v<Vec<TestType, 4>, 4>);

        static_assert(not nt::is_vec_integer_size_v<Vec<TestType, 2>, 3>);
        static_assert(not nt::is_vec_real_size_v<Vec<TestType, 4>, 4>);
        static_assert(not nt::is_vec_integer_size_v<Vec<TestType, 3>, 2>);
        static_assert(not nt::is_vec_integer_size_v<Vec<TestType, 3>, 4>);
        static_assert(not nt::is_vec_boolean_size_v<Vec<TestType, 4>, 4>);
        static_assert(not nt::is_vec_integer_size_v<Vec<TestType, 4>, 3>);

    } else {
        static_assert(nt::is_vec_sinteger_v<Vec<TestType, 1>>);
        static_assert(nt::is_vec_sinteger_size_v<Vec<TestType, 2>, 2>);
        static_assert(nt::is_vec_sinteger_size_v<Vec<TestType, 4>, 4>);
        static_assert(nt::is_vec_sinteger_size_v<Vec<TestType, 4>, 4>);

        static_assert(nt::is_vec_integer_v<Vec<TestType, 1>>);
        static_assert(nt::is_vec_integer_size_v<Vec<TestType, 2>, 2>);
        static_assert(nt::is_vec_integer_size_v<Vec<TestType, 4>, 4>);
        static_assert(nt::is_vec_integer_size_v<Vec<TestType, 4>, 4>);

        static_assert(not nt::is_vec_uinteger_v<Vec<TestType, 1>>);
        static_assert(not nt::is_vec_uinteger_size_v<Vec<TestType, 2>, 2>);
        static_assert(not nt::is_vec_uinteger_size_v<Vec<TestType, 4>, 4>);
        static_assert(not nt::is_vec_uinteger_size_v<Vec<TestType, 4>, 4>);

        static_assert(not nt::is_vec_integer_size_v<Vec<TestType, 2>, 3>);
        static_assert(not nt::is_vec_real_size_v<Vec<TestType, 4>, 4>);
        static_assert(not nt::is_vec_integer_size_v<Vec<TestType, 3>, 2>);
        static_assert(not nt::is_vec_integer_size_v<Vec<TestType, 3>, 4>);
        static_assert(not nt::is_vec_boolean_size_v<Vec<TestType, 4>, 4>);
        static_assert(not nt::is_vec_integer_size_v<Vec<TestType, 4>, 3>);
    }
}
