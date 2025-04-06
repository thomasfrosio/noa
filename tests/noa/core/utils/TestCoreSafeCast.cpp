#include <noa/core/utils/SafeCast.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;

TEMPLATE_TEST_CASE("core::safe_cast, floating-point to signed integer", "", i8, i16, i32, i64) {
    using int_limit = std::numeric_limits<TestType>;
    REQUIRE(123 == noa::safe_cast<TestType>(123.f));
    REQUIRE(-123 == noa::safe_cast<TestType>(-123.f));
    REQUIRE(123 == noa::safe_cast<TestType>(123.));
    REQUIRE(-123 == noa::safe_cast<TestType>(-123.));
    REQUIRE(123 == noa::safe_cast<TestType>(f16(123.f)));
    REQUIRE(-123 == noa::safe_cast<TestType>(f16(-123.f)));

    REQUIRE(static_cast<TestType>(123.354f) == noa::safe_cast<TestType>(123.354f));
    REQUIRE(static_cast<TestType>(-123.354f) == noa::safe_cast<TestType>(-123.354f));
    REQUIRE(static_cast<TestType>(123.354) == noa::safe_cast<TestType>(123.354));
    REQUIRE(static_cast<TestType>(-123.354) == noa::safe_cast<TestType>(-123.354));
    REQUIRE(static_cast<TestType>(f16(123.354f)) == noa::safe_cast<TestType>(f16(123.354f)));
    REQUIRE(static_cast<TestType>(f16(-123.354f)) == noa::safe_cast<TestType>(f16(-123.354f)));

    REQUIRE_THROWS_AS(noa::safe_cast<TestType>(::sqrt(-1)), noa::Exception);
    REQUIRE_THROWS_AS(noa::safe_cast<TestType>(sqrt(f16(-1))), noa::Exception);

    REQUIRE_THROWS_AS(noa::safe_cast<TestType>(f32(int_limit::max()) * 2), noa::Exception);
    REQUIRE_THROWS_AS(noa::safe_cast<TestType>(f32(int_limit::min()) * 2), noa::Exception);
    REQUIRE_THROWS_AS(noa::safe_cast<TestType>(std::numeric_limits<f32>::max()), noa::Exception);
    REQUIRE_THROWS_AS(noa::safe_cast<TestType>(std::numeric_limits<f32>::lowest()), noa::Exception);
    REQUIRE(0 == noa::safe_cast<TestType>(std::numeric_limits<f32>::min()));

    REQUIRE_THROWS_AS(noa::safe_cast<TestType>(f64(int_limit::max()) * 2), noa::Exception);
    REQUIRE_THROWS_AS(noa::safe_cast<TestType>(f64(int_limit::min()) * 2), noa::Exception);
    REQUIRE_THROWS_AS(noa::safe_cast<TestType>(std::numeric_limits<f64>::max()), noa::Exception);
    REQUIRE_THROWS_AS(noa::safe_cast<TestType>(std::numeric_limits<f64>::lowest()), noa::Exception);
    REQUIRE(0 == noa::safe_cast<TestType>(std::numeric_limits<f64>::min()));

    if constexpr (sizeof(TestType) > 2) {
        REQUIRE(TestType(std::numeric_limits<f16>::max()) == noa::safe_cast<TestType>(std::numeric_limits<f16>::max()));
        REQUIRE(TestType(std::numeric_limits<f16>::lowest()) == noa::safe_cast<TestType>(std::numeric_limits<f16>::lowest()));
    } else {
        REQUIRE_THROWS_AS(noa::safe_cast<TestType>(std::numeric_limits<f16>::max()), noa::Exception);
        REQUIRE_THROWS_AS(noa::safe_cast<TestType>(std::numeric_limits<f16>::lowest()), noa::Exception);
    }
    REQUIRE(0 == noa::safe_cast<TestType>(std::numeric_limits<f16>::min()));
}

TEMPLATE_TEST_CASE("core::safe_cast, floating-point to unsigned integer", "", u8, u16, u32, u64) {
    REQUIRE(123 == noa::safe_cast<TestType>(123.f));
    REQUIRE_THROWS_AS(noa::safe_cast<TestType>(-123.f), noa::Exception);
    REQUIRE(123 == noa::safe_cast<TestType>(123.));
    REQUIRE_THROWS_AS(noa::safe_cast<TestType>(-123.), noa::Exception);
    REQUIRE(123 == noa::safe_cast<TestType>(f16(123.f)));
    REQUIRE_THROWS_AS(noa::safe_cast<TestType>(f16(-123.f)), noa::Exception);

    REQUIRE(static_cast<TestType>(123.354f) == noa::safe_cast<TestType>(123.354f));
    REQUIRE(static_cast<TestType>(123.845) == noa::safe_cast<TestType>(123.845));
    REQUIRE(static_cast<TestType>(f16(123.5f)) == noa::safe_cast<TestType>(f16(123.5f)));
    REQUIRE_THROWS_AS(noa::safe_cast<TestType>(::sqrt(-1)), noa::Exception);
    REQUIRE_THROWS_AS(noa::safe_cast<TestType>(sqrt(f16(-1))), noa::Exception);
}

TEMPLATE_TEST_CASE("core::safe_cast, integer to floating-point", "", i8, i16, i32, i64, u8, u16, u32, u64) {
    using int_limit = std::numeric_limits<TestType>;
    REQUIRE(static_cast<f32>(int_limit::min()) == noa::safe_cast<f32>(int_limit::min()));
    REQUIRE(static_cast<f32>(int_limit::max()) == noa::safe_cast<f32>(int_limit::max()));
    REQUIRE(static_cast<f64>(int_limit::min()) == noa::safe_cast<f64>(int_limit::min()));
    REQUIRE(static_cast<f64>(int_limit::max()) == noa::safe_cast<f64>(int_limit::max()));
}

TEMPLATE_TEST_CASE("core::safe_cast, f16 to float/double", "", f32, f64) {
    test::Randomizer<f16> randomizer(std::numeric_limits<f16>::lowest(), std::numeric_limits<f16>::max());
    for (size_t i = 0; i < 1000; ++i) {
        f16 v = randomizer.get();
        if (static_cast<TestType>(v) != noa::safe_cast<TestType>(v))
            REQUIRE(false);
    }
}

TEMPLATE_TEST_CASE("core::safe_cast, small integer to f16", "", i8, i16, u8) {
    using int_limit = std::numeric_limits<TestType>;
    REQUIRE(static_cast<f16>(int_limit::min()) == noa::safe_cast<f16>(int_limit::min()));
    REQUIRE(static_cast<f16>(int_limit::max()) == noa::safe_cast<f16>(int_limit::max()));
}

TEMPLATE_TEST_CASE("core::safe_cast, large integer to half_t", "", i32, i64, u16, u32, u64) {
    using int_limit = std::numeric_limits<TestType>;
    if constexpr (std::is_signed_v<TestType>)
        REQUIRE_THROWS_AS(noa::safe_cast<f16>(int_limit::min()), noa::Exception);
    REQUIRE_THROWS_AS(noa::safe_cast<f16>(int_limit::max()), noa::Exception);
}

TEST_CASE("core::safe_cast, double to float/f16") {
    using double_limit = std::numeric_limits<f64>;
    REQUIRE(static_cast<f32>(12456.251) == noa::safe_cast<f32>(12456.251));
    REQUIRE_THROWS_AS(noa::safe_cast<f32>(double_limit::lowest()), noa::Exception);
    REQUIRE_THROWS_AS(noa::safe_cast<f32>(double_limit::max()), noa::Exception);

    REQUIRE(static_cast<f16>(12456.251) == noa::safe_cast<f16>(12456.251));
    REQUIRE_THROWS_AS(noa::safe_cast<f16>(double_limit::lowest()), noa::Exception);
    REQUIRE_THROWS_AS(noa::safe_cast<f16>(double_limit::max()), noa::Exception);
}

TEST_CASE("core::safe_cast, integer to integer") {
    // Unsigned to sign:
    REQUIRE(123456 == noa::safe_cast<i32>(u32(123456)));
    REQUIRE(0 == noa::safe_cast<i32>(std::numeric_limits<u32>::min()));
    REQUIRE_THROWS(noa::safe_cast<i32>(std::numeric_limits<u32>::max()));

    REQUIRE(123456 == noa::safe_cast<i32>(u64(123456)));
    REQUIRE(0 == noa::safe_cast<i32>(std::numeric_limits<u64>::min()));
    REQUIRE_THROWS(noa::safe_cast<i32>(std::numeric_limits<u64>::max()));

    REQUIRE(23456 == noa::safe_cast<i32>(u16(23456)));
    REQUIRE(0 == noa::safe_cast<i32>(std::numeric_limits<u16>::min()));
    REQUIRE(i32(std::numeric_limits<u16>::max()) == noa::safe_cast<i32>(std::numeric_limits<u16>::max()));

    // Signed to unsigned:
    REQUIRE(23456 == noa::safe_cast<u32>(23456));
    REQUIRE_THROWS(noa::safe_cast<u32>(-23456));
    REQUIRE(u32(std::numeric_limits<i32>::max()) == noa::safe_cast<u32>(std::numeric_limits<i32>::max()));

    REQUIRE(23456 == noa::safe_cast<u32>(i64(23456)));
    REQUIRE_THROWS(noa::safe_cast<u32>(i64(-23456)));
    REQUIRE_THROWS(noa::safe_cast<u32>(std::numeric_limits<i64>::max()));

    REQUIRE(23456 == noa::safe_cast<u32>(i16(23456)));
    REQUIRE_THROWS(noa::safe_cast<u32>(std::numeric_limits<i16>::min()));
    REQUIRE(u32(std::numeric_limits<i16>::max()) == noa::safe_cast<u32>(std::numeric_limits<i16>::max()));

    // Unsigned to unsigned
    REQUIRE(123456u == noa::safe_cast<u32>(u32(123456)));
    REQUIRE(0u == noa::safe_cast<u32>(std::numeric_limits<u32>::min()));
    REQUIRE_THROWS(noa::safe_cast<u32>(std::numeric_limits<u64>::max()));

    REQUIRE(123456u == noa::safe_cast<u32>(u64(123456)));
    REQUIRE(0u == noa::safe_cast<u32>(std::numeric_limits<u64>::min()));
    REQUIRE_THROWS(noa::safe_cast<u32>(std::numeric_limits<u64>::max()));

    REQUIRE(23456u == noa::safe_cast<u32>(u16(23456)));
    REQUIRE(0u == noa::safe_cast<u32>(std::numeric_limits<u16>::min()));
    REQUIRE(u32(std::numeric_limits<u16>::max()) == noa::safe_cast<u32>(std::numeric_limits<u16>::max()));

    // Signed to signed
    REQUIRE(23456 == noa::safe_cast<i32>(23456));
    REQUIRE(-23456 == noa::safe_cast<i32>(-23456));
    REQUIRE(std::numeric_limits<i32>::max() == noa::safe_cast<i32>(std::numeric_limits<i32>::max()));

    REQUIRE(23456 == noa::safe_cast<i32>(i64(23456)));
    REQUIRE(-23456 == noa::safe_cast<i32>(i64(-23456)));
    REQUIRE_THROWS(noa::safe_cast<i32>(std::numeric_limits<i64>::min()));
    REQUIRE_THROWS(noa::safe_cast<i32>(std::numeric_limits<i64>::max()));

    REQUIRE(23456 == noa::safe_cast<i32>(i16(23456)));
    REQUIRE(-23456 == noa::safe_cast<i32>(i16(-23456)));
    REQUIRE(i32(std::numeric_limits<i16>::min()) == noa::safe_cast<i32>(std::numeric_limits<i16>::min()));
    REQUIRE(i32(std::numeric_limits<i16>::max()) == noa::safe_cast<i32>(std::numeric_limits<i16>::max()));
}
