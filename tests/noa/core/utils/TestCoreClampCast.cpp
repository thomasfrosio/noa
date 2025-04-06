#include <noa/core/utils/ClampCast.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;

TEMPLATE_TEST_CASE("core::clamp_cast, floating-point to signed integer", "", i8, i16, i32, i64) {
    using int_limit = std::numeric_limits<TestType>;
    REQUIRE(123 == noa::clamp_cast<TestType>(123.f));
    REQUIRE(-123 == noa::clamp_cast<TestType>(-123.f));
    REQUIRE(123 == noa::clamp_cast<TestType>(123.));
    REQUIRE(-123 == noa::clamp_cast<TestType>(-123.));
    REQUIRE(123 == noa::clamp_cast<TestType>(f16(123.f)));
    REQUIRE(-123 == noa::clamp_cast<TestType>(f16(-123.f)));

    REQUIRE(static_cast<TestType>(123.354f) == noa::clamp_cast<TestType>(123.354f));
    REQUIRE(static_cast<TestType>(-123.354f) == noa::clamp_cast<TestType>(-123.354f));
    REQUIRE(static_cast<TestType>(123.354) == noa::clamp_cast<TestType>(123.354));
    REQUIRE(static_cast<TestType>(-123.354) == noa::clamp_cast<TestType>(-123.354));
    REQUIRE(static_cast<TestType>(f16(123.354f)) == noa::clamp_cast<TestType>(f16(123.354f)));
    REQUIRE(static_cast<TestType>(f16(-123.354f)) == noa::clamp_cast<TestType>(f16(-123.354f)));

    REQUIRE(0 == noa::clamp_cast<TestType>(std::sqrt(-1)));
    REQUIRE(0 == noa::clamp_cast<TestType>(noa::sqrt(f16(-1))));

    REQUIRE(int_limit::max() == noa::clamp_cast<TestType>(f32(int_limit::max()) * 2));
    REQUIRE(int_limit::min() == noa::clamp_cast<TestType>(f32(int_limit::min()) * 2));
    REQUIRE(int_limit::max() == noa::clamp_cast<TestType>(std::numeric_limits<f32>::max()));
    REQUIRE(int_limit::min() == noa::clamp_cast<TestType>(std::numeric_limits<f32>::lowest()));
    REQUIRE(0 == noa::clamp_cast<TestType>(std::numeric_limits<f32>::min()));

    REQUIRE(int_limit::max() == noa::clamp_cast<TestType>(f64(int_limit::max()) * 2));
    REQUIRE(int_limit::min() == noa::clamp_cast<TestType>(f64(int_limit::min()) * 2));
    REQUIRE(int_limit::max() == noa::clamp_cast<TestType>(std::numeric_limits<f64>::max()));
    REQUIRE(int_limit::min() == noa::clamp_cast<TestType>(std::numeric_limits<f64>::lowest()));
    REQUIRE(0 == noa::clamp_cast<TestType>(std::numeric_limits<f64>::min()));

    REQUIRE(int_limit::max() == noa::clamp_cast<TestType>(f16(int_limit::max())));
    REQUIRE(int_limit::min() == noa::clamp_cast<TestType>(f16(int_limit::min())));
    if constexpr (sizeof(TestType) > 2) {
        REQUIRE(TestType(std::numeric_limits<f16>::max()) == noa::clamp_cast<TestType>(std::numeric_limits<f16>::max()));
        REQUIRE(TestType(std::numeric_limits<f16>::lowest()) == noa::clamp_cast<TestType>(std::numeric_limits<f16>::lowest()));
    } else {
        REQUIRE(int_limit::max() == noa::clamp_cast<TestType>(std::numeric_limits<f16>::max()));
        REQUIRE(int_limit::min() == noa::clamp_cast<TestType>(std::numeric_limits<f16>::lowest()));
    }
    REQUIRE(0 == noa::clamp_cast<TestType>(std::numeric_limits<f16>::min()));
}

TEMPLATE_TEST_CASE("core::clamp_cast, floating-point to unsigned integer", "", u8, u16, u32, u64) {
    using int_limit = std::numeric_limits<TestType>;
    REQUIRE(123 == noa::clamp_cast<TestType>(123.f));
    REQUIRE(0 == noa::clamp_cast<TestType>(-123.f));
    REQUIRE(123 == noa::clamp_cast<TestType>(123.));
    REQUIRE(0 == noa::clamp_cast<TestType>(-123.));
    REQUIRE(123 == noa::clamp_cast<TestType>(f16(123.f)));
    REQUIRE(0 == noa::clamp_cast<TestType>(f16(-123.f)));

    REQUIRE(static_cast<TestType>(123.354f) == noa::clamp_cast<TestType>(123.354f));
    REQUIRE(static_cast<TestType>(123.845) == noa::clamp_cast<TestType>(123.845));
    REQUIRE(static_cast<TestType>(f16(123.5f)) == noa::clamp_cast<TestType>(f16(123.5f)));
    REQUIRE(0 == noa::clamp_cast<TestType>(::sqrt(-1)));
    REQUIRE(0 == noa::clamp_cast<TestType>(noa::sqrt(f16(-1))));
    REQUIRE(int_limit::max() == noa::clamp_cast<TestType>(f32(int_limit::max())));
    REQUIRE(int_limit::min() == noa::clamp_cast<TestType>(f32(int_limit::min())));
    REQUIRE(int_limit::max() == noa::clamp_cast<TestType>(f64(int_limit::max())));
    REQUIRE(int_limit::min() == noa::clamp_cast<TestType>(f64(int_limit::min())));
    REQUIRE(int_limit::max() == noa::clamp_cast<TestType>(f16(int_limit::max())));
    REQUIRE(int_limit::min() == noa::clamp_cast<TestType>(f16(int_limit::min())));
}

TEMPLATE_TEST_CASE("core::clamp_cast, integer to floating-point", "", i8, i16, i32, i64, u8, u16, u32, u64) {
    using int_limit = std::numeric_limits<TestType>;
    REQUIRE(static_cast<f32>(int_limit::min()) == noa::clamp_cast<f32>(int_limit::min()));
    REQUIRE(static_cast<f32>(int_limit::max()) == noa::clamp_cast<f32>(int_limit::max()));
    REQUIRE(static_cast<f64>(int_limit::min()) == noa::clamp_cast<f64>(int_limit::min()));
    REQUIRE(static_cast<f64>(int_limit::max()) == noa::clamp_cast<f64>(int_limit::max()));
}

TEMPLATE_TEST_CASE("core::clamp_cast, f16 to f32/f64", "", f32, double) {
    test::Randomizer<f16> randomizer(std::numeric_limits<f16>::lowest(), std::numeric_limits<f16>::max());
    for (size_t i = 0; i < 1000; ++i) {
        f16 v = randomizer.get();
        if (static_cast<TestType>(v) != noa::clamp_cast<TestType>(v))
            REQUIRE(false);
    }
}

TEMPLATE_TEST_CASE("core::clamp_cast, small integer to f16", "", i8, i16, u8) {
    using int_limit = std::numeric_limits<TestType>;
    REQUIRE(static_cast<f16>(int_limit::min()) == noa::clamp_cast<f16>(int_limit::min()));
    REQUIRE(static_cast<f16>(int_limit::max()) == noa::clamp_cast<f16>(int_limit::max()));
}

TEMPLATE_TEST_CASE("core::clamp_cast, large integer to f16", "", i32, i64, u16, u32, u64) {
    using int_limit = std::numeric_limits<TestType>;
    if constexpr (std::is_signed_v<TestType>) {
        REQUIRE(std::numeric_limits<f16>::lowest() == noa::clamp_cast<f16>(int_limit::min()));
        REQUIRE(std::numeric_limits<f16>::max() == noa::clamp_cast<f16>(int_limit::max()));
    } else {
        REQUIRE(f16(0) == noa::clamp_cast<f16>(int_limit::min()));
        REQUIRE(std::numeric_limits<f16>::max() == noa::clamp_cast<f16>(int_limit::max()));
    }
}

NOA_NV_DIAG_SUPPRESS(221)
TEST_CASE("core::clamp_cast, f64 to f32/f16") {
    using double_limit = std::numeric_limits<f64>;
    REQUIRE(static_cast<f32>(12456.251) == noa::clamp_cast<f32>(12456.251));
    REQUIRE(static_cast<f32>(double_limit::min()) == noa::clamp_cast<f32>(double_limit::min()));
    REQUIRE(std::numeric_limits<f32>::lowest() == noa::clamp_cast<f32>(double_limit::lowest()));
    REQUIRE(std::numeric_limits<f32>::max() == noa::clamp_cast<f32>(double_limit::max()));

    REQUIRE(static_cast<f16>(12456.251) == noa::clamp_cast<f16>(12456.251));
    REQUIRE(static_cast<f16>(double_limit::min()) == noa::clamp_cast<f16>(double_limit::min()));
    REQUIRE(std::numeric_limits<f16>::lowest() == noa::clamp_cast<f16>(double_limit::lowest()));
    REQUIRE(std::numeric_limits<f16>::max() == noa::clamp_cast<f16>(double_limit::max()));
}
NOA_NV_DIAG_DEFAULT(221)

TEMPLATE_TEST_CASE("core::clamp_cast, scalar to complex", "", i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64) {
    auto c1 = noa::clamp_cast<Complex<f32>>(std::numeric_limits<TestType>::lowest());
    REQUIRE(noa::clamp_cast<f32>(std::numeric_limits<TestType>::lowest()) == c1.real);
    REQUIRE(0.f == c1.imag);

    c1 = noa::clamp_cast<Complex<f32>>(std::numeric_limits<TestType>::max());
    REQUIRE(noa::clamp_cast<f32>(std::numeric_limits<TestType>::max()) == c1.real);
    REQUIRE(0.f == c1.imag);

    auto c2 = noa::clamp_cast<Complex<f64>>(std::numeric_limits<TestType>::lowest());
    REQUIRE(noa::clamp_cast<f64>(std::numeric_limits<TestType>::lowest()) == c2.real);
    REQUIRE(0. == c2.imag);

    c2 = noa::clamp_cast<Complex<f64>>(std::numeric_limits<TestType>::max());
    REQUIRE(noa::clamp_cast<f64>(std::numeric_limits<TestType>::max()) == c2.real);
    REQUIRE(0. == c2.imag);

    auto c3 = noa::clamp_cast<Complex<f16>>(std::numeric_limits<TestType>::lowest());
    REQUIRE(noa::clamp_cast<f16>(std::numeric_limits<TestType>::lowest()) == c3.real);
    REQUIRE(f16(0) == c3.imag);

    c3 = noa::clamp_cast<Complex<f16>>(std::numeric_limits<TestType>::max());
    REQUIRE(noa::clamp_cast<f16>(std::numeric_limits<TestType>::max()) == c3.real);
    REQUIRE(f16(0) == c3.imag);
}

TEST_CASE("core::clamp_cast, integer to integer") {
    // Unsigned to sign:
    REQUIRE(123456 == noa::clamp_cast<i32>(u32(123456)));
    REQUIRE(0 == noa::clamp_cast<i32>(std::numeric_limits<u32>::min()));
    REQUIRE(std::numeric_limits<i32>::max() == noa::clamp_cast<i32>(std::numeric_limits<u32>::max()));

    REQUIRE(123456 == noa::clamp_cast<i32>(u64(123456)));
    REQUIRE(0 == noa::clamp_cast<i32>(std::numeric_limits<u64>::min()));
    REQUIRE(std::numeric_limits<i32>::max() == noa::clamp_cast<i32>(std::numeric_limits<u64>::max()));

    REQUIRE(23456 == noa::clamp_cast<i32>(u16(23456)));
    REQUIRE(0 == noa::clamp_cast<i32>(std::numeric_limits<u16>::min()));
    REQUIRE(i32(std::numeric_limits<u16>::max()) == noa::clamp_cast<i32>(std::numeric_limits<u16>::max()));

    // Signed to unsigned:
    REQUIRE(23456 == noa::clamp_cast<u32>(23456));
    REQUIRE(0 == noa::clamp_cast<u32>(-23456));
    REQUIRE(u32(std::numeric_limits<i32>::max()) == noa::clamp_cast<u32>(std::numeric_limits<i32>::max()));

    REQUIRE(23456 == noa::clamp_cast<u32>(i64(23456)));
    REQUIRE(0 == noa::clamp_cast<u32>(i64(-23456)));
    REQUIRE(std::numeric_limits<u32>::max() == noa::clamp_cast<u32>(std::numeric_limits<i64>::max()));

    REQUIRE(23456 == noa::clamp_cast<u32>(i16(23456)));
    REQUIRE(0 == noa::clamp_cast<u32>(std::numeric_limits<i16>::min()));
    REQUIRE(u32(std::numeric_limits<i16>::max()) == noa::clamp_cast<u32>(std::numeric_limits<i16>::max()));

    // Unsigned to unsigned
    REQUIRE(123456u == noa::clamp_cast<u32>(u32(123456)));
    REQUIRE(0u == noa::clamp_cast<u32>(std::numeric_limits<u32>::min()));
    REQUIRE(std::numeric_limits<u32>::max() == noa::clamp_cast<u32>(std::numeric_limits<u32>::max()));

    REQUIRE(123456u == noa::clamp_cast<u32>(u64(123456)));
    REQUIRE(0u == noa::clamp_cast<u32>(std::numeric_limits<u64>::min()));
    REQUIRE(std::numeric_limits<u32>::max() == noa::clamp_cast<u32>(std::numeric_limits<u64>::max()));

    REQUIRE(23456u == noa::clamp_cast<u32>(u16(23456)));
    REQUIRE(0u == noa::clamp_cast<u32>(std::numeric_limits<u16>::min()));
    REQUIRE(u32(std::numeric_limits<u16>::max()) == noa::clamp_cast<u32>(std::numeric_limits<u16>::max()));

    // Signed to signed
    REQUIRE(23456 == noa::clamp_cast<i32>(23456));
    REQUIRE(-23456 == noa::clamp_cast<i32>(-23456));
    REQUIRE(std::numeric_limits<i32>::max() == noa::clamp_cast<i32>(std::numeric_limits<i32>::max()));

    REQUIRE(23456 == noa::clamp_cast<i32>(i64(23456)));
    REQUIRE(-23456 == noa::clamp_cast<i32>(i64(-23456)));
    REQUIRE(std::numeric_limits<i32>::min() == noa::clamp_cast<i32>(std::numeric_limits<i64>::min()));
    REQUIRE(std::numeric_limits<i32>::max() == noa::clamp_cast<i32>(std::numeric_limits<i64>::max()));

    REQUIRE(23456 == noa::clamp_cast<i32>(i16(23456)));
    REQUIRE(-23456 == noa::clamp_cast<i32>(i16(-23456)));
    REQUIRE(i32(std::numeric_limits<i16>::min()) == noa::clamp_cast<i32>(std::numeric_limits<i16>::min()));
    REQUIRE(i32(std::numeric_limits<i16>::max()) == noa::clamp_cast<i32>(std::numeric_limits<i16>::max()));
}
