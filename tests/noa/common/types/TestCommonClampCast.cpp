#include <noa/common/types/ClampCast.h>
#include <catch2/catch.hpp>

#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("common: clamp_cast, floating-point to signed integer", "[noa][common]",
                   int8_t, int16_t, int32_t, int64_t) {
    using int_limit = std::numeric_limits<TestType>;
    REQUIRE(123 == clamp_cast<TestType>(123.f));
    REQUIRE(-123 == clamp_cast<TestType>(-123.f));
    REQUIRE(123 == clamp_cast<TestType>(123.));
    REQUIRE(-123 == clamp_cast<TestType>(-123.));
    REQUIRE(123 == clamp_cast<TestType>(half_t(123.f)));
    REQUIRE(-123 == clamp_cast<TestType>(half_t(-123.f)));

    REQUIRE(static_cast<TestType>(123.354f) == clamp_cast<TestType>(123.354f));
    REQUIRE(static_cast<TestType>(-123.354f) == clamp_cast<TestType>(-123.354f));
    REQUIRE(static_cast<TestType>(123.354) == clamp_cast<TestType>(123.354));
    REQUIRE(static_cast<TestType>(-123.354) == clamp_cast<TestType>(-123.354));
    REQUIRE(static_cast<TestType>(half_t(123.354f)) == clamp_cast<TestType>(half_t(123.354f)));
    REQUIRE(static_cast<TestType>(half_t(-123.354f)) == clamp_cast<TestType>(half_t(-123.354f)));

    REQUIRE(0 == clamp_cast<TestType>(::sqrtf(-1)));
    REQUIRE(0 == clamp_cast<TestType>(::sqrt(-1)));
    REQUIRE(0 == clamp_cast<TestType>(math::sqrt(half_t(-1))));

    REQUIRE(int_limit::max() == clamp_cast<TestType>(float(int_limit::max()) * 2));
    REQUIRE(int_limit::min() == clamp_cast<TestType>(float(int_limit::min()) * 2));
    REQUIRE(int_limit::max() == clamp_cast<TestType>(std::numeric_limits<float>::max()));
    REQUIRE(int_limit::min() == clamp_cast<TestType>(std::numeric_limits<float>::lowest()));
    REQUIRE(0 == clamp_cast<TestType>(std::numeric_limits<float>::min()));

    REQUIRE(int_limit::max() == clamp_cast<TestType>(double(int_limit::max()) * 2));
    REQUIRE(int_limit::min() == clamp_cast<TestType>(double(int_limit::min()) * 2));
    REQUIRE(int_limit::max() == clamp_cast<TestType>(std::numeric_limits<double>::max()));
    REQUIRE(int_limit::min() == clamp_cast<TestType>(std::numeric_limits<double>::lowest()));
    REQUIRE(0 == clamp_cast<TestType>(std::numeric_limits<double>::min()));

    REQUIRE(int_limit::max() == clamp_cast<TestType>(half_t(int_limit::max())));
    REQUIRE(int_limit::min() == clamp_cast<TestType>(half_t(int_limit::min())));
    if constexpr (sizeof(TestType) > 2) {
        REQUIRE(TestType(math::Limits<half_t>::max()) == clamp_cast<TestType>(math::Limits<half_t>::max()));
        REQUIRE(TestType(math::Limits<half_t>::lowest()) == clamp_cast<TestType>(math::Limits<half_t>::lowest()));
    } else {
        REQUIRE(int_limit::max() == clamp_cast<TestType>(std::numeric_limits<half_t>::max()));
        REQUIRE(int_limit::min() == clamp_cast<TestType>(std::numeric_limits<half_t>::lowest()));
    }
    REQUIRE(0 == clamp_cast<TestType>(std::numeric_limits<half_t>::min()));
}

TEMPLATE_TEST_CASE("common: clamp_cast, floating-point to unsigned integer", "[noa][common]",
                   uint8_t, uint16_t, uint32_t, uint64_t) {
    using int_limit = std::numeric_limits<TestType>;
    REQUIRE(123 == clamp_cast<TestType>(123.f));
    REQUIRE(0 == clamp_cast<TestType>(-123.f));
    REQUIRE(123 == clamp_cast<TestType>(123.));
    REQUIRE(0 == clamp_cast<TestType>(-123.));
    REQUIRE(123 == clamp_cast<TestType>(half_t(123.f)));
    REQUIRE(0 == clamp_cast<TestType>(half_t(-123.f)));

    REQUIRE(static_cast<TestType>(123.354f) == clamp_cast<TestType>(123.354f));
    REQUIRE(static_cast<TestType>(123.845) == clamp_cast<TestType>(123.845));
    REQUIRE(static_cast<TestType>(half_t(123.5f)) == clamp_cast<TestType>(half_t(123.5f)));
    REQUIRE(0 == clamp_cast<TestType>(::sqrtf(-1)));
    REQUIRE(0 == clamp_cast<TestType>(::sqrt(-1)));
    REQUIRE(0 == clamp_cast<TestType>(math::sqrt(half_t(-1))));
    REQUIRE(int_limit::max() == clamp_cast<TestType>(float(int_limit::max())));
    REQUIRE(int_limit::min() == clamp_cast<TestType>(float(int_limit::min())));
    REQUIRE(int_limit::max() == clamp_cast<TestType>(double(int_limit::max())));
    REQUIRE(int_limit::min() == clamp_cast<TestType>(double(int_limit::min())));
    REQUIRE(int_limit::max() == clamp_cast<TestType>(half_t(int_limit::max())));
    REQUIRE(int_limit::min() == clamp_cast<TestType>(half_t(int_limit::min())));
}

TEMPLATE_TEST_CASE("common: clamp_cast, integer to floating-point", "[noa][common]",
                   int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t) {
    using int_limit = std::numeric_limits<TestType>;
    REQUIRE(static_cast<float>(int_limit::min()) == clamp_cast<float>(int_limit::min()));
    REQUIRE(static_cast<float>(int_limit::max()) == clamp_cast<float>(int_limit::max()));
    REQUIRE(static_cast<double>(int_limit::min()) == clamp_cast<double>(int_limit::min()));
    REQUIRE(static_cast<double>(int_limit::max()) == clamp_cast<double>(int_limit::max()));
}

TEMPLATE_TEST_CASE("common: clamp_cast, half_t to float/double", "[noa][common]",
                   float, double) {
    test::Randomizer<half_t> randomizer(math::Limits<half_t>::lowest(), math::Limits<half_t>::max());
    for (size_t i = 0; i < 1000; ++i) {
        half_t v = randomizer.get();
        if (static_cast<TestType>(v) != clamp_cast<TestType>(v))
            REQUIRE(false);
    }
}

TEMPLATE_TEST_CASE("common: clamp_cast, small integer to half_t", "[noa][common]",
                   int8_t, int16_t, uint8_t) {
    using int_limit = std::numeric_limits<TestType>;
    REQUIRE(static_cast<half_t>(int_limit::min()) == clamp_cast<half_t>(int_limit::min()));
    REQUIRE(static_cast<half_t>(int_limit::max()) == clamp_cast<half_t>(int_limit::max()));
}

TEMPLATE_TEST_CASE("common: clamp_cast, large integer to half_t", "[noa][common]",
                   int32_t, int64_t, uint16_t, uint32_t, uint64_t) {
    using int_limit = std::numeric_limits<TestType>;
    if constexpr (std::is_signed_v<TestType>) {
        REQUIRE(math::Limits<half_t>::lowest() == clamp_cast<half_t>(int_limit::min()));
        REQUIRE(math::Limits<half_t>::max() == clamp_cast<half_t>(int_limit::max()));
    } else {
        REQUIRE(half_t(0) == clamp_cast<half_t>(int_limit::min()));
        REQUIRE(math::Limits<half_t>::max() == clamp_cast<half_t>(int_limit::max()));
    }
}

TEST_CASE("common: clamp_cast, double to float/half_t", "[noa][common]") {
    using double_limit = std::numeric_limits<double>;
    REQUIRE(static_cast<float>(12456.251) == clamp_cast<float>(12456.251));
    REQUIRE(static_cast<float>(double_limit::min()) == clamp_cast<float>(double_limit::min()));
    REQUIRE(std::numeric_limits<float>::lowest() == clamp_cast<float>(double_limit::lowest()));
    REQUIRE(std::numeric_limits<float>::max() == clamp_cast<float>(double_limit::max()));

    REQUIRE(static_cast<half_t>(12456.251) == clamp_cast<half_t>(12456.251));
    REQUIRE(static_cast<half_t>(double_limit::min()) == clamp_cast<half_t>(double_limit::min()));
    REQUIRE(std::numeric_limits<half_t>::lowest() == clamp_cast<half_t>(double_limit::lowest()));
    REQUIRE(std::numeric_limits<half_t>::max() == clamp_cast<half_t>(double_limit::max()));
}

TEMPLATE_TEST_CASE("common: clamp_cast, scalar to complex", "[noa][common]",
                   int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t, half_t, float, double) {
    auto c1 = clamp_cast<cfloat_t>(std::numeric_limits<TestType>::lowest());
    REQUIRE(clamp_cast<float>(std::numeric_limits<TestType>::lowest()) == c1.real);
    REQUIRE(0.f == c1.imag);

    c1 = clamp_cast<cfloat_t>(std::numeric_limits<TestType>::max());
    REQUIRE(clamp_cast<float>(std::numeric_limits<TestType>::max()) == c1.real);
    REQUIRE(0.f == c1.imag);

    auto c2 = clamp_cast<cdouble_t>(std::numeric_limits<TestType>::lowest());
    REQUIRE(clamp_cast<double>(std::numeric_limits<TestType>::lowest()) == c2.real);
    REQUIRE(0. == c2.imag);

    c2 = clamp_cast<cdouble_t>(std::numeric_limits<TestType>::max());
    REQUIRE(clamp_cast<double>(std::numeric_limits<TestType>::max()) == c2.real);
    REQUIRE(0. == c2.imag);

    auto c3 = clamp_cast<chalf_t>(std::numeric_limits<TestType>::lowest());
    REQUIRE(clamp_cast<half_t>(std::numeric_limits<TestType>::lowest()) == c3.real);
    REQUIRE(half_t(0) == c3.imag);

    c3 = clamp_cast<chalf_t>(std::numeric_limits<TestType>::max());
    REQUIRE(clamp_cast<half_t>(std::numeric_limits<TestType>::max()) == c3.real);
    REQUIRE(half_t(0) == c3.imag);
}

TEST_CASE("common: clamp_cast, integer to integer", "[noa][common]") {
    // Unsigned to sign:
    REQUIRE(123456 == clamp_cast<int32_t>(uint32_t(123456)));
    REQUIRE(0 == clamp_cast<int32_t>(std::numeric_limits<uint32_t>::min()));
    REQUIRE(std::numeric_limits<int32_t>::max() == clamp_cast<int32_t>(std::numeric_limits<uint32_t>::max()));

    REQUIRE(123456 == clamp_cast<int32_t>(uint64_t(123456)));
    REQUIRE(0 == clamp_cast<int32_t>(std::numeric_limits<uint64_t>::min()));
    REQUIRE(std::numeric_limits<int32_t>::max() == clamp_cast<int32_t>(std::numeric_limits<uint64_t>::max()));

    REQUIRE(23456 == clamp_cast<int32_t>(uint16_t(23456)));
    REQUIRE(0 == clamp_cast<int32_t>(std::numeric_limits<uint16_t>::min()));
    REQUIRE(int32_t(std::numeric_limits<uint16_t>::max()) == clamp_cast<int32_t>(std::numeric_limits<uint16_t>::max()));

    // Signed to unsigned:
    REQUIRE(23456 == clamp_cast<uint32_t>(23456));
    REQUIRE(0 == clamp_cast<uint32_t>(-23456));
    REQUIRE(uint32_t(std::numeric_limits<int32_t>::max()) == clamp_cast<uint32_t>(std::numeric_limits<int32_t>::max()));

    REQUIRE(23456 == clamp_cast<uint32_t>(int64_t(23456)));
    REQUIRE(0 == clamp_cast<uint32_t>(int64_t(-23456)));
    REQUIRE(std::numeric_limits<uint32_t>::max() == clamp_cast<uint32_t>(std::numeric_limits<int64_t>::max()));

    REQUIRE(23456 == clamp_cast<uint32_t>(int16_t(23456)));
    REQUIRE(0 == clamp_cast<uint32_t>(std::numeric_limits<int16_t>::min()));
    REQUIRE(uint32_t(std::numeric_limits<int16_t>::max()) == clamp_cast<uint32_t>(std::numeric_limits<int16_t>::max()));

    // Unsigned to unsigned
    REQUIRE(123456u == clamp_cast<uint32_t>(uint32_t(123456)));
    REQUIRE(0u == clamp_cast<uint32_t>(std::numeric_limits<uint32_t>::min()));
    REQUIRE(std::numeric_limits<uint32_t>::max() == clamp_cast<uint32_t>(std::numeric_limits<uint32_t>::max()));

    REQUIRE(123456u == clamp_cast<uint32_t>(uint64_t(123456)));
    REQUIRE(0u == clamp_cast<uint32_t>(std::numeric_limits<uint64_t>::min()));
    REQUIRE(std::numeric_limits<uint32_t>::max() == clamp_cast<uint32_t>(std::numeric_limits<uint64_t>::max()));

    REQUIRE(23456u == clamp_cast<uint32_t>(uint16_t(23456)));
    REQUIRE(0u == clamp_cast<uint32_t>(std::numeric_limits<uint16_t>::min()));
    REQUIRE(uint32_t(std::numeric_limits<uint16_t>::max()) == clamp_cast<uint32_t>(std::numeric_limits<uint16_t>::max()));

    // Signed to signed
    REQUIRE(23456 == clamp_cast<int32_t>(23456));
    REQUIRE(-23456 == clamp_cast<int32_t>(-23456));
    REQUIRE(std::numeric_limits<int32_t>::max() == clamp_cast<int32_t>(std::numeric_limits<int32_t>::max()));

    REQUIRE(23456 == clamp_cast<int32_t>(int64_t(23456)));
    REQUIRE(-23456 == clamp_cast<int32_t>(int64_t(-23456)));
    REQUIRE(std::numeric_limits<int32_t>::min() == clamp_cast<int32_t>(std::numeric_limits<int64_t>::min()));
    REQUIRE(std::numeric_limits<int32_t>::max() == clamp_cast<int32_t>(std::numeric_limits<int64_t>::max()));

    REQUIRE(23456 == clamp_cast<int32_t>(int16_t(23456)));
    REQUIRE(-23456 == clamp_cast<int32_t>(int16_t(-23456)));
    REQUIRE(int32_t(std::numeric_limits<int16_t>::min()) == clamp_cast<int32_t>(std::numeric_limits<int16_t>::min()));
    REQUIRE(int32_t(std::numeric_limits<int16_t>::max()) == clamp_cast<int32_t>(std::numeric_limits<int16_t>::max()));
}
