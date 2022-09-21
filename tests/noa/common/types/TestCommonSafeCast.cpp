#include <noa/common/types/SafeCast.h>
#include <catch2/catch.hpp>
#include <iostream>

#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("safe_cast, floating-point to signed integer", "[noa][common]",
                   int8_t, int16_t, int32_t, int64_t) {
    using int_limit = std::numeric_limits<TestType>;
    REQUIRE(123 == safe_cast<TestType>(123.f));
    REQUIRE(-123 == safe_cast<TestType>(-123.f));
    REQUIRE(123 == safe_cast<TestType>(123.));
    REQUIRE(-123 == safe_cast<TestType>(-123.));
    REQUIRE(123 == safe_cast<TestType>(half_t(123.f)));
    REQUIRE(-123 == safe_cast<TestType>(half_t(-123.f)));

    REQUIRE(static_cast<TestType>(123.354f) == safe_cast<TestType>(123.354f));
    REQUIRE(static_cast<TestType>(-123.354f) == safe_cast<TestType>(-123.354f));
    REQUIRE(static_cast<TestType>(123.354) == safe_cast<TestType>(123.354));
    REQUIRE(static_cast<TestType>(-123.354) == safe_cast<TestType>(-123.354));
    REQUIRE(static_cast<TestType>(half_t(123.354f)) == safe_cast<TestType>(half_t(123.354f)));
    REQUIRE(static_cast<TestType>(half_t(-123.354f)) == safe_cast<TestType>(half_t(-123.354f)));

    REQUIRE_THROWS_AS(safe_cast<TestType>(::sqrtf(-1)), noa::Exception);
    REQUIRE_THROWS_AS(safe_cast<TestType>(::sqrt(-1)), noa::Exception);
    REQUIRE_THROWS_AS(safe_cast<TestType>(math::sqrt(half_t(-1))), noa::Exception);

    REQUIRE_THROWS_AS(safe_cast<TestType>(float(int_limit::max()) * 2), noa::Exception);
    REQUIRE_THROWS_AS(safe_cast<TestType>(float(int_limit::min()) * 2), noa::Exception);
    REQUIRE_THROWS_AS(safe_cast<TestType>(std::numeric_limits<float>::max()), noa::Exception);
    REQUIRE_THROWS_AS(safe_cast<TestType>(std::numeric_limits<float>::lowest()), noa::Exception);
    REQUIRE(0 == safe_cast<TestType>(std::numeric_limits<float>::min()));

    REQUIRE_THROWS_AS(safe_cast<TestType>(double(int_limit::max()) * 2), noa::Exception);
    REQUIRE_THROWS_AS(safe_cast<TestType>(double(int_limit::min()) * 2), noa::Exception);
    REQUIRE_THROWS_AS(safe_cast<TestType>(std::numeric_limits<double>::max()), noa::Exception);
    REQUIRE_THROWS_AS(safe_cast<TestType>(std::numeric_limits<double>::lowest()), noa::Exception);
    REQUIRE(0 == safe_cast<TestType>(std::numeric_limits<double>::min()));

    if constexpr (sizeof(TestType) > 2) {
        REQUIRE(TestType(std::numeric_limits<half_t>::max()) == safe_cast<TestType>(std::numeric_limits<half_t>::max()));
        REQUIRE(TestType(std::numeric_limits<half_t>::lowest()) == safe_cast<TestType>(std::numeric_limits<half_t>::lowest()));
    } else {
        REQUIRE_THROWS_AS(safe_cast<TestType>(std::numeric_limits<half_t>::max()), noa::Exception);
        REQUIRE_THROWS_AS(safe_cast<TestType>(std::numeric_limits<half_t>::lowest()), noa::Exception);
    }
    REQUIRE(0 == safe_cast<TestType>(std::numeric_limits<half_t>::min()));
}

TEMPLATE_TEST_CASE("safe_cast, floating-point to unsigned integer", "[noa][common]",
                   uint8_t, uint16_t, uint32_t, uint64_t) {
    REQUIRE(123 == safe_cast<TestType>(123.f));
    REQUIRE_THROWS_AS(safe_cast<TestType>(-123.f), noa::Exception);
    REQUIRE(123 == safe_cast<TestType>(123.));
    REQUIRE_THROWS_AS(safe_cast<TestType>(-123.), noa::Exception);
    REQUIRE(123 == safe_cast<TestType>(half_t(123.f)));
    REQUIRE_THROWS_AS(safe_cast<TestType>(half_t(-123.f)), noa::Exception);

    REQUIRE(static_cast<TestType>(123.354f) == safe_cast<TestType>(123.354f));
    REQUIRE(static_cast<TestType>(123.845) == safe_cast<TestType>(123.845));
    REQUIRE(static_cast<TestType>(half_t(123.5f)) == safe_cast<TestType>(half_t(123.5f)));
    REQUIRE_THROWS_AS(safe_cast<TestType>(::sqrtf(-1)), noa::Exception);
    REQUIRE_THROWS_AS(safe_cast<TestType>(::sqrt(-1)), noa::Exception);
    REQUIRE_THROWS_AS(safe_cast<TestType>(math::sqrt(half_t(-1))), noa::Exception);
}

TEMPLATE_TEST_CASE("safe_cast, integer to floating-point", "[noa][common]",
                   int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t) {
    using int_limit = std::numeric_limits<TestType>;
    REQUIRE(static_cast<float>(int_limit::min()) == safe_cast<float>(int_limit::min()));
    REQUIRE(static_cast<float>(int_limit::max()) == safe_cast<float>(int_limit::max()));
    REQUIRE(static_cast<double>(int_limit::min()) == safe_cast<double>(int_limit::min()));
    REQUIRE(static_cast<double>(int_limit::max()) == safe_cast<double>(int_limit::max()));
}

TEMPLATE_TEST_CASE("safe_cast, half_t to float/double", "[noa][common]",
                   float, double) {
    test::Randomizer<half_t> randomizer(math::Limits<half_t>::lowest(), math::Limits<half_t>::max());
    for (size_t i = 0; i < 1000; ++i) {
        half_t v = randomizer.get();
        if (static_cast<TestType>(v) != safe_cast<TestType>(v))
            REQUIRE(false);
    }
}

TEMPLATE_TEST_CASE("safe_cast, small integer to half_t", "[noa][common]",
                   int8_t, int16_t, uint8_t) {
    using int_limit = std::numeric_limits<TestType>;
    REQUIRE(static_cast<half_t>(int_limit::min()) == safe_cast<half_t>(int_limit::min()));
    REQUIRE(static_cast<half_t>(int_limit::max()) == safe_cast<half_t>(int_limit::max()));
}

TEMPLATE_TEST_CASE("safe_cast, large integer to half_t", "[noa][common]",
                   int32_t, int64_t, uint16_t, uint32_t, uint64_t) {
    using int_limit = std::numeric_limits<TestType>;
    if constexpr (std::is_signed_v<TestType>)
        REQUIRE_THROWS_AS(safe_cast<half_t>(int_limit::min()), noa::Exception);
    REQUIRE_THROWS_AS(safe_cast<half_t>(int_limit::max()), noa::Exception);
}

TEST_CASE("safe_cast, double to float/half_t", "[noa][common]") {
    using double_limit = std::numeric_limits<double>;
    REQUIRE(static_cast<float>(12456.251) == safe_cast<float>(12456.251));
    REQUIRE_THROWS_AS(safe_cast<float>(double_limit::lowest()), noa::Exception);
    REQUIRE_THROWS_AS(safe_cast<float>(double_limit::max()), noa::Exception);

    REQUIRE(static_cast<half_t>(12456.251) == safe_cast<half_t>(12456.251));
    REQUIRE_THROWS_AS(safe_cast<half_t>(double_limit::lowest()), noa::Exception);
    REQUIRE_THROWS_AS(safe_cast<half_t>(double_limit::max()), noa::Exception);
}

TEST_CASE("safe_cast, integer to integer", "[noa][common]") {
    // Unsigned to sign:
    REQUIRE(123456 == safe_cast<int32_t>(uint32_t(123456)));
    REQUIRE(0 == safe_cast<int32_t>(std::numeric_limits<uint32_t>::min()));
    REQUIRE_THROWS(safe_cast<int32_t>(std::numeric_limits<uint32_t>::max()));

    REQUIRE(123456 == safe_cast<int32_t>(uint64_t(123456)));
    REQUIRE(0 == safe_cast<int32_t>(std::numeric_limits<uint64_t>::min()));
    REQUIRE_THROWS(safe_cast<int32_t>(std::numeric_limits<uint64_t>::max()));

    REQUIRE(23456 == safe_cast<int32_t>(uint16_t(23456)));
    REQUIRE(0 == safe_cast<int32_t>(std::numeric_limits<uint16_t>::min()));
    REQUIRE(int32_t(std::numeric_limits<uint16_t>::max()) == safe_cast<int32_t>(std::numeric_limits<uint16_t>::max()));

    // Signed to unsigned:
    REQUIRE(23456 == safe_cast<uint32_t>(23456));
    REQUIRE_THROWS(safe_cast<uint32_t>(-23456));
    REQUIRE(uint32_t(std::numeric_limits<int32_t>::max()) == safe_cast<uint32_t>(std::numeric_limits<int32_t>::max()));

    REQUIRE(23456 == safe_cast<uint32_t>(int64_t(23456)));
    REQUIRE_THROWS(safe_cast<uint32_t>(int64_t(-23456)));
    REQUIRE_THROWS(safe_cast<uint32_t>(std::numeric_limits<int64_t>::max()));

    REQUIRE(23456 == safe_cast<uint32_t>(int16_t(23456)));
    REQUIRE_THROWS(safe_cast<uint32_t>(std::numeric_limits<int16_t>::min()));
    REQUIRE(uint32_t(std::numeric_limits<int16_t>::max()) == safe_cast<uint32_t>(std::numeric_limits<int16_t>::max()));

    // Unsigned to unsigned
    REQUIRE(123456u == safe_cast<uint32_t>(uint32_t(123456)));
    REQUIRE(0u == safe_cast<uint32_t>(std::numeric_limits<uint32_t>::min()));
    REQUIRE_THROWS(safe_cast<uint32_t>(std::numeric_limits<uint64_t>::max()));

    REQUIRE(123456u == safe_cast<uint32_t>(uint64_t(123456)));
    REQUIRE(0u == safe_cast<uint32_t>(std::numeric_limits<uint64_t>::min()));
    REQUIRE_THROWS(safe_cast<uint32_t>(std::numeric_limits<uint64_t>::max()));

    REQUIRE(23456u == safe_cast<uint32_t>(uint16_t(23456)));
    REQUIRE(0u == safe_cast<uint32_t>(std::numeric_limits<uint16_t>::min()));
    REQUIRE(uint32_t(std::numeric_limits<uint16_t>::max()) == safe_cast<uint32_t>(std::numeric_limits<uint16_t>::max()));

    // Signed to signed
    REQUIRE(23456 == safe_cast<int32_t>(23456));
    REQUIRE(-23456 == safe_cast<int32_t>(-23456));
    REQUIRE(std::numeric_limits<int32_t>::max() == safe_cast<int32_t>(std::numeric_limits<int32_t>::max()));

    REQUIRE(23456 == safe_cast<int32_t>(int64_t(23456)));
    REQUIRE(-23456 == safe_cast<int32_t>(int64_t(-23456)));
    REQUIRE_THROWS(safe_cast<int32_t>(std::numeric_limits<int64_t>::min()));
    REQUIRE_THROWS(safe_cast<int32_t>(std::numeric_limits<int64_t>::max()));

    REQUIRE(23456 == safe_cast<int32_t>(int16_t(23456)));
    REQUIRE(-23456 == safe_cast<int32_t>(int16_t(-23456)));
    REQUIRE(int32_t(std::numeric_limits<int16_t>::min()) == safe_cast<int32_t>(std::numeric_limits<int16_t>::min()));
    REQUIRE(int32_t(std::numeric_limits<int16_t>::max()) == safe_cast<int32_t>(std::numeric_limits<int16_t>::max()));
}
