#include <noa/core/utils/Irange.hpp>

#include "Catch.hpp"

TEMPLATE_TEST_CASE("core::irange()", "", int32_t, uint32_t, int64_t, uint64_t) {
    TestType ii = 0;
    for (auto i: noa::irange<TestType>(10)) {
        REQUIRE(i == ii);
        ++ii;
    }

    ii = 0;
    for (auto i: noa::irange<TestType>(0, 10)) {
        REQUIRE(i == ii);
        ++ii;
    }

    ii = 5;
    for (auto i: noa::irange<TestType>(5, 10)) {
        REQUIRE(i == ii);
        ++ii;
    }
}
