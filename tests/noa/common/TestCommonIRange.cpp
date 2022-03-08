#include <noa/common/Irange.h>

#include <catch2/catch.hpp>

using namespace ::noa;

TEMPLATE_TEST_CASE("irange()", "[noa][common]", int32_t, uint32_t, int64_t, uint64_t) {
    TestType ii = 0;
    for (auto i: irange<TestType>(10)) {
        REQUIRE(i == ii);
        ++ii;
    }

    ii = 0;
    for (auto i: irange<TestType>(0, 10)) {
        REQUIRE(i == ii);
        ++ii;
    }

    ii = 5;
    for (auto i: irange<TestType>(5, 10)) {
        REQUIRE(i == ii);
        ++ii;
    }
}
