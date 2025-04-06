#include <noa/core/math/Comparison.hpp>

#include "Catch.hpp"

TEST_CASE("core::allclose - integer") {
    REQUIRE(noa::allclose(1, 1));
    REQUIRE(noa::allclose(1, 2, 1));
    REQUIRE(noa::allclose(-1, 1, 2));
    REQUIRE_FALSE(noa::allclose(1, 2));
    REQUIRE_FALSE(noa::allclose(0, 2, 1));
    REQUIRE_FALSE(noa::allclose(-1, 1, 1));
}

TEST_CASE("core::allclose - real") {
    REQUIRE(noa::allclose(1. / 2., 0.5));
    REQUIRE(noa::allclose(1., std::nextafter(1., 2.)));
    REQUIRE_FALSE(noa::allclose(1., 1.1));
    REQUIRE_FALSE(noa::allclose(0., 0.51, .5));
}
