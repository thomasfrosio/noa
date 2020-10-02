/*
 * Test noa/utils/Assert.h
 */

#include <catch2/catch.hpp>

#include "noa/utils/Assert.h"


SCENARIO("Noa::Assert::isAlmostEqual should give us a correct float-point comparison",
         "[noa][assert]") {
    //template<typename T, int ulp = 4>
    //    constexpr bool areAlmostEqual(T x, T y, T epsilon = 1e-6)
    using namespace Noa::Assert;

    GIVEN("two float-points that should be equal or _almost_ equal with default epsilon") {
        std::vector<float> x1 = {1.f, -1.22f};
        std::vector<float> y1 = {1.f, 1.22f};
        for (size_t i = 0; i < x1.size(); ++i) {
            REQUIRE(areAlmostEqual(x1[i], y1[i]));
        }

        std::vector<double> x2 = {};
        std::vector<double> y2 = {};
        for (size_t i = 0; i < x2.size(); ++i) {
            REQUIRE(areAlmostEqual(x2[i], y2[i]));
        }

        std::vector<long double> x3 = {};
        std::vector<long double> y3 = {};
        for (size_t i = 0; i < x3.size(); ++i) {
            REQUIRE(areAlmostEqual(x3[i], y3[i]));
        }

    }

}
