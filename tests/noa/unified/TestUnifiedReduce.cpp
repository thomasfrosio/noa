#include <noa/unified/Array.hpp>
#include <noa/unified/Ewise.hpp>
#include <noa/unified/Reduce.hpp>
#include <noa/unified/math/Random.hpp>
#include <noa/unified/math/Reduce.hpp>
#include <noa/unified/memory/Factory.hpp>

#include <catch2/catch.hpp>

#include "Helpers.h"

using namespace ::noa;

namespace {

}

TEST_CASE("unified::reduce_unary", "[noa][unified]") {
    const auto shape = test::get_random_shape4_batched(3);
    Array input = noa::math::random<f32>(noa::math::uniform_t{}, shape, 0, 500);

    const auto result = noa::reduce_unary(input, f32{0},
                                          [](f32 v) { return static_cast<f32>(v); },
                                          noa::plus_t{},
                                          [](f32 v) { return static_cast<f32>(v); });
    const auto expected = noa::math::sum(input);
    REQUIRE(result == expected);
}
