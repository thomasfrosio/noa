#include <noa/unified/Array.hpp>
#include <noa/unified/Ewise.hpp>
#include <noa/unified/Reduce.hpp>
#include <noa/unified/math/Random.hpp>
#include <noa/unified/math/Reduce.hpp>
#include <noa/unified/memory/Factory.hpp>

#include <catch2/catch.hpp>

#include "Helpers.h"
#include "noa/core/types/Functors.hpp"

using namespace ::noa;

namespace {

}

TEST_CASE("unified::reduce_unary", "[noa][unified]") {
    const auto shape = test::get_random_shape4_batched(3);
    const auto input = noa::math::random<f32>(noa::math::uniform_t{}, shape, 0, 500);

    const auto result = noa::reduce_unary(input, f32{0}, {}, noa::max_t{}, {});
    const auto expected = noa::math::max(input);
    REQUIRE(result == expected);
}

TEST_CASE("unified::reduce_binary", "[noa][unified]") {
    const auto shape = test::get_random_shape4_batched(3);
    const auto input = noa::math::random<f32>(noa::math::uniform_t{}, shape, -5, 5);

    const f64 offset = 1.;
    const auto result = noa::reduce_binary(input, input, f64{0},
                                           [offset](f32 lhs, f32 rhs) { return static_cast<f64>(lhs * rhs) + offset; },
                                           noa::plus_t{},
                                           [](f64 lhs) { return static_cast<f32>(lhs); });

    noa::ewise_unary(input, input, noa::square_t{});
    noa::ewise_binary(input, offset, input, noa::plus_t{});
    const auto expected = noa::math::sum(input);
    REQUIRE(result == expected);
}
