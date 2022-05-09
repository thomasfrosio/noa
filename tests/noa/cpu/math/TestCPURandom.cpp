#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Random.h>
#include <noa/cpu/math/Reduce.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cpu::math::randomize() - all", "[noa][cpu][math]", float, double) {
    using value_t = noa::traits::value_type_t<TestType>;
    const size4_t shape = test::getRandomShapeBatched(3);
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();
    cpu::memory::PtrHost<TestType> data{elements};
    test::memset(data.get(), elements, 20);

    cpu::Stream stream;
    cpu::math::randomize(math::uniform_t{}, data.share(), stride, shape, value_t{-10}, value_t{10}, stream);
    TestType min = cpu::math::min(data.share(), stride, shape, stream);
    TestType max = cpu::math::max(data.share(), stride, shape, stream);
    TestType mean = cpu::math::mean(data.share(), stride, shape, stream);
    REQUIRE(min >= value_t{-10});
    REQUIRE(max <= value_t{10});
    REQUIRE_THAT(mean, Catch::WithinAbs(0, 0.1));

    test::memset(data.get(), elements, 20);
    cpu::math::randomize(math::normal_t{}, data.share(), stride, shape, value_t{5}, value_t{2}, stream);
    mean = cpu::math::mean(data.share(), stride, shape, stream);
    value_t stddev = cpu::math::std(data.share(), stride, shape, stream);
    REQUIRE_THAT(mean, Catch::WithinAbs(5, 0.1));
    REQUIRE_THAT(stddev, Catch::WithinAbs(2, 0.1));
}
