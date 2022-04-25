#include <noa/gpu/cuda/memory/PtrManaged.h>
#include <noa/gpu/cuda/math/Random.h>
#include <noa/gpu/cuda/math/Reduce.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::math::randomize() - all", "[noa][cuda][math]", float, double) {
    using value_t = noa::traits::value_type_t<TestType>;
    const size4_t shape = test::getRandomShapeBatched(3);
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();
    cuda::memory::PtrManaged<TestType> data{elements};

    cuda::Stream stream;
    cuda::math::randomize(math::Uniform{}, data.share(), stride, shape, value_t{-10}, value_t{10}, stream);
    TestType min = cuda::math::min(data.share(), stride, shape, stream);
    TestType max = cuda::math::max(data.share(), stride, shape, stream);
    TestType mean = cuda::math::mean(data.share(), stride, shape, stream);
    REQUIRE(min >= value_t{-10});
    REQUIRE(max <= value_t{10});
    REQUIRE_THAT(mean, Catch::WithinAbs(0, 0.1));

    cuda::math::randomize(math::Normal{}, data.share(), stride, shape, value_t{5}, value_t{2}, stream);
    mean = cuda::math::mean(data.share(), stride, shape, stream);
    value_t stddev = cuda::math::std(data.share(), stride, shape, stream);
    REQUIRE_THAT(mean, Catch::WithinAbs(5, 0.1));
    REQUIRE_THAT(stddev, Catch::WithinAbs(2, 0.1));
}
