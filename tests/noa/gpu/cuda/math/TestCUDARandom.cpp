#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/PtrManaged.h>
#include <noa/gpu/cuda/memory/Set.h>
#include <noa/gpu/cuda/math/Random.h>
#include <noa/gpu/cuda/math/Reduce.h>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

TEMPLATE_TEST_CASE("cuda::math::randomize() - contiguous", "[noa][cuda][math]", float, double) {
    using value_t = noa::traits::value_type_t<TestType>;
    const size4_t shape = test::getRandomShapeBatched(3);
    const size4_t stride = shape.stride();
    const size_t elements = shape.elements();
    cuda::memory::PtrManaged<TestType> data{elements};
    test::memset(data.get(), elements, 20);

    cuda::Stream stream;
    cuda::math::randomize(math::uniform_t{}, data.share(), stride, shape, value_t{-10}, value_t{10}, stream);
    TestType min = cuda::math::min(data.share(), stride, shape, stream);
    TestType max = cuda::math::max(data.share(), stride, shape, stream);
    TestType mean = cuda::math::mean(data.share(), stride, shape, stream);
    REQUIRE(min >= value_t{-10});
    REQUIRE(max <= value_t{10});
    REQUIRE_THAT(mean, Catch::WithinAbs(0, 0.1));

    test::memset(data.get(), elements, 20);
    cuda::math::randomize(math::normal_t{}, data.share(), stride, shape, value_t{5}, value_t{2}, stream);
    mean = cuda::math::mean(data.share(), stride, shape, stream);
    value_t stddev = cuda::math::std(data.share(), stride, shape, stream);
    REQUIRE_THAT(mean, Catch::WithinAbs(5, 0.1));
    REQUIRE_THAT(stddev, Catch::WithinAbs(2, 0.1));
}

TEMPLATE_TEST_CASE("cuda::math::randomize() - padded", "[noa][cuda][math]", float, double) {
    using value_t = noa::traits::value_type_t<TestType>;
    const size4_t shape = test::getRandomShapeBatched(3);
    const size_t elements = shape.elements();
    cuda::memory::PtrDevicePadded<TestType> data{shape};

    cuda::Stream stream;
    cuda::memory::set(data.get(), elements, TestType{20}, stream);

    cuda::math::randomize(math::uniform_t{}, data.share(), data.stride(), shape, value_t{-10}, value_t{10}, stream);
    TestType min = cuda::math::min(data.share(), data.stride(), shape, stream);
    TestType max = cuda::math::max(data.share(), data.stride(), shape, stream);
    TestType mean = cuda::math::mean(data.share(), data.stride(), shape, stream);
    REQUIRE(min >= value_t{-10});
    REQUIRE(max <= value_t{10});
    REQUIRE_THAT(mean, Catch::WithinAbs(0, 0.1));

    cuda::memory::set(data.get(), elements, TestType{20}, stream);
    cuda::math::randomize(math::normal_t{}, data.share(), data.stride(), shape, value_t{5}, value_t{2}, stream);
    mean = cuda::math::mean(data.share(), data.stride(), shape, stream);
    value_t stddev = cuda::math::std(data.share(), data.stride(), shape, stream);
    REQUIRE_THAT(mean, Catch::WithinAbs(5, 0.1));
    REQUIRE_THAT(stddev, Catch::WithinAbs(2, 0.1));
}
