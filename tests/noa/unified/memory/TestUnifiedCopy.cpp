#include <noa/unified/Array.h>
#include <noa/unified/memory/Copy.h>
#include <noa/unified/memory/Factory.h>

#include <catch2/catch.hpp>

#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::memory::copy", "[noa][unified]", int32_t, float, double, cfloat_t) {
    const size4_t shape = test::getRandomShapeBatched(3);
    Device cpu("cpu");
    StreamGuard stream(cpu, Stream::DEFAULT); // CPU queue is synchronous

    AND_THEN("cpu -> cpu") {
        Array<TestType> a(shape);
        memory::fill(a, TestType{3});

        Array<TestType> b(shape);
        memory::copy(a, b);
        REQUIRE(test::Matcher(test::MATCH_ABS, a, b, 1e-8));

        memory::fill(a, TestType{4});
        b = a.to(cpu);
        REQUIRE(test::Matcher(test::MATCH_ABS, a, b, 1e-8));
    }

    if (!Device::any(Device::GPU))
        return;

    AND_THEN("cpu -> gpu -> gpu -> cpu") {
        Array<TestType> a(shape);
        memory::fill(a, TestType{3});

        Device gpu("gpu");
        Array<TestType> b(shape, {gpu, Allocator::DEFAULT_ASYNC});
        Array<TestType> c(shape, {gpu, Allocator::PITCHED});
        memory::copy(a, b);
        memory::copy(b, c);

        Array<TestType> d = c.to(cpu);
        REQUIRE(test::Matcher(test::MATCH_ABS, a, d, 1e-8));
    }
}

TEST_CASE("unified::memory::copy, strided data from GPU to CPU", "[noa][unified]") {
    if (!Device::any(Device::GPU))
        return;

    const auto shape = dim4_t{1, 10, 10, 10};
    const auto gpu_array_full = memory::arange(shape, 0, 1, Device("gpu"));

    // Select top half
    using namespace ::noa::indexing;
    const auto gpu_array_top = gpu_array_full.subregion(ellipsis_t{}, slice_t{5, 10}, full_extent_t{});
    REQUIRE(all(gpu_array_top.shape() == dim4_t{1, 10, 5, 10}));

    // Try to copy to CPU
    const auto cpu_array_top = gpu_array_top.to(Device("cpu"));
    REQUIRE(all(cpu_array_top.shape() == dim4_t{1, 10, 5, 10}));

    // Check the copy was successful.
    const auto cpu_array_full = memory::arange(shape, 0, 1);
    const auto cpu_array_expected_top = cpu_array_full.subregion(ellipsis_t{}, slice_t{5, 10}, full_extent_t{});
    REQUIRE(test::Matcher(test::MATCH_ABS, cpu_array_top, cpu_array_expected_top, 1e-7));
}
