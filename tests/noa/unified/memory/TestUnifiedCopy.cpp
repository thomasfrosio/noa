#include <noa/unified/Array.hpp>
#include <noa/unified/memory/Copy.hpp>
#include <noa/unified/memory/Factory.hpp>

#include <catch2/catch.hpp>

#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::memory::copy", "[noa][unified]", i32, f32, f64, c32) {
    const auto shape = test::get_random_shape4_batched(3);
    const Device cpu("cpu");
    const StreamGuard stream(cpu, StreamMode::DEFAULT); // CPU queue is synchronous

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

    if (!Device::is_any(DeviceType::GPU))
        return;

    AND_THEN("cpu -> gpu -> gpu -> cpu") {
        const Array<TestType> a(shape);
        memory::fill(a, TestType{3});

        const Device gpu("gpu");
        const Array<TestType> b(shape, {gpu, Allocator::DEFAULT_ASYNC});
        const Array<TestType> c(shape, {gpu, Allocator::PITCHED});
        memory::copy(a, b);
        memory::copy(b, c);

        const Array<TestType> d = c.to(cpu);
        REQUIRE(test::Matcher(test::MATCH_ABS, a, d, 1e-8));
    }
}

TEST_CASE("unified::memory::copy, strided data from GPU to CPU", "[noa][unified]") {
    if (!Device::is_any(DeviceType::GPU))
        return;

    const auto shape = Shape4<i64>{1, 10, 10, 10};
    const auto gpu_array_full = memory::arange(shape, 0, 1, Device("gpu"));

    // Select top half
    using namespace ::noa::indexing;
    const auto gpu_array_top = gpu_array_full.subregion(Ellipsis{}, Slice{5, 10}, FullExtent{});
    REQUIRE(all(gpu_array_top.shape() == Shape4<i64>{1, 10, 5, 10}));

    // Try to copy to CPU
    const auto cpu_array_top = gpu_array_top.to(Device("cpu"));
    REQUIRE(all(cpu_array_top.shape() == Shape4<i64>{1, 10, 5, 10}));

    // Check the copy was successful.
    const auto cpu_array_full = memory::arange(shape, 0, 1);
    const auto cpu_array_expected_top = cpu_array_full.subregion(Ellipsis{}, Slice{5, 10}, FullExtent{});
    REQUIRE(test::Matcher(test::MATCH_ABS, cpu_array_top, cpu_array_expected_top, 1e-7));
}
