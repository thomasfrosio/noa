#include <noa/runtime/Array.hpp>
#include <noa/runtime/Factory.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;

TEMPLATE_TEST_CASE("runtime::copy", "", i32, f32, f64, c32) {
    const auto shape = test::random_shape_batched(3);
    const auto cpu = Device("cpu");
    const auto stream = StreamGuard(cpu, Stream::DEFAULT); // CPU queue is synchronous

    AND_THEN("cpu -> cpu") {
        Array<TestType> a(shape);
        noa::fill(a, {3});

        Array<TestType> b(shape);
        noa::copy(a, b);
        REQUIRE(test::allclose_abs(a, b, 1e-8));

        noa::fill(a, {4});
        b = a.to({.device=cpu});
        REQUIRE(test::allclose_abs(a, b, 1e-8));
    }

    if (not Device::is_any(Device::GPU))
        return;

    AND_THEN("cpu -> gpu -> gpu -> cpu") {
        const Array<TestType> a(shape);
        noa::fill(a, {3});

        const Device gpu("gpu");
        const Array<TestType> b(shape, {gpu, Allocator::ASYNC});
        const Array<TestType> c(shape, {gpu, Allocator::PITCHED});
        noa::copy(a, b);
        noa::copy(b, c);

        const Array<TestType> d = c.to({.device=cpu});
        REQUIRE(test::allclose_abs(a, d, 1e-7));
    }
}

TEST_CASE("runtime::copy, strided data from GPU to CPU") {
    if (not Device::is_any(Device::GPU))
        return;

    const auto shape = Shape4{1, 10, 10, 10};
    const auto gpu_array_full = noa::arange(shape, noa::Arange{0, 1}, {.device="gpu"});

    // Select top half
    const auto gpu_array_top = gpu_array_full.subregion(Ellipsis{}, Slice{5, 10}, Full{});
    REQUIRE(gpu_array_top.shape() == Shape4{1, 10, 5, 10});

    // Try to copy to CPU
    const auto cpu_array_top = gpu_array_top.to({.device="cpu"});
    REQUIRE(cpu_array_top.shape() == Shape4{1, 10, 5, 10});

    // Check the copy was successful.
    const auto cpu_array_full = noa::arange(shape, noa::Arange<i32>{});
    const auto cpu_array_expected_top = cpu_array_full.subregion(Ellipsis{}, Slice{5, 10}, Full{});
    REQUIRE(test::allclose_abs(cpu_array_top, cpu_array_expected_top, 1e-7));
}
