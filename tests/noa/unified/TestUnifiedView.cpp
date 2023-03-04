#include <noa/unified/Array.hpp>
#include <noa/unified/View.hpp>
#include <noa/unified/io/ImageFile.hpp>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::View, copy metadata", "[noa][unified]", i32, u64, f32, f64, c32, c64) {
    const auto shape = test::get_random_shape4_batched(2);
    const Allocator alloc = GENERATE(Allocator::DEFAULT,
                                     Allocator::DEFAULT_ASYNC,
                                     Allocator::PITCHED,
                                     Allocator::PINNED,
                                     Allocator::MANAGED,
                                     Allocator::MANAGED_GLOBAL);
    INFO(alloc);

    // CPU
    Array<TestType> a(shape, {Device{}, alloc});
    View<TestType> va = a.view();
    REQUIRE(va.device().is_cpu());
    REQUIRE(va.allocator() == alloc);
    REQUIRE(va.get());

    Array<TestType> b = va.to(Device(DeviceType::CPU));
    REQUIRE(b.device().is_cpu());
    REQUIRE(b.allocator() == Allocator::DEFAULT);
    REQUIRE(b.get());
    REQUIRE(b.get() != a.get());

    // GPU
    if (!Device::any(DeviceType::GPU))
        return;

    const Device gpu("gpu:0");
    a = Array<TestType>(shape, ArrayOption(gpu, alloc));
    va = a.view();
    b.to(va);
    REQUIRE(va.device().is_gpu());
    REQUIRE(va.allocator() == alloc);
    REQUIRE(va.get());
}


TEMPLATE_TEST_CASE("unified::View, copy values", "[noa][unified]", i32, u64, f32, f64, c32, c64) {
    const auto shape = test::get_random_shape4_batched(3);
    const auto input = Array<TestType>(shape, Allocator::MANAGED);
    const auto view = input.view();

    // arange
    const auto input_accessor_1d = view.accessor_contiguous_1d();
    for (i64 i = 0; i < view.elements(); ++i)
        input_accessor_1d[i] = static_cast<TestType>(i);

    AND_THEN("cpu -> cpu") {
        const auto output = view.copy();
        REQUIRE(test::Matcher(test::MATCH_ABS, input, output, 1e-10));
    }

    AND_THEN("cpu -> gpu") {
        if (Device::any(DeviceType::GPU)) {
            const auto dst_options = ArrayOption(Device(DeviceType::GPU), Allocator::MANAGED);
            const auto output = view.to(dst_options);
            REQUIRE(test::Matcher(test::MATCH_ABS, input, output, 1e-10));
        }
    }

    AND_THEN("gpu -> gpu") {
        if (Device::any(DeviceType::GPU)) {
            const auto dst_options = ArrayOption(Device(DeviceType::GPU), Allocator::MANAGED);
            const auto output0 = view.to(dst_options);
            const auto output1 = output0.copy();
            REQUIRE(test::Matcher(test::MATCH_ABS, output0, output1, 1e-10));
        }
    }

    AND_THEN("gpu -> cpu") {
        if (Device::any(DeviceType::GPU)) {
            const auto dst_options = ArrayOption(Device(DeviceType::GPU), Allocator::MANAGED);
            const auto output0 = view.to(dst_options);
            const auto output1 = output0.to_cpu();
            REQUIRE(test::Matcher(test::MATCH_ABS, output0, output1, 1e-10));
        }
    }
}
