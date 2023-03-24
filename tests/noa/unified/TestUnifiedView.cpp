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
    if (!Device::is_any(DeviceType::GPU))
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
        if (Device::is_any(DeviceType::GPU)) {
            const auto dst_options = ArrayOption(Device(DeviceType::GPU), Allocator::MANAGED);
            const auto output = view.to(dst_options);
            REQUIRE(test::Matcher(test::MATCH_ABS, input, output, 1e-10));
        }
    }

    AND_THEN("gpu -> gpu") {
        if (Device::is_any(DeviceType::GPU)) {
            const auto dst_options = ArrayOption(Device(DeviceType::GPU), Allocator::MANAGED);
            const auto output0 = view.to(dst_options);
            const auto output1 = output0.copy();
            REQUIRE(test::Matcher(test::MATCH_ABS, output0, output1, 1e-10));
        }
    }

    AND_THEN("gpu -> cpu") {
        if (Device::is_any(DeviceType::GPU)) {
            const auto dst_options = ArrayOption(Device(DeviceType::GPU), Allocator::MANAGED);
            const auto output0 = view.to(dst_options);
            const auto output1 = output0.to_cpu();
            REQUIRE(test::Matcher(test::MATCH_ABS, output0, output1, 1e-10));
        }
    }
}

TEMPLATE_TEST_CASE("unified::View, shape manipulation", "[noa][unified]", i32, u64, f32, f64, c32, c64) {
    AND_THEN("as another type") {
        Array<f64> buffer({2, 3, 4, 5});
        View<f64> c = buffer.view();
        View<unsigned char> d = c.as<unsigned char>();
        REQUIRE(all(d.shape() == Shape4<i64>{2, 3, 4, 40}));
        REQUIRE(all(d.strides() == Strides4<i64>{480, 160, 40, 1}));

        Array<c64> e({2, 3, 4, 5});
        View f = e.view().as<f64>();
        REQUIRE(all(f.shape() == Shape4<i64>{2, 3, 4, 10}));
        REQUIRE(all(f.strides() == Strides4<i64>{120, 40, 10, 1}));

        auto g = f.as<c64>();
        REQUIRE(all(g.shape() == Shape4<i64>{2, 3, 4, 5}));
        REQUIRE(all(g.strides() == Strides4<i64>{60, 20, 5, 1}));
    }

    AND_THEN("reshape") {
        const Array<TestType> buffer({4, 10, 50, 30});
        auto a = buffer.view();
        a = a.flat();
        REQUIRE(all(a.strides() == a.shape().strides()));
        REQUIRE((a.shape().is_vector() && a.shape().ndim() == 1));
        a = a.reshape({4, 10, 50, 30});
        REQUIRE(all(a.strides() == a.shape().strides()));
        a = a.reshape({10, 4, 30, 50});
        REQUIRE(all(a.strides() == a.shape().strides()));
        REQUIRE(all(a.shape() == Shape4<i64>{10, 4, 30, 50}));
    }

    AND_THEN("permute") {
        Array<TestType> buffer({4, 10, 50, 30});
        const auto a = buffer.view();
        View<const TestType> b = a.permute({0, 1, 2, 3});
        REQUIRE(all(b.shape() == Shape4<i64>{4, 10, 50, 30}));
        REQUIRE(all(b.strides() == Strides4<i64>{15000, 1500, 30, 1}));

        b = a.permute({1, 0, 3, 2});
        REQUIRE(all(b.shape() == Shape4<i64>{10, 4, 30, 50}));
        REQUIRE(all(b.strides() == Strides4<i64>{1500, 15000, 1, 30}));

        const auto c = a.permute_copy({1, 0, 3, 2});
        REQUIRE(all(c.shape() == Shape4<i64>{10, 4, 30, 50}));
        REQUIRE(all(c.strides() == Strides4<i64>{6000, 1500, 50, 1}));
    }
}
