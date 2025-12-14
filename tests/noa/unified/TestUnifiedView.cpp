#include <noa/unified/Array.hpp>
#include <noa/unified/View.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;

TEMPLATE_TEST_CASE("unified::View, copy metadata", "", i32, u64, f32, f64, c32, c64) {
    const auto guard = StreamGuard(Device{}, Stream::DEFAULT);

    const auto shape = test::random_shape<isize>(2);
    const Allocator allocator = GENERATE(as<Allocator>(),
        Allocator::DEFAULT,
        Allocator::DEFAULT_ASYNC,
        Allocator::PITCHED,
        Allocator::PINNED,
        Allocator::MANAGED,
        Allocator::MANAGED_GLOBAL);

    // CPU
    auto a = Array<TestType>(shape, {.device={}, .allocator=allocator});
    View va = a.view();
    REQUIRE(va.device().is_cpu());
    REQUIRE(va.allocator() == allocator);
    REQUIRE(va.get());

    Array b = va.to({.device="cpu"});
    REQUIRE(b.device().is_cpu());
    REQUIRE(b.allocator() == Allocator::DEFAULT);
    REQUIRE(b.get());
    REQUIRE(b.get() != a.get());

    // GPU
    if (not Device::is_any_gpu())
        return;

    const auto gpu = Device("gpu:0");
    a = Array<TestType>(shape, {.device="gpu", .allocator=allocator});
    va = a.view();
    b.to(va);
    REQUIRE(va.device() == gpu);
    REQUIRE(va.allocator() == allocator);
    REQUIRE(va.get());
}

TEMPLATE_TEST_CASE("unified::View, copy values", "", i32, u64, f32, f64, c32, c64) {
    auto guard = StreamGuard(Device{}, Stream::ASYNC);
    const auto shape = test::random_shape<isize>(3);
    const auto input = Array<TestType>(shape, {.allocator="managed"});
    const auto view = input.view();

    // arange
    using real_t = noa::traits::value_type_t<TestType>;
    for (isize i{}; auto& e: view.span_1d())
        e = static_cast<real_t>(i++);

    AND_THEN("cpu -> cpu") {
        const auto output = view.copy();
        REQUIRE(test::allclose_abs(input, output, 1e-10));
    }

    AND_THEN("cpu -> gpu") {
        if (Device::is_any(Device::GPU)) {
            const auto output = view.to({.device="gpu", .allocator="managed"});
            REQUIRE(test::allclose_abs(input, output, 1e-10));
        }
    }

    AND_THEN("gpu -> gpu") {
        if (Device::is_any(Device::GPU)) {
            const auto output0 = view.to({.device="gpu", .allocator="managed"});
            const auto output1 = output0.copy();
            REQUIRE(test::allclose_abs(output0, output1, 1e-10));
        }
    }

    AND_THEN("gpu -> cpu") {
        if (Device::is_any(Device::GPU)) {
            const auto output0 = view.to({.device="gpu", .allocator="managed"});
            const auto output1 = output0.to_cpu();
            REQUIRE(test::allclose_abs(output0, output1, 1e-10));
        }
    }
}

TEMPLATE_TEST_CASE("unified::View, shape manipulation", "", i32, u64, f32, f64, c32, c64) {
    AND_THEN("as another type") {
        Array<f64> buffer({2, 3, 4, 5});
        View<f64> c = buffer.view();
        View<unsigned char> d = c.reinterpret_as<unsigned char>();
        REQUIRE(d.shape() == Shape4{2, 3, 4, 40});
        REQUIRE(d.strides() == Strides4{480, 160, 40, 1});

        Array<c64> e({2, 3, 4, 5});
        View f = e.view().reinterpret_as<f64>();
        REQUIRE(f.shape() == Shape4{2, 3, 4, 10});
        REQUIRE(f.strides() == Strides4{120, 40, 10, 1});

        auto g = f.reinterpret_as<c64>();
        REQUIRE(g.shape() == Shape4{2, 3, 4, 5});
        REQUIRE(g.strides() == Strides4{60, 20, 5, 1});
    }

    AND_THEN("reshape") {
        const Array<TestType> buffer({4, 10, 50, 30});
        auto a = buffer.view();
        a = a.flat();
        REQUIRE(a.strides() == a.shape().strides());
        REQUIRE((a.shape().is_vector() and a.shape().ndim() == 1));
        a = a.reshape({4, 10, 50, 30});
        REQUIRE(a.strides() == a.shape().strides());
        a = a.reshape({10, 4, 30, 50});
        REQUIRE(a.strides() == a.shape().strides());
        REQUIRE(a.shape() == Shape4{10, 4, 30, 50});
    }

    AND_THEN("permute") {
        Array<TestType> buffer({4, 10, 50, 30});
        const auto a = buffer.view();
        View<const TestType> b = a.permute({0, 1, 2, 3});
        REQUIRE(b.shape() == Shape4{4, 10, 50, 30});
        REQUIRE(b.strides() == Strides4{15000, 1500, 30, 1});

        b = a.permute({1, 0, 3, 2});
        REQUIRE(b.shape() == Shape4{10, 4, 30, 50});
        REQUIRE(b.strides() == Strides4{1500, 15000, 1, 30});

        const auto c = a.permute_copy({1, 0, 3, 2});
        REQUIRE(c.shape() == Shape4{10, 4, 30, 50});
        REQUIRE(c.strides() == Strides4{6000, 1500, 50, 1});
    }
}
