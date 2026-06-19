#include <noa/runtime/Array.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace ::noa::types;

namespace {
    struct Simple {
        i32 a[5];
    };
}

TEMPLATE_TEST_CASE("runtime::Array, allocate", "", i32, f32, c32, (Vec<i32, 4>), Simple) {
    constexpr usize N = 4;
    auto guard = StreamGuard(Device{}, Stream::DEFAULT);
    Array<TestType, N> a;
    REQUIRE(a.is_empty());

    const auto shape = test::random_shape<isize, N>(2);
    const Allocator allocator = GENERATE(as<Allocator>(),
        Allocator::DEFAULT,
        Allocator::DEFAULT_ASYNC,
        Allocator::PITCHED,
        Allocator::PINNED,
        Allocator::MANAGED,
        Allocator::MANAGED_GLOBAL,
        Allocator::PITCHED_MANAGED);

    // CPU
    a = Array<TestType, N>(shape, {.device=Device{}, .allocator=allocator});
    REQUIRE(a.device().is_cpu());
    REQUIRE(a.allocator() == allocator);
    REQUIRE(a.shape() == shape);
    REQUIRE(a.get());
    REQUIRE_FALSE(a.is_empty());

    // GPU
    if (not Device::is_any_gpu())
        return;

    auto b = Array<TestType, N>(shape, {"gpu:0", allocator});
    REQUIRE(b.device().is_gpu());
    REQUIRE(b.allocator() == allocator);
    REQUIRE(b.shape() == shape);
    REQUIRE(b.get());
    REQUIRE_FALSE(b.is_empty());

    if (allocator.is_any(Allocator::PINNED, Allocator::MANAGED, Allocator::MANAGED_GLOBAL, Allocator::PITCHED_MANAGED)) {
        auto c = a.reinterpret_as(Device::GPU, {.prefetch = true});
        REQUIRE(c.device().is_gpu());
        c = b.reinterpret_as(Device::CPU, {.prefetch = true});
        REQUIRE(c.device() == Device{});
    } else {
        REQUIRE_THROWS_AS(a.reinterpret_as(Device::GPU), noa::Exception);
        REQUIRE_THROWS_AS(b.reinterpret_as(Device::CPU), noa::Exception);
    }
}

TEMPLATE_TEST_CASE("runtime::Array, copy metadata", "", i32, u64, f32, f64, c32, c64) {
    constexpr usize N = 2;
    auto guard = StreamGuard(Device{}, Stream::DEFAULT);
    const auto shape = test::random_shape<isize, N>(2);
    const auto allocator = GENERATE(as<Allocator>(),
        Allocator::DEFAULT,
        Allocator::DEFAULT_ASYNC,
        Allocator::PITCHED,
        Allocator::PINNED,
        Allocator::MANAGED,
        Allocator::MANAGED_GLOBAL,
        Allocator::PITCHED_MANAGED);

    // CPU
    Array<TestType, N> a(shape, {Device{}, allocator});
    REQUIRE(a.device().is_cpu());
    REQUIRE(a.allocator() == allocator);
    REQUIRE(a.get());

    Array<TestType, N> b = a.to({.device=Device(Device::CPU)});
    REQUIRE(b.device().is_cpu());
    REQUIRE(b.allocator() == Allocator::DEFAULT);
    REQUIRE(b.get());
    REQUIRE(b.get() != a.get());

    // GPU
    if (!Device::is_any(Device::GPU))
        return;
    const Device gpu("gpu:0");
    a = Array<TestType, N>(shape, ArrayOption{}.set_device(gpu).set_allocator(allocator));
    REQUIRE(a.device().is_gpu());
    REQUIRE(a.allocator() == allocator);
    REQUIRE(a.get());

    b = a.to(ArrayOption{.device=gpu});
    REQUIRE(b.device().is_gpu());
    REQUIRE(b.allocator() == Allocator::DEFAULT);
    REQUIRE(b.get());
    REQUIRE(b.get() != a.get());

    a = b.to(ArrayOption{.device=Device(Device::CPU)});
    REQUIRE(a.device().is_cpu());
    REQUIRE(a.allocator() == Allocator::DEFAULT);
    REQUIRE(a.get());
    REQUIRE(a.get() != b.get());

    a = b.to(ArrayOption{}.set_device(gpu).set_allocator(Allocator::PITCHED));
    REQUIRE(a.device().is_gpu());
    REQUIRE(a.allocator() == Allocator::PITCHED);
    REQUIRE(a.get());
    REQUIRE(b.get() != a.get());
}

TEMPLATE_TEST_CASE("runtime::Array, copy values", "", i32, u64, f32, f64, c32, c64) {
    constexpr usize N = 3;
    StreamGuard guard(Device{}, Stream::DEFAULT);
    const auto shape = test::random_shape<isize, N>(3);
    const auto input = Array<TestType, N>(shape, {.allocator="managed"});

    // arange
    using real_t = noa::traits::value_type_t<TestType>;
    for (i64 i{}; auto& e: input.span_1d())
        e = static_cast<real_t>(i++);

    AND_THEN("cpu -> cpu") {
        const auto output = input.copy();
        REQUIRE(test::allclose_abs(input, output, 1e-10));
    }

    AND_THEN("cpu -> gpu") {
        if (Device::is_any(Device::GPU)) {
            const auto output = input.to({.device="gpu", .allocator="managed"});
            REQUIRE(test::allclose_abs(input, output, 1e-10));
        }
    }

    AND_THEN("gpu -> gpu") {
        if (Device::is_any(Device::GPU)) {
            const auto output0 = input.to({.device="gpu", .allocator="managed"});
            const auto output1 = output0.copy();
            REQUIRE(test::allclose_abs(output0, output1, 1e-10));
        }
    }

    AND_THEN("gpu -> cpu") {
        if (Device::is_any(Device::GPU)) {
            const auto output0 = input.to({.device="gpu", .allocator="managed"});
            const auto output1 = output0.to_cpu();
            REQUIRE(test::allclose_abs(output0, output1, 1e-10));
        }
    }
}

TEST_CASE("runtime::Array, .to returns the output") {
    {
        auto a0 = Array<f32, 1>(1);
        auto a1 = Array<f32, 1>(1);
        REQUIRE(a1.get() == std::move(a0).to(a1).get());
    }
    {
        auto a0 = Array<f32, 6>(1);
        auto a1 = Array<f32, 6>(1);
        REQUIRE(a1.get() == std::move(a0).to(a1).get());
    }
}

TEST_CASE("runtime::Array, encapsulate") {
    auto a = Array<f32, 2>({10, 10});
    auto b = Array(a.data(), a.shape());
    static_assert(std::is_same_v<decltype(b), Array<f32, 2, ArrayOwnership::VIEW>>);
    auto c = Array(a.share(), a.shape());
    static_assert(std::is_same_v<decltype(c), Array<f32, 2, ArrayOwnership::RC>>);
    REQUIRE(c.data() == a.data());
}

TEMPLATE_TEST_CASE("runtime::Array, shape manipulation", "", i32, u64, f32, f64, c32, c64) {
    StreamGuard guard(Device{}, Stream::DEFAULT);
    AND_THEN("as another type") {
        Array<f64> c({2, 3, 4, 5});
        Array<unsigned char> d = c.as<unsigned char>();
        REQUIRE(d.shape() == Shape4{2, 3, 4, 40});
        REQUIRE(d.strides() == Strides4{480, 160, 40, 1});

        Array<c64> e({2, 3, 4, 5});
        Array f = e.as<f64>();
        REQUIRE(f.shape() == Shape4{2, 3, 4, 10});
        REQUIRE(f.strides() == Strides4{120, 40, 10, 1});

        e = f.as<c64>();
        REQUIRE(e.shape() == Shape4{2, 3, 4, 5});
        REQUIRE(e.strides() == Strides4{60, 20, 5, 1});
    }

    AND_THEN("reshape") {
        Array<TestType> a({4, 10, 50, 30});
        a = a.flat();
        REQUIRE(a.strides() == a.shape().strides());
        REQUIRE((a.shape() == Shape4{1, 1, 1, 4 * 10 * 50 * 30}));
        a = a.reshape(Shape4{4, 10, 50, 30});
        REQUIRE(a.strides() == a.shape().strides());
        a = a.template reshape<4>({10, 4, 30, 50});
        REQUIRE(a.strides() == a.shape().strides());
        REQUIRE(a.shape() == Shape4{10, 4, 30, 50});
    }

    AND_THEN("permute") {
        Array<TestType> a({4, 10, 50, 30});
        Array<TestType> b = a.permute({0, 1, 2, 3});
        REQUIRE(b.shape() == Shape4{4, 10, 50, 30});
        REQUIRE(b.strides() == Strides4{15000, 1500, 30, 1});

        b = a.permute(Vec{1, 0, 3, 2});
        REQUIRE(b.shape() == Shape4{10, 4, 30, 50});
        REQUIRE(b.strides() == Strides4{1500, 15000, 1, 30});

        b = a.permute_copy(Vec{1, 0, 3, 2});
        REQUIRE(b.shape() == Shape4{10, 4, 30, 50});
        REQUIRE(b.strides() == Strides4{6000, 1500, 50, 1});
    }
}

TEST_CASE("runtime::Array, overlap") {
    Array<f32> lhs;
    Array<f32> rhs;

    REQUIRE_FALSE(noa::are_overlapped(lhs, rhs));

    lhs = Array<f32>(4);
    REQUIRE_FALSE(noa::are_overlapped(lhs, rhs));
    rhs = Array<f32>(4);
    REQUIRE_FALSE(noa::are_overlapped(lhs, rhs));

    rhs = lhs.subregion(noa::Ellipsis{}, 1);
    REQUIRE(noa::are_overlapped(lhs, rhs));
    REQUIRE(noa::are_overlapped(rhs, lhs));
}

TEST_CASE("runtime::Array, nd and span") {
    StreamGuard guard(Device{"cpu"}, Stream::DEFAULT);
    auto lhs = Array<f32, 4>({9, 10, 11, 12});
    REQUIRE((Shape<isize, 1>{11880} == lhs.as_nd<1>().shape()));
    REQUIRE((Shape<isize, 2>{990, 12} == lhs.as_nd<2>().shape()));
    REQUIRE((Shape<isize, 3>{90, 11, 12} == lhs.as_nd<3>().shape()));
    REQUIRE((Shape<isize, 5>{1, 9, 10, 11, 12} == lhs.as_nd<5>().shape()));

    for (i64 i{}; auto& e: lhs.span_1d())
        e = static_cast<f32>(i++);

    const i64 offset = noa::offset_at(lhs.strides(), 3, 5, 1, 10);
    REQUIRE(lhs.span_1d()[offset] == static_cast<f32>(offset));

    const auto span = lhs.span<unsigned char, 4>();
    for (i64 i{}; i < span.shape()[0]; ++i)
        for (i64 j{}; j < span.shape()[1]; ++j)
            for (i64 k{}; k < span.shape()[2]; ++k)
                for (i64 l{}; l < span.shape()[3]; ++l)
                    span(i, j, k, l) = 0;
    REQUIRE(test::allclose_abs(lhs, 0.f, 1e-10));
}

TEST_CASE("runtime::Array, drop") {
    auto a = Array<f32>(10);
    auto b = a.drop();

    REQUIRE(a.is_empty());
    REQUIRE(a.data() == nullptr);

    a = std::move(b).drop();
    REQUIRE(b.is_empty());
    REQUIRE(b.data() == nullptr);
    REQUIRE(a.size() == 10);
}

TEST_CASE("runtime::Array, to const / view") {
    auto a = Array<f32, 1>(10);
    Array<const f32, 1, ArrayOwnership::VIEW> a1 = a.view();
    REQUIRE(a.data() == a1.data());
    REQUIRE(a.shape() == a1.shape());
    REQUIRE(a.device() == a1.device());
    REQUIRE(a.allocator() == a1.allocator());

    Array<const f32, 1, ArrayOwnership::VIEW> a3(a);
    REQUIRE(a3.data() == a1.data());
    REQUIRE(a3.shape() == a1.shape());
    REQUIRE(a3.device() == a1.device());
    REQUIRE(a3.allocator() == a1.allocator());

    Array<const f32, 1> b = a;
    auto c = a.as_const();
    static_assert(std::is_const_v<typename decltype(c)::value_type>);
    REQUIRE(c.data() == b.data());
}
