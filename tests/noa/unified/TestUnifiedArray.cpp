#include <noa/unified/Array.hpp>
//#include <noa/unified/memory/Factory.hpp>
//#include <noa/unified/io/ImageFile.hpp>

#include <catch2/catch.hpp>
#include "Utils.hpp"

using namespace ::noa::types;

TEMPLATE_TEST_CASE("unified::Array, allocate", "[noa][unified]", i32, f32, c32, Vec4<i32>, Mat22<f64>) {
    auto guard = StreamGuard(Device{}, Stream::DEFAULT);
    Array<TestType> a;
    REQUIRE(a.is_empty());

    const auto shape = test::random_shape(2);
    const Allocator allocator = GENERATE(as<Allocator>(),
        Allocator::DEFAULT,
        Allocator::DEFAULT_ASYNC,
        Allocator::PITCHED,
        Allocator::PINNED,
        Allocator::MANAGED,
        Allocator::MANAGED_GLOBAL);

    // CPU
    a = Array<TestType>(shape, {.device=Device{}, .allocator=allocator});
    REQUIRE(a.device().is_cpu());
    REQUIRE(a.allocator() == allocator);
    REQUIRE(all(a.shape() == shape));
    REQUIRE(a.get());
    REQUIRE_FALSE(a.is_empty());

    // GPU
    if (not Device::is_any_gpu())
        return;

    Array<TestType> b(shape, {"gpu:0", allocator});
    REQUIRE(b.device().is_gpu());
    REQUIRE(b.allocator() == allocator);
    REQUIRE(all(b.shape() == shape));
    REQUIRE(b.get());
    REQUIRE_FALSE(b.is_empty());

    if (allocator.is_any(Allocator::PINNED, Allocator::MANAGED, Allocator::MANAGED_GLOBAL)) {
        Array<TestType> c = a.as(Device::GPU, /*prefetch=*/ true);
        REQUIRE(c.device().is_gpu());
        c = b.as(Device::CPU, /*prefetch=*/ true);
        REQUIRE(c.device() == Device{});
    } else {
        REQUIRE_THROWS_AS(a.as(Device::GPU), noa::Exception);
        REQUIRE_THROWS_AS(b.as(Device::CPU), noa::Exception);
    }
}

TEMPLATE_TEST_CASE("unified::Array, copy metadata", "[noa][unified]", i32, u64, f32, f64, c32, c64) {
    StreamGuard guard(Device{}, Stream::DEFAULT);
    const auto shape = test::random_shape(2);
    const auto allocator = GENERATE(as<Allocator>(),
        Allocator::DEFAULT,
        Allocator::DEFAULT_ASYNC,
        Allocator::PITCHED,
        Allocator::PINNED,
        Allocator::MANAGED,
        Allocator::MANAGED_GLOBAL);

    // CPU
    Array<TestType> a(shape, {Device{}, allocator});
    REQUIRE(a.device().is_cpu());
    REQUIRE(a.allocator() == allocator);
    REQUIRE(a.get());

    Array<TestType> b = a.to({.device=Device(Device::CPU)});
    REQUIRE(b.device().is_cpu());
    REQUIRE(b.allocator() == Allocator::DEFAULT);
    REQUIRE(b.get());
    REQUIRE(b.get() != a.get());

    // GPU
    if (!Device::is_any(Device::GPU))
        return;
    const Device gpu("gpu:0");
    a = Array<TestType>(shape, ArrayOption{}.set_device(gpu).set_allocator(allocator));
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

TEMPLATE_TEST_CASE("unified::Array, copy values", "[noa][unified]", i32, u64, f32, f64, c32, c64) {
    StreamGuard guard(Device{}, Stream::DEFAULT);
    const auto shape = test::random_shape(3);
    const auto input = Array<TestType>(shape, {.allocator="managed"});

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

TEMPLATE_TEST_CASE("unified::Array, shape manipulation", "[noa][unified]", i32, u64, f32, f64, c32, c64) {
    StreamGuard guard(Device{}, Stream::DEFAULT);
    AND_THEN("as another type") {
        Array<f64> c({2, 3, 4, 5});
        Array<unsigned char> d = c.as<unsigned char>();
        REQUIRE(all(d.shape() == Shape4<i64>{2, 3, 4, 40}));
        REQUIRE(all(d.strides() == Strides4<i64>{480, 160, 40, 1}));

        Array<c64> e({2, 3, 4, 5});
        Array f = e.as<f64>();
        REQUIRE(all(f.shape() == Shape4<i64>{2, 3, 4, 10}));
        REQUIRE(all(f.strides() == Strides4<i64>{120, 40, 10, 1}));

        e = f.as<c64>();
        REQUIRE(all(e.shape() == Shape4<i64>{2, 3, 4, 5}));
        REQUIRE(all(e.strides() == Strides4<i64>{60, 20, 5, 1}));
    }

    AND_THEN("reshape") {
        Array<TestType> a({4, 10, 50, 30});
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
        Array<TestType> a({4, 10, 50, 30});
        Array<TestType> b = a.permute({0, 1, 2, 3});
        REQUIRE(all(b.shape() == Shape4<i64>{4, 10, 50, 30}));
        REQUIRE(all(b.strides() == Strides4<i64>{15000, 1500, 30, 1}));

        b = a.permute({1, 0, 3, 2});
        REQUIRE(all(b.shape() == Shape4<i64>{10, 4, 30, 50}));
        REQUIRE(all(b.strides() == Strides4<i64>{1500, 15000, 1, 30}));

        b = a.permute_copy({1, 0, 3, 2});
        REQUIRE(all(b.shape() == Shape4<i64>{10, 4, 30, 50}));
        REQUIRE(all(b.strides() == Strides4<i64>{6000, 1500, 50, 1}));
    }
}

//TEST_CASE("unified::io, quick load and save", "[noa][unified][io]") {
//    const Path directory = fs::current_path() / "test_unified_io";
//    const Path file_path = directory / "test_quick_save.mrc";
//    fs::create_directory(directory);
//
//    const auto shape = test::get_random_shape4_batched(2);
//    const Array input = memory::linspace<f32>(shape, -10, 10);
//    io::save(input, file_path);
//
//    const Array output = io::load_data<f32>(file_path);
//
//    REQUIRE(all(input.shape() == output.shape()));
//    REQUIRE(all(input.strides() == output.strides()));
//    REQUIRE(test::Matcher(test::MATCH_ABS, input.get(), output.get(), shape.elements(), 1e-7));
//
//    std::error_code er;
//    fs::remove_all(directory, er); // silence possible error
//}

TEST_CASE("unified::Array, overlap", "[noa][unified]") {
    Array<f32> lhs;
    Array<f32> rhs;

    namespace ni = noa::indexing;
    REQUIRE_FALSE(ni::are_overlapped(lhs, rhs));

    lhs = Array<f32>(4);
    REQUIRE_FALSE(ni::are_overlapped(lhs, rhs));
    rhs = Array<f32>(4);
    REQUIRE_FALSE(ni::are_overlapped(lhs, rhs));

    rhs = lhs.subregion(ni::Ellipsis{}, 1);
    REQUIRE(ni::are_overlapped(lhs, rhs));
    REQUIRE(ni::are_overlapped(rhs, lhs));
}

TEST_CASE("unified::Array, span", "[noa][unified]") {
    StreamGuard guard(Device{"cpu"}, Stream::DEFAULT);
    Array<f32> lhs({9, 10, 11, 12});

    for (i64 i{}; auto& e: lhs.span_1d_contiguous())
        e = static_cast<f32>(i++);

    const i64 offset = noa::indexing::offset_at(lhs.strides(), 3, 5, 1, 10);
    REQUIRE(lhs.span_1d()[offset] == static_cast<f32>(offset));

    const auto span = lhs.span<unsigned char, 4>();
    for (i64 i{}; i < span.shape()[0]; ++i)
        for (i64 j{}; j < span.shape()[1]; ++j)
            for (i64 k{}; k < span.shape()[2]; ++k)
                for (i64 l{}; l < span.shape()[3]; ++l)
                    span(i, j, k, l) = 0;
    REQUIRE(test::allclose_abs(lhs, 0.f, 1e-10));
}
