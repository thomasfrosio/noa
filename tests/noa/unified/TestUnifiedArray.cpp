#include <noa/unified/Array.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/io/ImageFile.hpp>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::Array, allocate", "[noa][unified]", i32, f32, c32, Vec4<i32>, Double22) {
    StreamGuard guard(Device{}, StreamMode::DEFAULT);
    Array<TestType> a;
    REQUIRE(a.is_empty());

    const auto shape = test::get_random_shape4_batched(2);
    const Allocator alloc = GENERATE(Allocator::DEFAULT,
                                     Allocator::DEFAULT_ASYNC,
                                     Allocator::PITCHED,
                                     Allocator::PINNED,
                                     Allocator::MANAGED,
                                     Allocator::MANAGED_GLOBAL);

    // CPU
    a = Array<TestType>(shape, {Device{}, alloc});
    REQUIRE(a.device().is_cpu());
    REQUIRE(a.allocator() == alloc);
    REQUIRE(all(a.shape() == shape));
    REQUIRE(a.get());
    REQUIRE_FALSE(a.is_empty());

    // GPU
    if (!Device::is_any(DeviceType::GPU))
        return;

    const Device gpu("gpu:0");
    Array<TestType> b(shape, ArrayOption{}.set_device(gpu).set_allocator(alloc));
    REQUIRE(b.device().is_gpu());
    REQUIRE(b.allocator() == alloc);
    REQUIRE(all(b.shape() == shape));
    REQUIRE(b.get());
    REQUIRE_FALSE(b.is_empty());

    if (alloc == Allocator::PINNED ||
        alloc == Allocator::MANAGED ||
        alloc == Allocator::MANAGED_GLOBAL) {
        Array<TestType> c = a.as(DeviceType::GPU);
        REQUIRE(c.device() == gpu);
        c = b.as(DeviceType::CPU);
        REQUIRE(c.device() == Device{});
    } else {
        REQUIRE_THROWS_AS(a.as(DeviceType::GPU), noa::Exception);
        REQUIRE_THROWS_AS(b.as(DeviceType::CPU), noa::Exception);
    }
}

TEMPLATE_TEST_CASE("unified::Array, copy metadata", "[noa][unified]", i32, u64, f32, f64, c32, c64) {
    StreamGuard guard(Device{}, StreamMode::DEFAULT);
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
    REQUIRE(a.device().is_cpu());
    REQUIRE(a.allocator() == alloc);
    REQUIRE(a.get());

    Array<TestType> b = a.to(Device(DeviceType::CPU));
    REQUIRE(b.device().is_cpu());
    REQUIRE(b.allocator() == Allocator::DEFAULT);
    REQUIRE(b.get());
    REQUIRE(b.get() != a.get());

    // GPU
    if (!Device::is_any(DeviceType::GPU))
        return;
    const Device gpu("gpu:0");
    a = Array<TestType>(shape, ArrayOption{}.set_device(gpu).set_allocator(alloc));
    REQUIRE(a.device().is_gpu());
    REQUIRE(a.allocator() == alloc);
    REQUIRE(a.get());

    b = a.to(gpu);
    REQUIRE(b.device().is_gpu());
    REQUIRE(b.allocator() == Allocator::DEFAULT);
    REQUIRE(b.get());
    REQUIRE(b.get() != a.get());

    a = b.to(Device(DeviceType::CPU));
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
    StreamGuard guard(Device{}, StreamMode::DEFAULT);
    const auto shape = test::get_random_shape4_batched(3);
    const auto input = Array<TestType>(shape, Allocator::MANAGED);

    // arange
    const auto input_accessor_1d = input.accessor_contiguous_1d();
    for (i64 i = 0; i < input.elements(); ++i)
        input_accessor_1d[i] = static_cast<TestType>(i);

    AND_THEN("cpu -> cpu") {
        const auto output = input.copy();
        REQUIRE(test::Matcher(test::MATCH_ABS, input, output, 1e-10));
    }

    AND_THEN("cpu -> gpu") {
        if (Device::is_any(DeviceType::GPU)) {
            const auto dst_options = ArrayOption(Device(DeviceType::GPU), Allocator::MANAGED);
            const auto output = input.to(dst_options);
            REQUIRE(test::Matcher(test::MATCH_ABS, input, output, 1e-10));
        }
    }

    AND_THEN("gpu -> gpu") {
        if (Device::is_any(DeviceType::GPU)) {
            const auto dst_options = ArrayOption(Device(DeviceType::GPU), Allocator::MANAGED);
            const auto output0 = input.to(dst_options);
            const auto output1 = output0.copy();
            REQUIRE(test::Matcher(test::MATCH_ABS, output0, output1, 1e-10));
        }
    }

    AND_THEN("gpu -> cpu") {
        if (Device::is_any(DeviceType::GPU)) {
            const auto dst_options = ArrayOption(Device(DeviceType::GPU), Allocator::MANAGED);
            const auto output0 = input.to(dst_options);
            const auto output1 = output0.to_cpu();
            REQUIRE(test::Matcher(test::MATCH_ABS, output0, output1, 1e-10));
        }
    }
}

TEMPLATE_TEST_CASE("unified::Array, shape manipulation", "[noa][unified]", i32, u64, f32, f64, c32, c64) {
    StreamGuard guard(Device{}, StreamMode::DEFAULT);
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

TEST_CASE("unified::io, quick load and save", "[noa][unified][io]") {
    const Path directory = fs::current_path() / "test_unified_io";
    const Path file_path = directory / "test_quick_save.mrc";
    fs::create_directory(directory);

    const auto shape = test::get_random_shape4_batched(2);
    const Array input = memory::linspace<f32>(shape, -10, 10);
    io::save(input, file_path);

    const Array output = io::load_data<f32>(file_path);

    REQUIRE(all(input.shape() == output.shape()));
    REQUIRE(all(input.strides() == output.strides()));
    REQUIRE(test::Matcher(test::MATCH_ABS, input.get(), output.get(), shape.elements(), 1e-7));

    std::error_code er;
    fs::remove_all(directory, er); // silence possible error
}

TEST_CASE("unified::Array, overlap", "[noa][unified]") {
    Array<f32> lhs;
    Array<f32> rhs;

    REQUIRE_FALSE(indexing::are_overlapped(lhs, rhs));

    lhs = Array<f32>(4);
    REQUIRE_FALSE(indexing::are_overlapped(lhs, rhs));
    rhs = Array<f32>(4);
    REQUIRE_FALSE(indexing::are_overlapped(lhs, rhs));

    rhs = lhs.subregion(indexing::Ellipsis{}, 1);
    REQUIRE(indexing::are_overlapped(lhs, rhs));
    REQUIRE(indexing::are_overlapped(rhs, lhs));
}

TEST_CASE("unified::Array, accessor", "[noa][unified]") {
    StreamGuard guard(Device{"cpu"}, StreamMode::DEFAULT);
    Array<f32> lhs({9, 10, 11, 12});

    const auto accessor_1d = lhs.accessor_contiguous_1d();
    for (i64 i = 0; i < lhs.elements(); ++i)
        accessor_1d[i] = static_cast<f32>(i);

    const i64 offset = indexing::at(3, 5, 1, 10, lhs.strides());
    REQUIRE(accessor_1d(offset) == static_cast<f32>(offset));

    const auto [accessor_byte, a_shape] = lhs.accessor_and_shape<unsigned char, 4>();
    for (i64 i = 0; i < a_shape[0]; ++i)
        for (i64 j = 0; j < a_shape[1]; ++j)
            for (i64 k = 0; k < a_shape[2]; ++k)
                for (i64 l = 0; l < a_shape[3]; ++l)
                    accessor_byte[i][j][k][l] = 0;
    REQUIRE(test::Matcher(test::MATCH_ABS, lhs, 0.f, 1e-10));
}
