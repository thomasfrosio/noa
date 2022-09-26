#include <noa/unified/Array.h>
#include <noa/unified/memory/Factory.h>
#include <noa/unified/io/ImageFile.h>

#include <catch2/catch.hpp>

#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::Array, allocate", "[noa][unified]", int32_t, float, cfloat_t, int4_t, double22_t) {
    Array<TestType> a;
    REQUIRE(a.empty());

    const size4_t shape = test::getRandomShapeBatched(2);
    const Allocator alloc = GENERATE(Allocator::DEFAULT,
                                     Allocator::DEFAULT_ASYNC,
                                     Allocator::PITCHED,
                                     Allocator::PINNED,
                                     Allocator::MANAGED,
                                     Allocator::MANAGED_GLOBAL);

    // CPU
    a = Array<TestType>(shape, {Device{}, alloc});
    REQUIRE(a.device().cpu());
    REQUIRE(a.allocator() == alloc);
    REQUIRE(all(a.shape() == shape));
    REQUIRE(a.get());
    REQUIRE_FALSE(a.empty());

    // GPU
    if (!Device::any(Device::GPU))
        return;
    const Device gpu("gpu:0");
    Array<TestType> b(shape, ArrayOption{}.device(gpu).allocator(alloc));
    REQUIRE(b.device().gpu());
    REQUIRE(b.allocator() == alloc);
    REQUIRE(all(b.shape() == shape));
    REQUIRE(b.get());
    REQUIRE_FALSE(b.empty());

    if (alloc == Allocator::PINNED ||
        alloc == Allocator::MANAGED ||
        alloc == Allocator::MANAGED_GLOBAL) {
        Array<TestType> c = a.as(Device::GPU);
        REQUIRE(c.device() == gpu);
        c = b.as(Device::CPU);
        REQUIRE(c.device() == Device{});
    } else {
        REQUIRE_THROWS_AS(a.as(Device::GPU), noa::Exception);
        REQUIRE_THROWS_AS(b.as(Device::CPU), noa::Exception);
    }
}

TEMPLATE_TEST_CASE("unified::Array, copy", "[noa][unified]",
                   int32_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    const size4_t shape = test::getRandomShapeBatched(2);
    const Allocator alloc = GENERATE(Allocator::DEFAULT,
                                     Allocator::DEFAULT_ASYNC,
                                     Allocator::PITCHED,
                                     Allocator::PINNED,
                                     Allocator::MANAGED,
                                     Allocator::MANAGED_GLOBAL);
    INFO(alloc);

    // CPU
    Array<TestType> a(shape, {Device{}, alloc});
    REQUIRE(a.device().cpu());
    REQUIRE(a.allocator() == alloc);
    REQUIRE(a.get());

    Array<TestType> b = a.to(Device(Device::CPU));
    REQUIRE(b.device().cpu());
    REQUIRE(b.allocator() == Allocator::DEFAULT);
    REQUIRE(b.get());
    REQUIRE(b.get() != a.get());

    // GPU
    if (!Device::any(Device::GPU))
        return;
    const Device gpu("gpu:0");
    a = Array<TestType>{shape, ArrayOption{}.device(gpu).allocator(alloc)};
    REQUIRE(a.device().gpu());
    REQUIRE(a.allocator() == alloc);
    REQUIRE(a.get());

    b = a.to(gpu);
    REQUIRE(b.device().gpu());
    REQUIRE(b.allocator() == Allocator::DEFAULT);
    REQUIRE(b.get());
    REQUIRE(b.get() != a.get());

    a = b.to(Device{Device::CPU});
    REQUIRE(a.device().cpu());
    REQUIRE(a.allocator() == Allocator::DEFAULT);
    REQUIRE(a.get());
    REQUIRE(a.get() != b.get());

    a = b.to(ArrayOption{}.device(gpu).allocator(Allocator::PITCHED));
    REQUIRE(a.device().gpu());
    REQUIRE(a.allocator() == Allocator::PITCHED);
    REQUIRE(a.get());
    REQUIRE(b.get() != a.get());
}

TEMPLATE_TEST_CASE("unified::Array, shape manipulation", "[noa][unified]",
                   int32_t, uint64_t, float, double, cfloat_t, cdouble_t) {
    AND_THEN("as another type") {
        Array<double> c({2, 3, 4, 5});
        Array<unsigned char> d = c.as<unsigned char>();
        REQUIRE(all(d.shape() == size4_t{2, 3, 4, 40}));
        REQUIRE(all(d.strides() == size4_t{480, 160, 40, 1}));

        Array<cdouble_t> e({2, 3, 4, 5});
        Array<double> f = e.as<double>();
        REQUIRE(all(f.shape() == size4_t{2, 3, 4, 10}));
        REQUIRE(all(f.strides() == size4_t{120, 40, 10, 1}));

        e = f.as<cdouble_t>();
        REQUIRE(all(e.shape() == size4_t{2, 3, 4, 5}));
        REQUIRE(all(e.strides() == size4_t{60, 20, 5, 1}));
    }

    AND_THEN("reshape") {
        Array<TestType> a({4, 10, 50, 30});
        a = a.reshape({1, 1, 1, a.elements()});
        REQUIRE(all(a.strides() == a.shape().strides()));
        a = a.reshape({4, 10, 50, 30});
        REQUIRE(all(a.strides() == a.shape().strides()));
        a = a.reshape({10, 4, 30, 50});
        REQUIRE(all(a.strides() == a.shape().strides()));
        REQUIRE(all(a.shape() == size4_t{10, 4, 30, 50}));
    }

    AND_THEN("permute") {
        Array<TestType> a({4, 10, 50, 30});
        Array<TestType> b = a.permute({0, 1, 2, 3});
        REQUIRE(all(b.shape() == size4_t{4, 10, 50, 30}));
        REQUIRE(all(b.strides() == size4_t{15000, 1500, 30, 1}));

        b = a.permute({1, 0, 3, 2});
        REQUIRE(all(b.shape() == size4_t{10, 4, 30, 50}));
        REQUIRE(all(b.strides() == size4_t{1500, 15000, 1, 30}));

        b = a.permute({1, 0, 3, 2}, true);
        REQUIRE(all(b.shape() == size4_t{10, 4, 30, 50}));
        REQUIRE(all(b.strides() == size4_t{6000, 1500, 50, 1}));
    }
}

TEST_CASE("unified::io, quick load and save", "[noa][unified][io]") {
    const path_t directory = fs::current_path() / "test_unified_io";
    const path_t file_path = directory / "test_quick_save.mrc";
    fs::create_directory(directory);

    const size4_t shape = test::getRandomShapeBatched(2);
    Array input = memory::linspace<float>(shape, -10, 10);
    io::save(input, file_path);

    Array output = io::load<float>(file_path);

    REQUIRE(all(input.shape() == output.shape()));
    REQUIRE(all(input.strides() == output.strides()));
    REQUIRE(test::Matcher(test::MATCH_ABS, input.get(), output.get(), shape.elements(), 1e-7));

    std::error_code er;
    fs::remove_all(directory, er); // silence possible error
}

TEST_CASE("unified::Array, overlap", "[noa][unified]") {
    Array<float> lhs;
    Array<float> rhs;

    REQUIRE_FALSE(indexing::isOverlap(lhs, rhs));

    lhs = Array<float>(4);
    REQUIRE_FALSE(indexing::isOverlap(lhs, rhs));
    rhs = Array<float>(4);
    REQUIRE_FALSE(indexing::isOverlap(lhs, rhs));

    rhs = lhs.subregion(indexing::ellipsis_t{}, 1);
    REQUIRE(indexing::isOverlap(lhs, rhs));
    REQUIRE(indexing::isOverlap(rhs, lhs));
}

TEST_CASE("unified::Array, view and accessor", "[noa][unified]") {
    Array<float> lhs({9, 10, 11, 12});

    const auto [accessor0, size] = lhs.accessor<float, 1>();
    for (dim_t i = 0; i < size; ++i)
        accessor0[i] = static_cast<float>(i);

    const dim_t offset = indexing::at(3, 5, 1, 10, lhs.strides());
    REQUIRE(accessor0(offset) == static_cast<float>(offset));

    // Just check that it compiles
    const auto [accessor1, a_shape] = lhs.accessor<unsigned char, 4>();
    for (dim_t i = 0; i < a_shape[0]; ++i)
        for (dim_t j = 0; j < a_shape[1]; ++j)
            for (dim_t k = 0; k < a_shape[2]; ++k)
                for (dim_t l = 0; l < a_shape[3]; ++l)
                    accessor1[i][j][k][l] = 0;
    REQUIRE(test::Matcher(test::MATCH_ABS, lhs, 0.f, 1e-10));
}
