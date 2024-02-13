#include <catch2/catch.hpp>
#include <noa/core/utils/Interfaces.hpp>
#include <noa/gpu/cuda/AllocatorManaged.hpp>
#include <noa/gpu/cuda/Iwise.hpp>

#include "Helpers.h"

namespace {
    using namespace noa::types;

    struct Tracked {
        noa::Vec<int, 2> count{};
        constexpr Tracked() = default;
        NOA_HD constexpr Tracked(const Tracked& t) : count(t.count) { count[0] += 1; }
        NOA_HD constexpr Tracked(Tracked&& t)  noexcept : count(t.count) { count[1] += 1; }
    };

    struct Op {
        Vec<int, 2>* ptr;
        Tracked tracked;
        NOA_HD constexpr void operator()(i64 i) const {
            ptr[i] = tracked.count;
        }
    };
}

TEST_CASE("cuda::iwise") {
    using namespace noa::types;
    using noa::cuda::iwise;
    using noa::cuda::Stream;
    using noa::cuda::Device;
    using noa::cuda::AllocatorManaged;

    Stream stream(Device::current());

    AND_THEN("check operator") {
        const auto shape = Shape1<i64>{2};
        auto value = AllocatorManaged<Vec2<int>>::allocate(2, stream);
        auto op0 = Op{value.get(), Tracked{}};

        iwise(shape, op0, stream); // operator is copied once to the kernel
        stream.synchronize();
        REQUIRE((value[0][0] == 1 and value[0][1] == 0));
        REQUIRE((value[1][0] == 1 and value[1][1] == 0));

        iwise(shape, std::move(op0), stream); // operator is moved once into the kernel
        stream.synchronize();
        REQUIRE((value[0][0] == 0 and value[0][1] == 1));
        REQUIRE((value[1][0] == 0 and value[1][1] == 1));
    }

    AND_THEN("1d") {
        const auto shape = Shape1<i64>{10000};
        const auto elements = shape.elements();

        const auto buffer = AllocatorManaged<i32>::allocate(elements, stream);
        iwise(shape, [ptr = buffer.get()] __device__ (i64 i) { ptr[i] = static_cast<i32>(i); }, stream);

        const auto expected = std::make_unique<i32[]>(elements);
        for (i32 i{0}; auto& e: Span(expected.get(), elements))
            e = i++;

        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 1e-6));
    }

    AND_THEN("2d") {
        const auto shape = Shape2<i64>{150, 100};
        const auto elements = shape.elements();

        const auto buffer = AllocatorManaged<i32>::allocate(elements, stream);
        const auto accessor = AccessorContiguousI64<i32, 2>(buffer.get(), shape.strides());
        iwise(shape, [=] __device__ (i64 i, i64 j) { accessor(i, j) = static_cast<i32>(i * 100 + j); }, stream);

        const auto expected = std::make_unique<i32[]>(elements);
        for (i32 i{0}; auto& e: Span(expected.get(), elements))
            e = i++;

        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 0));
    }

    AND_THEN("3d") {
        const auto shape = Shape3<i64>{25, 15, 35};
        const auto elements = shape.elements();

        const auto buffer = AllocatorManaged<i32>::allocate(elements, stream);
        const auto accessor = AccessorContiguousI64<i32, 3>(buffer.get(), shape.strides());
        iwise(shape, [=] __device__(Vec3<i64> indices) {
            accessor(indices) = static_cast<i32>(accessor.offset_at(indices));
        }, stream);

        const auto expected = std::make_unique<i32[]>(elements);
        for (i32 i{0}; auto& e: Span(expected.get(), elements))
            e = i++;

        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 0));
    }

    AND_THEN("4d") {
        const auto shape = Shape4<i64>{3, 5, 25, 60};
        const auto elements = shape.elements();

        const auto buffer = AllocatorManaged<i32>::allocate(elements, stream);
        const auto accessor = AccessorContiguousI64<i32, 4>(buffer.get(), shape.strides());
        iwise(shape, [=] __device__(Vec4<i64> indices) {
            accessor(indices) = static_cast<i32>(accessor.offset_at(indices));
        }, stream);

        const auto expected = std::make_unique<i32[]>(elements);
        for (i32 i{0}; auto& e: Span(expected.get(), elements))
            e = i++;

        stream.synchronize();
        REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 0));
    }
}
