#include <noa/cpu/Iwise.hpp>
#include <catch2/catch.hpp>

#include "Helpers.h"

namespace {
    struct Tracked {
        std::array<int, 3> count{};
        Tracked() { count[0] += 1; }
        Tracked(const Tracked& t) : count(t.count) { count[1] += 1; }
        Tracked(Tracked&& t)  noexcept : count(t.count) { count[2] += 1; }
    };
}

TEST_CASE("cpu::iwise") {
    using namespace noa::types;
    using noa::cpu::iwise;

    AND_THEN("check operator") {
        const auto shape = Shape1<i64>{2};
        std::array<int, 3> value[2];

        struct Op {
            Tracked tracked;
            std::array<int, 3>* ptr;

            void operator()(i64 i) const {
                ptr[i] = tracked.count;
            }
        };
        auto op0 = Op{Tracked(), value};

        iwise(shape, op0); // operator is copied once to the kernel
        REQUIRE((value[0][0] == 1 and value[0][1] == 1 and value[0][2] == 0));
        REQUIRE((value[1][0] == 1 and value[1][1] == 1 and value[1][2] == 0));

        iwise(shape, std::move(op0)); // operator is moved once into the kernel
        REQUIRE((value[0][0] == 1 and value[0][1] == 0 and value[0][2] == 1));
        REQUIRE((value[1][0] == 1 and value[1][1] == 0 and value[1][2] == 1));
    }

    AND_THEN("1d") {
        const auto shape = Shape1<i64>{100};
        const auto elements = shape.elements();

        const auto buffer = std::make_unique<i32[]>(elements);
        iwise(shape, [ptr = buffer.get()](i64 i) { ptr[i] = static_cast<i32>(i); });

        const auto expected = std::make_unique<i32[]>(elements);
        for (i32 i{0}; auto& e: Span(expected.get(), elements))
            e = i++;

        REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 1e-6));
    }

    AND_THEN("2d") {
        const auto shape = Shape2<i64>{10, 10};
        const auto elements = shape.elements();

        const auto buffer = std::make_unique<i32[]>(elements);
        const auto accessor = AccessorContiguousI64<i32, 2>(buffer.get(), shape.strides());
        iwise(shape, [=](i64 i, i64 j) { accessor(i, j) = static_cast<i32>(i * 10 + j); });

        const auto expected = std::make_unique<i32[]>(elements);
        for (i32 i{0}; auto& e: Span(expected.get(), elements))
            e = i++;

        REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 0));
    }

    AND_THEN("3d") {
        const auto shape = Shape3<i64>{10, 10, 10};
        const auto elements = shape.elements();

        const auto buffer = std::make_unique<i32[]>(elements);
        const auto accessor = AccessorContiguousI64<i32, 3>(buffer.get(), shape.strides());
        iwise(shape, [=, i = int{0}](Vec3<i64> indices) mutable { accessor(indices) = i++; });

        const auto expected = std::make_unique<i32[]>(elements);
        for (i32 i{0}; auto& e: Span(expected.get(), elements))
            e = i++;

        REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 0));
    }

    AND_THEN("4d") {
        const auto shape = Shape4<i64>{10, 10, 10, 10};
        const auto elements = shape.elements();

        const auto buffer = std::make_unique<i32[]>(elements);
        const auto accessor = AccessorContiguousI64<i32, 4>(buffer.get(), shape.strides());
        iwise(shape, [=, i = int{0}](Vec4<i64> indices) mutable { accessor(indices) = i++; });

        const auto expected = std::make_unique<i32[]>(elements);
        for (i32 i{0}; auto& e: Span(expected.get(), elements))
            e = i++;

        REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 0));
    }
}
