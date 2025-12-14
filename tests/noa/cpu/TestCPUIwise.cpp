#include <noa/cpu/Iwise.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

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
        const auto shape = Shape<i64, 1>{2};
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
        const auto shape = Shape<i64, 1>{100};
        const auto elements = shape.n_elements();

        const auto buffer = std::make_unique<i32[]>(static_cast<size_t>(elements));
        iwise(shape, [ptr = buffer.get()](i64 i) { ptr[i] = static_cast<i32>(i); });

        const auto expected = std::make_unique<i32[]>(static_cast<size_t>(elements));
        for (i32 i{}; auto& e: Span(expected.get(), elements))
            e = i++;

        REQUIRE(test::allclose_abs(buffer.get(), expected.get(), elements, 1e-6));
    }

    AND_THEN("2d") {
        const auto shape = Shape<i64, 2>{10, 10};
        const auto elements = shape.n_elements();

        const auto buffer = std::make_unique<i32[]>(static_cast<size_t>(elements));
        const auto accessor = AccessorContiguous<i32, 2, i64>(buffer.get(), shape.strides());
        iwise(shape, [=](i64 i, i64 j) { accessor(i, j) = static_cast<i32>(i * 10 + j); });

        const auto expected = std::make_unique<i32[]>(static_cast<size_t>(elements));
        for (i32 i{}; auto& e: Span(expected.get(), elements))
            e = i++;

        REQUIRE(test::allclose_abs(buffer.get(), expected.get(), elements, 0));
    }

    AND_THEN("3d") {
        const auto shape = Shape<i64, 3>{10, 10, 10};
        const auto elements = shape.n_elements();

        const auto buffer = std::make_unique<i32[]>(static_cast<size_t>(elements));
        const auto accessor = AccessorContiguous<i32, 3, i64>(buffer.get(), shape.strides());
        iwise(shape, [=, i = int{}](Vec<i64, 3> indices) mutable { accessor(indices) = i++; });

        const auto expected = std::make_unique<i32[]>(static_cast<size_t>(elements));
        for (i32 i{}; auto& e: Span(expected.get(), elements))
            e = i++;

        REQUIRE(test::allclose_abs(buffer.get(), expected.get(), elements, 0));
    }

    AND_THEN("4d") {
        const auto shape = Shape<i64, 4>{10, 10, 10, 10};
        const auto elements = shape.n_elements();

        const auto buffer = std::make_unique<i32[]>(static_cast<size_t>(elements));
        const auto accessor = AccessorContiguous<i32, 4, i64>(buffer.get(), shape.strides());
        iwise(shape, [=, i = int{}](Vec<i64, 4> indices) mutable { accessor(indices) = i++; });

        const auto expected = std::make_unique<i32[]>(static_cast<size_t>(elements));
        for (i32 i{}; auto& e: Span(expected.get(), elements))
            e = i++;

        REQUIRE(test::allclose_abs(buffer.get(), expected.get(), elements, 0));
    }
}
