#include <noa/cpu/Ewise.hpp>
#include <catch2/catch.hpp>

#include "Helpers.h"

namespace {
    struct Tracked {
        std::array<int, 2> count{};
        Tracked() = default;
        Tracked(const Tracked& t) : count(t.count) { count[0] += 1; }
        Tracked(Tracked&& t)  noexcept : count(t.count) { count[1] += 1; }
    };
}

TEST_CASE("cpu::ewise") {
    using namespace noa::types;
    using noa::cpu::ewise;
    using noa::cpu::EwiseConfig;

    AND_THEN("check operator") {
        const auto shape = Shape4<i64>{1, 1, 1, 1};
        std::array<int, 2> buffer[2]{};
        Tuple<AccessorValue<Tracked>> input{};
        auto output_contiguous = noa::make_tuple(
                AccessorI64<std::array<int, 2>, 4>(buffer + 0, shape.strides()),
                AccessorI64<std::array<int, 2>, 4>(buffer + 1, shape.strides()));

        struct Op {
            Tracked t1{};
            constexpr void operator()(const Tracked& t0, std::array<int, 2>& o0, std::array<int, 2>& o1) const {
                o0 = t0.count;
                o1 = t1.count;
            }
        };
        auto op0 = Op{};
        auto op1 = Op{};

        // Operator is copied once to the kernel
        ewise(shape, op0, input, output_contiguous);
        REQUIRE((buffer[0][0] == 1 and buffer[0][1] == 1)); // input is copied once and moved once*
        REQUIRE((buffer[1][0] == 1 and buffer[1][1] == 0)); // operator is copied once
        //* in the contiguous case, the input/output are forwarded to a 1d accessor (+1 copy|move),
        //  which is then moved into the kernel

        ewise(shape, std::move(op0), std::move(input), output_contiguous);
        REQUIRE((buffer[0][0] == 0 and buffer[0][1] == 2)); // input is moved twice
        REQUIRE((buffer[1][0] == 0 and buffer[1][1] == 1)); // operator is moved once

        // Create a non-contiguous case by broadcasting.
        auto shape_strided = Shape4<i64>{1, 1, 1, 2};
        auto output_strided = noa::make_tuple(
                AccessorI64<std::array<int, 2>, 4>(buffer + 0, Strides4<i64>{}),
                AccessorI64<std::array<int, 2>, 4>(buffer + 1, Strides4<i64>{}));

        ewise(shape_strided, op1, input, output_strided);
        REQUIRE((buffer[0][0] == 1 and buffer[0][1] == 0)); // input is copied once*
        REQUIRE((buffer[1][0] == 1 and buffer[1][1] == 0)); // operator is copied once
        //* in the non-contiguous case, things are simpler, and the input/output are simply forwarded once

        ewise(shape_strided, std::move(op1), std::move(input), output_strided);
        REQUIRE((buffer[0][0] == 0 and buffer[0][1] == 1)); // input is moved once
        REQUIRE((buffer[1][0] == 0 and buffer[1][1] == 1)); // operator is moved once
    }

    AND_THEN("simply fill and copy") {
        const auto shape = Shape4<i64>{1, 1, 1, 100};
        const auto elements = shape.elements();

        const auto buffer = std::make_unique<f64[]>(elements);
        const auto expected = std::make_unique<f64[]>(elements);
        for (i64 i{0}; auto& e: Span(expected.get(), elements))
            e = static_cast<f64>(i++);

        const auto input = noa::make_tuple(Accessor<f64, 4, i64>(buffer.get(), shape.strides()));
        const auto output = noa::make_tuple(Accessor<f64, 4, i64>(expected.get(), shape.strides()));

        ewise(shape, [i = i64{0}](f64& e) mutable { e = static_cast<f64>(i++); }, input, Tuple<>{});
        REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 1e-8));

        std::fill_n(expected.get(), elements, 0.);
        ewise(shape, [](f64 i, f64& o) mutable { o = i; }, input, output);
        REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 1e-8));
    }

    AND_THEN("no outputs, simple memset") {
        const auto shape = Shape4<i64>{1, 1, 1, 100};
        const auto elements = shape.elements();

        const auto buffer = std::make_unique<f64[]>(elements);
        const auto expected = std::make_unique<f64[]>(elements);

        const auto input = noa::make_tuple(Accessor<f64, 4, i64>(buffer.get(), shape.strides()));

        ewise(shape, [](f64& e) { e = 1.4; }, input, Tuple<>{});
        for (auto& e: Span(expected.get(), elements))
            e = 1.4;
        REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 1e-8));

        ewise<EwiseConfig{.zip_input=true}>(shape, [](Tuple<f64&> e) { e[Tag<0>{}] = 1.5; }, input, Tuple<>{});
        for (auto& e: Span(expected.get(), elements))
            e = 1.5;
        REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 1e-8));
    }

    AND_THEN("fill using runtime value") {
        const auto shape = Shape4<i64>{1, 1, 1, 100};
        const auto elements = shape.elements();

        auto value = AccessorValue<f64>(1.57);

        const auto buffer = std::make_unique<f64[]>(elements);
        const auto expected = std::make_unique<f64[]>(elements);
        for (auto& e: Span(expected.get(), elements))
            e = value.deref();

        auto input = noa::make_tuple(value);
        auto output = noa::make_tuple(Accessor<f64, 4, i64>(buffer.get(), shape.strides()));
        ewise<EwiseConfig{.zip_input=true}>(shape, [](Tuple<f64&> i, f64& o) { o = i[Tag<0>{}]; }, input, output);
        REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 1e-8));

        std::fill_n(buffer.get(), elements, 0.);
        ewise(shape, [](f64 i, f64& o) { o = i; }, input, output);
        REQUIRE(test::Matcher(test::MATCH_ABS, buffer.get(), expected.get(), elements, 1e-8));
    }
}
