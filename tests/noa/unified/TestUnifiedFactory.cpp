#include <noa/unified/Factory.hpp>
#include <catch2/catch.hpp>

#include "Utils.hpp"

TEST_CASE("unified:: Factory functions", "[noa]") {
    using namespace noa::types;

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    AND_THEN("arange") {
        Array expected = noa::empty<f64>(test::random_shape_batched(3));
        for (i64 i{}; auto& e: expected.span_1d_contiguous())
        e = static_cast<f64>(i++);

        for (auto device: devices) {
            auto a = noa::arange<f64>(expected.shape(), noa::Arange{0., 1.}, {.device=device, .allocator="unified"});
            REQUIRE(test::allclose_abs(a, expected));
        }
    }

    AND_THEN("linspace") {
        Array expected = noa::empty<f64>(test::random_shape_batched(3));
        auto linspace = noa::Linspace{.start=0., .stop=30., .endpoint=true}.for_size(expected.n_elements());
        for (i64 i{}; auto& e: expected.span_1d_contiguous())
            e = linspace(i++);

        for (auto device: devices) {
            auto a = noa::linspace(expected.shape(), noa::Linspace{0., 30.}, {.device=device, .allocator="unified"});
            REQUIRE(test::allclose_abs(a, expected));
        }
    }

    AND_THEN("iota") {
        Array expected = noa::empty<f64>({2, 20, 20, 20});
        auto tile = Vec{1, 5, 5, 5};

        Span span = expected.span_contiguous();
        for (i64 i{}; i < span.shape()[0]; ++i)
            for (i64 j{}; j < span.shape()[1]; ++j)
                for (i64 k{}; k < span.shape()[2]; ++k)
                    for (i64 l{}; l < span.shape()[3]; ++l)
                        span(i, j, k, l) = static_cast<f64>(
                                noa::indexing::offset_at(expected.strides(), Vec{i, j, k, l} % tile.as<i64>()));

        for (auto device: devices) {
            auto a = noa::iota<f64>(expected.shape(), tile, {.device=device, .allocator="unified"});
            REQUIRE(test::allclose_abs(a, expected));
        }
    }
}
