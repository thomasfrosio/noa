#include <noa/unified/Array.hpp>
#include <noa/unified/Iwise.hpp>
#include <catch2/catch.hpp>

#include "Utils.hpp"

namespace {
    using namespace ::noa::types;

    struct Arange4d {
        Span<f32, 4> span;

        constexpr void operator()(const Vec<i64, 4>& indices) {
            span(indices) = static_cast<f32>(noa::indexing::offset_at(span.strides(), indices));
        }
    };
}

TEST_CASE("unified::iwise", "[noa][unified]") {
    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    const auto shape = test::random_shape_batched(3);
    for (auto& device: devices) {
        const auto stream = StreamGuard(device, Stream::ASYNC);
        const auto options = ArrayOption{device, "managed"};

        const auto a0 = Array<f32>(shape, options);
        noa::iwise(a0.shape(), a0.device(), Arange4d{a0.span()}, a0);

        const auto a1 = Array<f32>(shape, options);
        test::arange(a1.get(), a1.n_elements());

        REQUIRE(test::allclose_abs(a0, a1, 1e-6));
    }
}
