#include <noa/core/Reduce.hpp>
#include <noa/unified/Array.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/Iwise.hpp>
#include <noa/unified/Ewise.hpp>
#include <noa/unified/ReduceIwise.hpp>
#include <noa/unified/ReduceEwise.hpp>
#include <noa/unified/ReduceAxesEwise.hpp>
#include <noa/unified/ReduceAxesIwise.hpp>

#include <catch2/catch.hpp>
#include "Helpers.h"

TEST_CASE("unified::iwise", "[noa][unified]") {
    using namespace ::noa::types;

    std::vector<Device> devices{"cpu"};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const auto shape = test::get_random_shape4_batched(4);
    for (auto& device: devices) {
        const auto stream = StreamGuard(device, StreamMode::ASYNC);
        const auto options = ArrayOption{device, "managed"};

        const auto a0 = Array<f32>(shape, options);
        auto arange_op = noa::Arange4d(a0.accessor(), a0.shape());
        noa::iwise(a0.shape(), a0.device(), arange_op, a0);

        const auto a1 = Array<f32>(shape, options);
        test::arange(a1.get(), a1.elements());

        REQUIRE(test::Matcher<f32>(test::MATCH_ABS_SAFE, a0, a1, 1e-6));
    }
}
