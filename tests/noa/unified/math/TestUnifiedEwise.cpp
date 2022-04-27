#include <noa/unified/Array.h>
#include <noa/unified/math/Ewise.h>
#include <noa/unified/math/Random.h>
#include <noa/unified/math/Reduce.h>
#include <noa/unified/memory/Factory.h>

#include <catch2/catch.hpp>

#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::math::ewise", "[noa][unified]", int32_t, float, double) {
    const size4_t shape = test::getRandomShapeBatched(3);

    std::vector<Device> devices = {Device{"cpu"}};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream{device};
        ArrayOption options{device, Allocator::MANAGED};
        INFO(device);

        Array data0 = math::random<TestType>(math::uniform_t{}, shape, -50, 50, options);
        REQUIRE(math::min(data0) >= -50);
        REQUIRE(math::max(data0) <= 50);
        if constexpr (std::is_floating_point_v<TestType>)
            REQUIRE_THAT(math::mean(data0), Catch::WithinAbs(0, 0.1));

        math::ewise(data0, TestType{2}, data0, math::plus_t{});
        REQUIRE(math::min(data0) >= -48);
        REQUIRE(math::max(data0) <= 52);
        if constexpr (std::is_floating_point_v<TestType>)
            REQUIRE_THAT(math::mean(data0), Catch::WithinAbs(2, 0.1));
    }
}

TEMPLATE_TEST_CASE("unified::math::ewise, broadcast", "[noa][unified]", int32_t, float, double) {
    std::vector<Device> devices = {Device{"cpu"}};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream{device};
        ArrayOption options{device, Allocator::MANAGED};

        Array<TestType> data0 = memory::linspace<TestType>({1, 1, 64, 64}, -50, 50, true, options);
        Array<TestType> data1{size4_t{1, 64, 64, 64}, options};
        math::ewise(data0, TestType{-1}, data1, math::multiply_t{});

        const auto view = data1.eval().view();
        REQUIRE_THAT(view.front(), Catch::WithinAbs(50, 1e-6));
        REQUIRE_THAT(view.back(), Catch::WithinAbs(-50, 1e-6));
    }
}
