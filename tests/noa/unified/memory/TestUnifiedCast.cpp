#include <noa/unified/math/Random.h>
#include <noa/unified/memory/Cast.h>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::memory::cast", "[noa][unified]", int32_t, float, double) {
    const size4_t shape = test::getRandomShapeBatched(3);

    std::vector<Device> devices = {Device{"cpu"}};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream{device};
        ArrayOption options{device, Allocator::MANAGED};

        Array data0 = math::random<int32_t>(math::Uniform{}, shape, -50, 50, options);

        Array<TestType> result{shape, options};
        memory::cast(data0, result);

        Array<int32_t> data1{shape, options};
        memory::cast(result, data1);

        data1.eval();
        REQUIRE(test::Matcher(test::MATCH_ABS, data0.get(), data1.get(), shape.elements(), 1e-7));
    }
}
