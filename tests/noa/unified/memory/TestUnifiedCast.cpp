#include <noa/unified/math/Random.hpp>
#include <noa/unified/memory/Cast.hpp>

#include <catch2/catch.hpp>
#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::memory::cast", "[noa][unified]", i32, f32, f64) {
    const bool pad = GENERATE(false, true);
    const auto subregion_shape = test::get_random_shape4_batched(3);
    auto shape = subregion_shape;
    if (pad) {
        shape[1] += 10;
        shape[2] += 11;
        shape[3] += 12;
    }

    std::vector<Device> devices{Device("cpu")};
    if (Device::any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, Allocator::MANAGED);

        auto results = Array<TestType>(shape, options);
        results = results.subregion(
                noa::indexing::ellipsis_t{},
                noa::indexing::slice_t{0, subregion_shape[1]},
                noa::indexing::slice_t{0, subregion_shape[2]},
                noa::indexing::slice_t{0, subregion_shape[3]});

        const Array data0 = math::random<i32>(math::uniform_t{}, results.shape(), -50, 50, options);
        memory::cast(data0, results);

        const Array<i32> data1(results.shape(), options);
        memory::cast(results, data1);

        REQUIRE(test::Matcher(test::MATCH_ABS, data0, data1, 1e-7));
    }
}
