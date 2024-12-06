#include <noa/unified/Random.hpp>
#include <noa/unified/Factory.hpp>
#include <catch2/catch.hpp>
#include "Utils.hpp"

using namespace ::noa::types;

TEMPLATE_TEST_CASE("unified::cast", "[noa][unified]", i32, f32, f64) {
    const bool pad = GENERATE(false, true);
    const auto subregion_shape = test::random_shape_batched(3);
    auto shape = subregion_shape;
    if (pad) {
        shape[1] += 10;
        shape[2] += 11;
        shape[3] += 12;
    }

    std::vector<Device> devices{"cpu"};
    if (Device::is_any(Device::GPU))
        devices.emplace_back("gpu");

    for (const auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, "unified");

        using namespace noa::indexing;
        auto results = Array<TestType>(shape, options);
        results = results.subregion(
            Ellipsis{},
            Slice{0, subregion_shape[1]},
            Slice{0, subregion_shape[2]},
            Slice{0, subregion_shape[3]});

        const Array data0 = noa::random(noa::Uniform(-50, 50), results.shape(), options);
        noa::cast(data0, results);

        const auto data1 = noa::like<i32>(results);
        noa::cast(results, data1);
        REQUIRE(test::allclose_abs(data0, data1));
    }
}
