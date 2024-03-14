#include <noa/unified/Random.hpp>
#include <noa/unified/Reduce.hpp>

#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa::types;

TEMPLATE_TEST_CASE("unified::randomize()", "[noa][unified]", f32, f64) {
    using value_t = noa::traits::value_type_t<TestType>;
    const auto pad = GENERATE(true, false);
    const auto subregion_shape = test::get_random_shape4_batched(3);
    auto shape = subregion_shape;
    if (pad) {
        shape[1] += 8;
        shape[2] += 9;
        shape[2] += 10;
    }

    std::vector<Device> devices{Device{}};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    auto data = Array<TestType>(shape);
    test::memset(data.get(), data.elements(), 20);

    for (auto& device: devices) {
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, "managed");
        INFO(device);
        data = device.is_cpu() ? data : data.to(options);

        const auto subregion = data.subregion(
                noa::indexing::FullExtent{},
                noa::indexing::Slice{0, subregion_shape[1]},
                noa::indexing::Slice{0, subregion_shape[2]},
                noa::indexing::Slice{0, subregion_shape[3]});

        noa::randomize(noa::Uniform<value_t>{-10, 10}, subregion);
        const auto min = noa::min(subregion);
        const auto max = noa::max(subregion);
        auto mean = noa::mean(subregion);
        REQUIRE(min >= value_t{-10});
        REQUIRE(max <= value_t{10});
        REQUIRE_THAT(mean, Catch::WithinAbs(0, 0.1));

        test::memset(data.get(), data.elements(), 20);

        noa::randomize(noa::Normal<value_t>{5, 2}, subregion);
        mean = noa::mean(subregion);
        const auto stddev = noa::stddev(subregion);
        REQUIRE_THAT(mean, Catch::WithinAbs(5, 0.1));
        REQUIRE_THAT(stddev, Catch::WithinAbs(2, 0.1));
    }
}
