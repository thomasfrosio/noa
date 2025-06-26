#include <noa/unified/Random.hpp>
#include <noa/unified/Reduce.hpp>

#include "Catch.hpp"
#include "Utils.hpp"

using namespace noa::types;

TEMPLATE_TEST_CASE("unified::randomize()", "", f32, f64) {
    using value_t = noa::traits::value_type_t<TestType>;
    const auto pad = GENERATE(true, false);
    const auto subregion_shape = test::random_shape_batched(3);
    auto shape = subregion_shape;
    if (pad) {
        shape[1] += 8;
        shape[2] += 9;
        shape[2] += 10;
    }

    std::vector<Device> devices{"cpu"};
    if (Device::is_any_gpu())
        devices.emplace_back("gpu");

    auto data = Array<TestType>(shape);
    test::fill(data.get(), data.n_elements(), 20);

    for (auto& device: devices) {
        const auto guard = DeviceGuard(device);
        const auto stream = StreamGuard(device);
        const auto options = ArrayOption(device, "managed");
        INFO(device);
        data = device.is_cpu() ? data : data.to(options);

        const auto subregion = data.subregion(
                noa::indexing::Full{},
                noa::indexing::Slice{0, subregion_shape[1]},
                noa::indexing::Slice{0, subregion_shape[2]},
                noa::indexing::Slice{0, subregion_shape[3]});

        noa::randomize(noa::Uniform<value_t>{-10, 10}, subregion);
        const auto min = noa::min(subregion);
        const auto max = noa::max(subregion);
        auto mean = noa::mean(subregion);
        REQUIRE(min >= value_t{-10});
        REQUIRE(max <= value_t{10});
        REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(0, 0.1));

        test::fill(data.get(), data.n_elements(), 20);

        noa::randomize(noa::Normal<value_t>{5, 2}, subregion);
        mean = noa::mean(subregion);
        const auto stddev = noa::stddev(subregion);
        REQUIRE_THAT(mean, Catch::Matchers::WithinAbs(5, 0.1));
        REQUIRE_THAT(stddev, Catch::Matchers::WithinAbs(2, 0.1));
    }
}

TEST_CASE("unified::random_generator(), random_value()") {
    auto distribution = noa::Uniform(-10., 10.);
    auto random_generator = noa::random_generator(distribution);

    for ([[maybe_unused]] auto _: noa::irange(1000)) {
        auto random_value = noa::random_value(distribution);
        auto value = random_generator();
        REQUIRE((random_value >= -10. and random_value <= 10.));
        REQUIRE((value >= -10. and value <= 10.));
    }
}
