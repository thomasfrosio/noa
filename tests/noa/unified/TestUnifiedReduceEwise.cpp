#include <noa/core/Reduce.hpp>
#include <noa/unified/Array.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/ReduceEwise.hpp>

#include <catch2/catch.hpp>

#include "Helpers.h"

using namespace noa::types;

namespace {
    template<typename A>
    struct SumAdd {
        A accessor;
        constexpr void init(const auto& indices, f64& sum) const {
            sum += static_cast<f64>(accessor(indices));
        }
        constexpr void join(f64 isum, f64& sum) const {
            sum += isum;
        }
        constexpr void final(f64 sum, auto& output) const {
            output += static_cast<std::decay_t<decltype(output)>>(sum);
        }
    };
}

TEMPLATE_TEST_CASE("unified::reduce_ewise - simple", "[noa][unified]", i32, f64) {
    auto shape = Shape4<i64>{5, 35, 64, 81};

    auto input = noa::empty<TestType>(shape);
    auto randomizer = test::Randomizer<TestType>(-50, 100);

    f64 sum{};
    for (auto& value: input.span()) {
        value = randomizer.get();
        sum += static_cast<f64>(value);
    }

    std::vector<Device> devices{"cpu"};
    if (Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        auto stream = StreamGuard(device, StreamMode::DEFAULT);
        const auto options = ArrayOption{device, "managed"};
        INFO(device);

        if (device != input.device())
            input = input.to(options);

        f64 output{};
        noa::reduce_ewise(input, f64{}, output, noa::ReduceSum{});
        REQUIRE_THAT(output, Catch::WithinAbs(sum, 1e-6));
    }
}
