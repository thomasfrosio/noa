#include <noa/Array.h>
#include <noa/math/Ewise.h>
#include <noa/math/Random.h>
#include <noa/math/Reduce.h>
#include <noa/memory/Factory.h>
#include <noa/memory/Index.h>

#include <catch2/catch.hpp>

#include "Helpers.h"

using namespace ::noa;

TEMPLATE_TEST_CASE("unified::memory::{extract|insert}, subregions", "[noa][unified]",
                   int32_t, float, double, cfloat_t) {
    std::vector<Device> devices = {Device{"cpu"}};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::MANAGED);

        Array<TestType> data({2, 100, 200, 300}, options);
        Array<TestType> subregions({3, 64, 64, 64}, options);

        math::randomize(math::uniform_t{}, data, -5, 5);
        memory::fill(subregions, TestType{0});

        std::vector<int4_t> origins = {{0, 0, 0, 0},
                                       {0, 34, 130, -20},
                                       {1, 60, 128, 255}};
        Array origins_view{origins.data(), origins.size()};

        memory::extract(data, subregions, origins_view);
        Array result = data.to(options);
        memory::insert(subregions, result, origins_view);

        REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, data, result, 1e-7));
    }
}

TEMPLATE_TEST_CASE("unified::memory::{extract|insert}, sequences", "[noa][unified]",
                   int32_t, float, double) {
    std::vector<Device> devices = {Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    for (auto& device: devices) {
        StreamGuard stream(device);
        ArrayOption options(device, Allocator::DEFAULT_ASYNC);
        Array data = math::random<TestType>(math::uniform_t{}, {2, 100, 200, 300}, -5, 5, options);

        auto[values, indexes] = memory::extract<TestType, uint64_t>(data, data, TestType{0}, math::less_t{});
        math::ewise(values, values, math::abs_t{});
        memory::insert(values, indexes, data);

        TestType min = math::min(data);
        REQUIRE(min >= 0);
        if constexpr (noa::traits::is_float_v<TestType>) {
            TestType mean = math::mean(data);
            REQUIRE_THAT(mean, Catch::WithinAbs(2.5, 0.1));
        }
    }
}
