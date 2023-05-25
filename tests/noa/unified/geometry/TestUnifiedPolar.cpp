#include <noa/unified/geometry/Polar.hpp>
#include <noa/unified/geometry/Shape.hpp>
#include <noa/unified/memory/Factory.hpp>
#include <noa/unified/io/ImageFile.hpp>

#include <catch2/catch.hpp>

#include "Helpers.h"
#include "Assets.h"

using namespace noa;

TEST_CASE("unified::geometry::cartesian2polar", "[noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices{Device("cpu")};
    if (!COMPUTE_ASSETS && Device::is_any(DeviceType::GPU))
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["polar"];

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto cartesian_shape = test["cartesian_shape"].as<Shape4<i64>>();
        const auto polar_shape = test["polar_shape"].as<Shape4<i64>>();
        const auto cartesian_center = test["cartesian_center"].as<Vec2<f32>>();
        const auto radius_range = test["radius_range"].as<Vec2<f32>>();
        const auto angle_range = test["angle_range"].as<Vec2<f32>>();
        const auto interpolation_mode = test["interpolation_mode"].as<InterpMode>();
        const auto input_filename = path_base / test["input"].as<Path>();
        const auto output_filename = path_base / test["output"].as<Path>();

        if constexpr (COMPUTE_ASSETS) {
            // Generate a smooth sphere as input.
            const auto input = noa::memory::empty<f32>(cartesian_shape);
            noa::geometry::sphere({}, input, cartesian_center, 0, radius_range[1]);
            noa::io::save(input, input_filename);

            const auto output = noa::memory::empty<f32>(polar_shape);
            noa::geometry::cartesian2polar(
                    input, output, cartesian_center,
                    radius_range, false,
                    angle_range, false,
                    interpolation_mode);
            noa::io::save(output, output_filename);

            continue;
        }

        for (const auto& device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::load_data<f32>(input_filename, false, option);
            if (noa::any(input.shape() != cartesian_shape))
                FAIL("input shape is not correct");

            const auto output = noa::memory::empty<f32>(polar_shape, option);
            noa::geometry::cartesian2polar(
                    input, output, cartesian_center,
                    radius_range, false,
                    angle_range, false,
                    interpolation_mode);

            const auto expected = noa::io::load_data<f32>(output_filename, false);
            REQUIRE(test::Matcher(test::MATCH_ABS_SAFE, output, expected, 1e-6));
        }
    }
}

// TODO Random polar -> cartesian -> polar (preserve scaling)
