#include <noa/unified/geometry/PolarTransform.hpp>
#include <noa/unified/geometry/DrawShape.hpp>
#include <noa/unified/Factory.hpp>
#include <noa/unified/IO.hpp>

#include <catch2/catch.hpp>

#include "Utils.hpp"
#include "Assets.h"

using namespace noa::types;

TEST_CASE("unified::geometry::cartesian2polar", "[noa][unified]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices{"cpu"};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["polar"];

    for (size_t nb{}; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto cartesian_shape = test["cartesian_shape"].as<Shape4<i64>>();
        const auto polar_shape = test["polar_shape"].as<Shape4<i64>>();
        const auto cartesian_center = test["cartesian_center"].as<Vec2<f64>>();
        const auto radius_range = test["radius_range"].as<Vec2<f64>>();
        const auto angle_range = test["angle_range"].as<Vec2<f64>>();
        const auto interpolation_mode = test["interpolation_mode"].as<noa::Interp>();
        const auto input_filename = path_base / test["input"].as<Path>();
        const auto output_filename = path_base / test["output"].as<Path>();

        if constexpr (COMPUTE_ASSETS) {
            // Generate a smooth sphere as input.
            const auto input = noa::empty<f32>(cartesian_shape);
            noa::geometry::draw_shape({}, input, noa::geometry::Sphere{
                .center=cartesian_center,
                .radius=0.,
                .smoothness=radius_range[1]
            });
            noa::io::write(input, input_filename);

            const auto output = noa::empty<f32>(polar_shape);
            noa::geometry::cartesian2polar(
                input, output, cartesian_center, {
                    .rho_range = radius_range,
                    .rho_endpoint = false,
                    .phi_range = angle_range,
                    .phi_endpoint = false,
                    .interp = interpolation_mode,
                });
            noa::io::write(output, output_filename);

            continue;
        }

        for (const auto& device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            const auto input = noa::io::read_data<f32>(input_filename, {.enforce_2d_stack = false}, option);
            if (noa::any(input.shape() != cartesian_shape))
                FAIL("input shape is not correct");

            const auto output = noa::empty<f32>(polar_shape, option);
            noa::geometry::cartesian2polar(
                input, output, cartesian_center, {
                    .rho_range = radius_range,
                    .rho_endpoint = false,
                    .phi_range = angle_range,
                    .phi_endpoint = false,
                    .interp = interpolation_mode,
                });

            const auto expected = noa::io::read_data<f32>(output_filename);
            REQUIRE(test::allclose_abs_safe(output, expected, 1e-6));
        }
    }
}

// TODO Random polar -> cartesian -> polar (preserve scaling)
