#include <noa/xform/PolarTransform.hpp>
#include <noa/xform/Draw.hpp>

#include <noa/Runtime.hpp>
#include <noa/IO.hpp>

#include "Assets.hpp"
#include "Catch.hpp"
#include "Utils.hpp"

using namespace noa::types;

TEST_CASE("xform::cartesian2polar", "[asset]") {
    constexpr bool COMPUTE_ASSETS = false;
    std::vector<Device> devices{"cpu"};
    if (not COMPUTE_ASSETS and Device::is_any_gpu())
        devices.emplace_back("gpu");

    const Path path_base = test::NOA_DATA_PATH / "xform";
    const YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["polar"];

    for (size_t nb{}; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = tests[nb];
        const auto cartesian_shape = test["cartesian_shape"].as<Shape4>();
        const auto polar_shape = test["polar_shape"].as<Shape4>();
        const auto cartesian_center = test["cartesian_center"].as<Vec<f64, 2>>();
        const auto radius_range = test["radius_range"].as<Vec<f64, 2>>();
        const auto angle_range = test["angle_range"].as<Vec<f64, 2>>();
        const auto interpolation_mode = test["interpolation_mode"].as<noa::xform::Interp>();
        const auto input_filename = path_base / test["input"].as<Path>();
        const auto output_filename = path_base / test["output"].as<Path>();

        if constexpr (COMPUTE_ASSETS) {
            // Generate a smooth sphere as input.
            const auto input = noa::empty<f32>(cartesian_shape);
            noa::xform::draw({}, input, noa::xform::Sphere{
                .center=cartesian_center,
                .radius=0.,
                .smoothness=radius_range[1]
            }.draw());
            noa::write_image(input, input_filename);

            const auto output = noa::empty<f32>(polar_shape);
            noa::xform::cartesian2polar(
                input, output, cartesian_center, {
                    .rho_range = noa::Linspace{radius_range[0], radius_range[1], false},
                    .phi_range = noa::Linspace{angle_range[0], angle_range[1], false},
                    .interp = interpolation_mode,
                });
            noa::write_image(output, output_filename);

            continue;
        }

        for (const auto& device: devices) {
            const auto option = ArrayOption(device, Allocator::MANAGED);
            INFO(device);

            const auto input = noa::read_image<f32>(input_filename, {.enforce_2d_stack = false}, option).data;
            if (input.shape() != cartesian_shape)
                FAIL("input shape is not correct");

            const auto output = noa::empty<f32>(polar_shape, option);
            noa::xform::cartesian2polar(
                input, output, cartesian_center, {
                    .rho_range = noa::Linspace{radius_range[0], radius_range[1], false},
                    .phi_range = noa::Linspace{angle_range[0], angle_range[1], false},
                    .interp = interpolation_mode,
                });

            const auto expected = noa::read_image<f32>(output_filename).data;
            REQUIRE(test::allclose_abs_safe(output, expected, 1e-6));
        }
    }
}

// TODO Random polar -> cartesian -> polar (preserve scaling)
