#include <noa/common/geometry/Euler.h>
#include <noa/unified/fft/Remap.h>
#include <noa/unified/math/Random.h>
#include <noa/unified/signal/fft/Shape.h>
#include <noa/unified/io/ImageFile.h>

#include <catch2/catch.hpp>
#include "Helpers.h"
#include "Assets.h"

using namespace ::noa;

TEST_CASE("unified::signal::fft::ellipse", "[noa][unified]") {
    std::vector<Device> devices{Device("cpu")};
    if (Device::any(Device::GPU))
        devices.emplace_back("gpu");

    const path_t path_base = test::NOA_DATA_PATH / "signal" / "fft";
    YAML::Node tests = YAML::LoadFile(path_base / "tests.yaml")["ellipse"]["tests"];
    io::ImageFile file;

    for (size_t nb = 0; nb < tests.size(); ++nb) {
        INFO("test number = " << nb);

        const auto test = tests[nb];
        const auto filename = path_base / test["filename"].as<path_t>();
        const auto shape = test["shape"].as<dim4_t>();
        const auto center = test["center"].as<float3_t>();
        const auto radius = test["radius"].as<float3_t>();
        const auto edge_size = test["edge_size"].as<float>();
        const auto fwd_transform = geometry::euler2matrix(math::deg2rad(test["angles"].as<float3_t>()), "ZYX");

        constexpr bool COMPUTE_ASSETS = false;
        if constexpr (COMPUTE_ASSETS) {
            Array<float> output(shape);
            signal::fft::ellipse<fft::FC2FC>({}, output, center, radius, edge_size, math::inverse(fwd_transform));
            io::save(output, filename);
            continue;
        }

        const Array expected = io::load<float>(filename);
        const Array expected_remap = fft::remap(fft::FC2F, expected, shape);
        for (auto& device: devices) {
            StreamGuard stream(device);
            ArrayOption options(device, Allocator::MANAGED);

            Array<float> output(shape, options);
            signal::fft::ellipse<fft::FC2FC>({}, output, center, radius, edge_size, math::inverse(fwd_transform));
            REQUIRE(test::Matcher(test::MATCH_ABS, expected, output, 5e-5));

            signal::fft::ellipse<fft::F2F>({}, output, center, radius, edge_size, math::inverse(fwd_transform));
            REQUIRE(test::Matcher(test::MATCH_ABS, expected_remap, output, 5e-5));
        }
    }
}
