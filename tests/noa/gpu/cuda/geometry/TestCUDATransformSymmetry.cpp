#include <noa/common/io/MRCFile.h>
#include <noa/common/geometry/Transform.h>
#include <noa/common/geometry/Euler.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/geometry/Transform.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::geometry::transform2D() - symmetry", "[assets][noa][cuda][geometry]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform2D_symmetry"];
    const auto input_path = path_base / param["input"].as<path_t>();

    io::MRCFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        // Parameters:
        const auto current = param["tests"][nb];
        const auto filename_expected = path_base / current["expected"].as<path_t>();
        const auto shift = current["shift"].as<float2_t>();
        const float22_t matrix = geometry::rotate(-math::toRad(current["angle"].as<float>())); // inverse
        const geometry::Symmetry symmetry(current["symmetry"].as<std::string>());
        const auto center = current["center"].as<float2_t>();
        const auto interp = current["interp"].as<InterpMode>();

        // Prepare data:
        file.open(input_path, io::READ);
        const size4_t shape = file.shape();
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        cuda::Stream stream;
        cpu::memory::PtrHost<float> output(elements);
        cuda::memory::PtrDevicePadded<float> d_input(shape);

        cuda::memory::copy(input.share(), stride, d_input.share(), d_input.strides(), shape, stream);
        cuda::geometry::transform2D(d_input.share(), d_input.strides(), shape,
                                    d_input.share(), d_input.strides(), shape,
                                    shift, matrix, symmetry, center, interp, true, true, stream);
        cuda::memory::copy(d_input.share(), d_input.strides(), output.share(), stride, shape, stream);
        stream.synchronize();

        if (interp != INTERP_NEAREST) {
            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), elements, 5e-4f));
        } else {
            const float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
        }
    }
}

TEST_CASE("cuda::geometry::transform3D() - symmetry", "[assets][noa][cuda][geometry]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform3D_symmetry"];

    io::MRCFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        // Parameters:
        const auto current = param["tests"][nb];
        const auto filename_expected = path_base / current["expected"].as<path_t>();
        const auto filename_input = path_base / current["input"].as<path_t>();
        const auto shift = current["shift"].as<float3_t>();
        const auto euler = math::toRad(current["angle"].as<float3_t>());
        float33_t matrix = geometry::euler2matrix(euler, "ZYZ").transpose();
        const geometry::Symmetry symmetry(current["symmetry"].as<std::string>());
        const auto center = current["center"].as<float3_t>();
        const auto interp = current["interp"].as<InterpMode>();

        // Prepare data:
        file.open(filename_input, io::READ);
        const size4_t shape = file.shape();
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        cuda::Stream stream;
        cpu::memory::PtrHost<float> output(elements);
        cuda::memory::PtrDevicePadded<float> d_input(shape);

        cuda::memory::copy(input.share(), stride, d_input.share(), d_input.strides(), shape, stream);
        cuda::geometry::transform3D(d_input.share(), d_input.strides(), shape,
                                    d_input.share(), d_input.strides(), shape,
                                    shift, matrix, symmetry, center, interp, true, true, stream);
        cuda::memory::copy(d_input.share(), d_input.strides(), output.share(), stride, shape, stream);
        stream.synchronize();

        if (interp != INTERP_NEAREST) {
            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), elements, 5e-4f));
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
        }
    }
}
