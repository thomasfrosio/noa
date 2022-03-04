#include <noa/common/io/ImageFile.h>
#include <noa/cpu/memory/PtrHost.h>

#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/transform/Scale.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::geometry::scale2D()", "[assets][noa][cuda][geometry]") {
    const path_t path_base = test::PATH_NOA_DATA / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["scale2D"];
    const auto input_filename = path_base / param["input"].as<path_t>();
    const auto scale = param["scale"].as<float2_t>();
    const auto center = param["center"].as<float2_t>();

    io::ImageFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto expected_filename = path_base / test["expected"].as<path_t>();
        const auto interp = test["interp"].as<InterpMode>();
        const auto border = test["border"].as<BorderMode>();

        // Some BorderMode, or BorderMode-InterpMode combination, are not supported on the CUDA implementations.
        if (border == BORDER_VALUE || border == BORDER_REFLECT)
            continue;
        else if (border == BORDER_MIRROR || border == BORDER_PERIODIC)
            if (interp != INTERP_LINEAR_FAST && interp != INTERP_NEAREST)
                continue;

        // Get input.
        file.open(input_filename, io::READ);
        const size4_t shape = file.shape();
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(expected_filename, io::READ);
        file.readAll(expected.get());

        cuda::Stream stream;
        cpu::memory::PtrHost<float> output(elements);
        cuda::memory::PtrDevicePadded<float> d_input(shape);
        cuda::memory::copy(input.get(), stride, d_input.get(), d_input.strides(), shape, stream);
        cuda::geometry::scale2D(d_input.get(), d_input.strides(), shape,
                                d_input.get(), d_input.strides(), shape,
                                 scale, center, interp, border, stream);
        cuda::memory::copy(d_input.get(), d_input.strides(), output.get(), stride, shape, stream);
        stream.synchronize();

        if (interp != INTERP_NEAREST) {
            REQUIRE(test::Matcher(test::MATCH_ABS, output.get(), expected.get(), elements, 1e-4f));
        } else {
            const float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
        }
    }
}

TEST_CASE("cuda::geometry::scale3D()", "[assets][noa][cuda][geometry]") {
    const path_t path_base = test::PATH_NOA_DATA / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["scale3D"];
    const auto input_filename = path_base / param["input"].as<path_t>();
    const auto scale = param["scale"].as<float3_t>();
    const auto center = param["center"].as<float3_t>();

    io::ImageFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        const YAML::Node& test = param["tests"][nb];
        const auto expected_filename = path_base / test["expected"].as<path_t>();
        const auto interp = test["interp"].as<InterpMode>();
        const auto border = test["border"].as<BorderMode>();

        if (border == BORDER_VALUE || border == BORDER_REFLECT)
            continue;
        else if (border == BORDER_MIRROR || border == BORDER_PERIODIC)
            if (interp != INTERP_LINEAR_FAST && interp != INTERP_NEAREST)
                continue;

        // Get input.
        file.open(input_filename, io::READ);
        const size4_t shape = file.shape();
        const size4_t stride = shape.strides();
        const size_t elements = shape.elements();
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(expected_filename, io::READ);
        file.readAll(expected.get());

        cuda::Stream stream;
        cpu::memory::PtrHost<float> output(elements);
        cuda::memory::PtrDevicePadded<float> d_input(shape);
        cuda::memory::copy(input.get(), stride, d_input.get(), d_input.strides(), shape, stream);
        cuda::geometry::scale3D(d_input.get(), d_input.strides(), shape,
                                d_input.get(), d_input.strides(), shape,
                                scale, center, interp, border, stream);
        cuda::memory::copy(d_input.get(), d_input.strides(), output.get(), stride, shape, stream);
        stream.synchronize();

        if (interp != INTERP_NEAREST) {
            REQUIRE(test::Matcher(test::MATCH_ABS, output.get(), expected.get(), elements, 1e-4f));
        } else {
            const float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
        }
    }
}
