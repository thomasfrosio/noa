#include <noa/common/io/ImageFile.h>
#include <noa/common/geometry/Transform.h>
#include <noa/common/geometry/Euler.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/geometry/Transform.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

// TestCUDARotate is checking against the CPU backend for all InterpMode and BorderMode.
// Since rotate() is calling apply(), here tests are only checking against assets.

using namespace ::noa;

TEST_CASE("cuda::geometry::transform2D()", "[assets][noa][cuda][geometry]") {
    const path_t path_base = test::PATH_NOA_DATA / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform2D"];
    const auto input_filename = path_base / param["input"].as<path_t>();

    const auto center = param["center"].as<float2_t>();
    const auto scale = param["scale"].as<float2_t>();
    const auto rotate = math::toRad(param["rotate"].as<float>());
    const auto shift = param["shift"].as<float2_t>();
    float33_t matrix(geometry::translate(center) *
                     geometry::translate(shift) *
                     float33_t(geometry::rotate(rotate)) *
                     float33_t(geometry::scale(scale)) *
                     geometry::translate(-center));
    matrix = math::inverse(matrix);

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
        cuda::geometry::transform2D(d_input.get(), d_input.strides(), shape,
                                    d_input.get(), d_input.strides(), shape,
                                    matrix, interp, border, stream);
        cuda::memory::copy(d_input.get(), d_input.strides(), output.get(), stride, shape, stream);
        stream.synchronize();

        if (interp != INTERP_NEAREST) {
            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), elements, 5e-4f));
        } else {
            const float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
        }
    }
}

TEST_CASE("cuda::geometry::transform3D()", "[assets][noa][cuda][geometry]") {
    const path_t path_base = test::PATH_NOA_DATA / "geometry";
    const YAML::Node param = YAML::LoadFile(path_base / "tests.yaml")["transform3D"];
    const auto input_filename = path_base / param["input"].as<path_t>();

    const auto center = param["center"].as<float3_t>();
    const auto scale = param["scale"].as<float3_t>();
    const auto euler = math::toRad(param["euler"].as<float3_t>());
    const auto shift = param["shift"].as<float3_t>();
    float44_t matrix(geometry::translate(center) *
                     geometry::translate(shift) *
                     float44_t(geometry::euler2matrix(euler)) * // ZYZ intrinsic right-handed
                     float44_t(geometry::scale(scale)) *
                     geometry::translate(-center));
    matrix = math::inverse(matrix);

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
        cuda::geometry::transform3D(d_input.get(), d_input.strides(), shape,
                                    d_input.get(), d_input.strides(), shape,
                                    matrix, interp, border, stream);
        cuda::memory::copy(d_input.get(), d_input.strides(), output.get(), stride, shape, stream);
        stream.synchronize();

        if (interp != INTERP_NEAREST) {
            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), elements, 5e-4f));
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, Catch::WithinAbs(0, 1e-6));
        }
    }
}
