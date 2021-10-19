#include <noa/common/io/ImageFile.h>
#include <noa/common/transform/Geometry.h>
#include <noa/common/transform/Euler.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/math/Reductions.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/transform/Apply.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::transform::apply2D() - symmetry", "[assets][noa][cuda][transform]") {
    path_t path_base = test::PATH_TEST_DATA / "transform";
    YAML::Node param = YAML::LoadFile(path_base / "param.yaml")["apply2D_symmetry"];
    auto input_filename = path_base / param["input"].as<path_t>();

    io::ImageFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        auto current = param["tests"][nb];
        auto filename_expected = path_base / current["expected"].as<path_t>();
        auto shift = current["shift"].as<float2_t>();
        float22_t matrix = transform::rotate(-math::toRad(current["angle"].as<float>())); // inverse
        transform::Symmetry symmetry(current["symmetry"].as<std::string>());
        auto center = current["center"].as<float2_t>();
        auto interp = current["interp"].as<InterpMode>();

        // Get input.
        file.open(input_filename, io::READ);
        size3_t shape = file.shape();
        size_t elements = getElements(shape);
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        cuda::Stream stream;
        cpu::memory::PtrHost<float> output(elements);
        cuda::memory::PtrDevicePadded<float> d_input(shape);

        cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
        cuda::transform::apply2D(d_input.get(), d_input.pitch(),
                                 d_input.get(), d_input.pitch(), {shape.x, shape.y},
                                 shift, matrix, symmetry, center, interp, stream);
        cuda::memory::copy(d_input.get(), d_input.pitch(), output.get(), shape.x, shape, stream);
        stream.synchronize();

        if (interp != INTERP_NEAREST) {
            cpu::math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
            float min, max, mean;
            cpu::math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
            REQUIRE(math::abs(min) < 5e-4f);
            REQUIRE(math::abs(max) < 5e-4f);
            REQUIRE(math::abs(mean) < 1e-6f);
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
        }
    }
}

TEST_CASE("cuda::transform::apply3D() - symmetry", "[assets][noa][cuda][transform]") {
    path_t path_base = test::PATH_TEST_DATA / "transform";
    YAML::Node param = YAML::LoadFile(path_base / "param.yaml")["apply3D_symmetry"];

    io::ImageFile file;
    for (size_t nb = 0; nb < param["tests"].size(); ++nb) {
        INFO("test number = " << nb);

        auto current = param["tests"][nb];
        auto filename_expected = path_base / current["expected"].as<path_t>();
        auto filename_input = path_base / current["input"].as<path_t>();
        auto shift = current["shift"].as<float3_t>();
        float33_t matrix = transform::toMatrix<true>(math::toRad(current["angle"].as<float3_t>()));
        transform::Symmetry symmetry(current["symmetry"].as<std::string>());
        auto center = current["center"].as<float3_t>();
        auto interp = current["interp"].as<InterpMode>();

        // Get input.
        file.open(filename_input, io::READ);
        size3_t shape = file.shape();
        size_t elements = getElements(shape);
        cpu::memory::PtrHost<float> input(elements);
        file.readAll(input.get());

        // Get expected.
        cpu::memory::PtrHost<float> expected(elements);
        file.open(filename_expected, io::READ);
        file.readAll(expected.get());

        cuda::Stream stream;
        cpu::memory::PtrHost<float> output(elements);
        cuda::memory::PtrDevicePadded<float> d_input(shape);

        cuda::memory::copy(input.get(), shape.x, d_input.get(), d_input.pitch(), shape, stream);
        cuda::transform::apply3D(d_input.get(), d_input.pitch(),
                                 d_input.get(), d_input.pitch(), shape,
                                 shift, matrix, symmetry, center, interp, stream);
        cuda::memory::copy(d_input.get(), d_input.pitch(), output.get(), shape.x, shape, stream);
        stream.synchronize();

        if (interp != INTERP_NEAREST) {
            cpu::math::subtractArray(expected.get(), output.get(), output.get(), elements, 1);
            float min, max, mean;
            cpu::math::minMaxSumMean<float>(output.get(), &min, &max, nullptr, &mean, elements, 1);
            REQUIRE(math::abs(min) < 5e-4f);
            REQUIRE(math::abs(max) < 5e-4f);
            REQUIRE(math::abs(mean) < 1e-6f);
        } else {
            float diff = test::getDifference(expected.get(), output.get(), elements);
            REQUIRE_THAT(diff, test::isWithinAbs(0.f, 1e-6));
        }
    }
}
