#include <noa/common/io/ImageFile.h>
#include <noa/common/geometry/Euler.h>
#include <noa/common/geometry/Transform.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/geometry/fft/Transform.h>

#include <noa/gpu/cuda/math/Ewise.h>
#include <noa/gpu/cuda/fft/Transforms.h>
#include <noa/gpu/cuda/memory/PtrManaged.h>
#include <noa/gpu/cuda/geometry/fft/Transform.h>
#include <noa/gpu/cuda/signal/fft/Shift.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::transform::fft::apply2D()", "[assets][noa][cuda][transform]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node& tests = YAML::LoadFile(path_base / "tests.yaml")["transform2D"]["tests"];
    io::ImageFile file;
    cuda::Stream stream;

    for (size_t i = 0; i < tests.size(); ++i) {
        INFO("test: " << i);

        const YAML::Node& test = tests[i];
        const auto path_input = path_base / test["input"].as<path_t>();
        const auto path_expected = path_base / test["expected"].as<path_t>();
        const auto scale = test["scale"].as<float2_t>();
        const auto rotate = math::toRad(test["rotate"].as<float>());
        const auto center = test["center"].as<float2_t>();
        const auto shift = test["shift"].as<float2_t>();
        const auto cutoff = test["cutoff"].as<float>();
        const auto interp = test["interp"].as<InterpMode>();

        float22_t matrix(geometry::rotate(rotate) *
                         geometry::scale(1 / scale));
        matrix = math::inverse(matrix);

        // Load input:
        file.open(path_input, io::READ);
        const size4_t shape = file.shape();
        const size4_t shape_fft = shape.fft();
        const size4_t stride = shape.stride();
        const size4_t stride_fft = shape_fft.stride();

        cuda::memory::PtrManaged<float> input(shape.elements(), stream);
        file.readAll(input.get(), false);
        file.close();

        // Go to Fourier space:
        cuda::memory::PtrManaged<cfloat_t> input_fft(shape_fft.elements(), stream);
        cuda::fft::r2c(input.share(), input_fft.share(), shape, stream);
        const auto weight = 1.f / static_cast<float>(input.elements());
        cuda::math::ewise(input_fft.share(), stride_fft, math::sqrt(weight), input_fft.share(), stride_fft,
                         shape_fft, noa::math::multiply_t{}, stream);

        // Apply new geometry:
        cuda::memory::PtrManaged<cfloat_t> input_fft_centered(input_fft.elements(), stream);
        cuda::memory::PtrManaged<cfloat_t> output_fft(input_fft.elements(), stream);
        cuda::signal::fft::shift2D<fft::H2HC>(
                input_fft.share(), stride_fft, input_fft_centered.share(), stride_fft, shape, -center, cutoff, stream);
        cuda::geometry::fft::transform2D<fft::HC2H>(
                input_fft_centered.share(), stride_fft, output_fft.share(), stride_fft, shape,
                matrix, center + shift, cutoff, interp, stream);

        // Go back to real space:
        cuda::fft::c2r(output_fft.share(), input.share(), shape, stream);
        cuda::math::ewise(input.share(), stride, math::sqrt(weight), input.share(), stride,
                         shape, noa::math::multiply_t{}, stream);

        // Load excepted and compare
        cuda::memory::PtrManaged<float> expected(input.elements(), stream);
        file.open(path_expected, io::READ);
        file.readAll(expected.get(), false);
        file.close();

        stream.synchronize();
        test::Matcher<float> matcher(test::MATCH_ABS, input.get(), expected.get(), input.elements(), 1e-4);
        REQUIRE(matcher);
    }
}

TEMPLATE_TEST_CASE("cuda::geometry::fft::transform2D(), no remap", "[noa][cuda][geometry]", float) {
    const size4_t shape = test::getRandomShapeBatched(2);
    const size4_t shape_fft = shape.fft();
    const size4_t stride_fft = shape_fft.stride();
    const size_t elements_fft = stride_fft[0] * shape[0];
    const float cutoff = 0.5f;
    const auto interp = INTERP_LINEAR;
    cuda::Stream stream;
    cpu::Stream cpu_stream(cpu::Stream::DEFAULT);

    // Prepare transformation:
    test::Randomizer<float> randomizer(-3, 3);
    cpu::memory::PtrHost<float22_t> matrices(shape[0]);
    cpu::memory::PtrHost<float2_t> shifts(shape[0]);
    for (size_t batch = 0; batch < shape[0]; ++batch) {
        const float2_t scale = {0.9, 1.1};
        const float rotate = randomizer.get();
        const float22_t matrix(geometry::rotate(rotate) *
                               geometry::scale(1 / scale));
        matrices[batch] = math::inverse(matrix);
        shifts[batch] = {randomizer.get() * 10, randomizer.get() * 10};
    }

    cuda::memory::PtrManaged<TestType> input(elements_fft, stream);
    test::randomize(input.get(), input.elements(), randomizer);

    cuda::memory::PtrManaged<TestType> d_output_fft(input.elements(), stream);
    cuda::geometry::fft::transform2D<fft::HC2HC>(
            input.share(), stride_fft, d_output_fft.share(), stride_fft, shape,
            matrices.share(), shifts.share(), cutoff, interp, stream);

    cpu::memory::PtrHost<TestType> h_output_fft(input.elements());
    cpu::geometry::fft::transform2D<fft::HC2HC>(
            input.share(), stride_fft, h_output_fft.share(), stride_fft, shape,
            matrices.share(), shifts.share(), cutoff, interp, cpu_stream);

    stream.synchronize();
    test::Matcher<TestType> matcher(test::MATCH_ABS, h_output_fft.get(), d_output_fft.get(), input.elements(), 5e-4);
    REQUIRE(matcher);
}

TEST_CASE("cuda::transform::fft::apply3D()", "[assets][noa][cuda][transform]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node& tests = YAML::LoadFile(path_base / "tests.yaml")["transform3D"]["tests"];

    io::ImageFile file;
    cuda::Stream stream;
    for (size_t i = 0; i < tests.size(); ++i) {
        INFO("test: " << i);
        const YAML::Node& test = tests[i];
        const auto path_input = path_base / test["input"].as<path_t>();
        const auto path_expected = path_base / test["expected"].as<path_t>();
        const auto scale = test["scale"].as<float3_t>();
        const auto rotate = geometry::euler2matrix(math::toRad(test["rotate"].as<float3_t>()));
        const auto center = test["center"].as<float3_t>();
        const auto shift = test["shift"].as<float3_t>();
        const auto cutoff = test["cutoff"].as<float>();
        const auto interp = test["interp"].as<InterpMode>();

        float33_t matrix(rotate * geometry::scale(1 / scale));
        matrix = math::inverse(matrix);

        // Load input:
        file.open(path_input, io::READ);
        const size4_t shape = file.shape();
        const size4_t shape_fft = shape.fft();
        const size4_t stride = shape.stride();
        const size4_t stride_fft = shape_fft.stride();

        cuda::memory::PtrManaged<float> input(shape.elements(), stream);
        file.readAll(input.get(), false);

        // Go to Fourier space:
        cuda::memory::PtrManaged<cfloat_t> input_fft(shape_fft.elements(), stream);
        cuda::fft::r2c(input.share(), input_fft.share(), shape, stream);
        const auto weight = 1.f / static_cast<float>(input.elements());
        cuda::math::ewise(input_fft.share(), stride_fft, math::sqrt(weight), input_fft.share(), stride_fft,
                          shape_fft, noa::math::multiply_t{}, stream);

        // Apply new geometry:
        cuda::memory::PtrManaged<cfloat_t> input_fft_centered(input_fft.elements(), stream);
        cuda::memory::PtrManaged<cfloat_t> output_fft(input_fft.elements(), stream);
        cuda::signal::fft::shift3D<fft::H2HC>(
                input_fft.share(), stride_fft, input_fft_centered.share(), stride_fft, shape, -center, cutoff, stream);
        cuda::geometry::fft::transform3D<fft::HC2H>(
                input_fft_centered.share(), stride_fft, output_fft.share(), stride_fft, shape,
                matrix, center + shift, cutoff, interp, stream);

        // Go back to real space:
        cuda::fft::c2r(output_fft.share(), input.share(), shape, stream);
        cuda::math::ewise(input.share(), stride, math::sqrt(weight), input.share(), stride,
                          shape, noa::math::multiply_t{}, stream);

        // Load excepted and compare
        cuda::memory::PtrManaged<float> expected(input.elements(), stream);
        file.open(path_expected, io::READ);
        file.readAll(expected.get());

        stream.synchronize();
        test::Matcher<float> matcher(test::MATCH_ABS, input.get(), expected.get(), input.elements(), 2e-4);
        REQUIRE(matcher);
    }
}

TEMPLATE_TEST_CASE("cuda::geometry::fft::transform3D(), no remap", "[noa][cuda][geometry]", float, cfloat_t) {
    // Sometimes due to a precision difference between host and device,
    // some elements at the cutoff are included in the host and when they're excluded in the device, and vice-versa.
    const size4_t shape = {3,54,122,123}; // test::getRandomShapeBatched(3);
    const size4_t shape_fft = shape.fft();
    const size4_t stride_fft = shape_fft.stride();
    const size_t elements_fft = stride_fft[0] * shape[0];
    const float cutoff = 0.5f;
    const auto interp = INTERP_LINEAR;
    cuda::Stream stream;
    cpu::Stream cpu_stream(cpu::Stream::DEFAULT);
    INFO(shape);

    // Prepare transformation:
    test::Randomizer<float> randomizer(-3, 3);
    cpu::memory::PtrHost<float33_t> matrices(shape[0]);
    cpu::memory::PtrHost<float3_t> shifts(shape[0]);
    for (size_t batch = 0; batch < shape[0]; ++batch) {
        const float3_t scale = {1.1, 1, 0.9};
        const float3_t eulers = {1.54, -2.85, -0.53};
        const float33_t matrix{geometry::euler2matrix(eulers) *
                               geometry::scale(1 / scale)};
        matrices[batch] = math::inverse(matrix);
        shifts[batch] = {randomizer.get() * 10, randomizer.get() * 10, randomizer.get() * 10};
    }

    cuda::memory::PtrManaged<TestType> input(elements_fft, stream);
    test::randomize(input.get(), input.elements(), randomizer);

    cuda::memory::PtrManaged<TestType> d_output_fft(input.elements());
    cuda::geometry::fft::transform3D<fft::HC2HC>(
            input.share(), stride_fft, d_output_fft.share(), stride_fft, shape,
            matrices.share(), shifts.share(), cutoff, interp, stream);
    stream.synchronize();

    cpu::memory::PtrHost<TestType> h_output_fft(input.elements());
    cpu::geometry::fft::transform3D<fft::HC2HC>(
            input.share(), stride_fft, h_output_fft.share(), stride_fft, shape,
            matrices.share(), shifts.share(), cutoff, interp, cpu_stream);

    test::Matcher<TestType> matcher(test::MATCH_ABS, h_output_fft.get(), d_output_fft.get(), input.elements(), 5e-4);
    REQUIRE(matcher);
}
