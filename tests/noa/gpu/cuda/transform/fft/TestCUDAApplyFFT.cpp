#include <noa/common/io/ImageFile.h>
#include <noa/common/transform/Euler.h>
#include <noa/common/transform/Geometry.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/transform/fft/Apply.h>
#include <noa/cpu/fft/Transforms.h>

#include <noa/gpu/cuda/memory/PtrDevice.h>
#include <noa/gpu/cuda/memory/PtrDevicePadded.h>
#include <noa/gpu/cuda/memory/Copy.h>
#include <noa/gpu/cuda/transform/fft/Apply.h>
#include <noa/gpu/cuda/transform/fft/Shift.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cuda::transform::fft::apply2D()", "[assets][noa][cuda][transform]") {
    path_t path_base = test::PATH_NOA_DATA / "transform" / "fft";
    const YAML::Node& tests = YAML::LoadFile(path_base / "tests.yaml")["apply2D"]["tests"];
    io::ImageFile file;
    cpu::Stream cpu_stream;

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

        float22_t matrix(transform::rotate(rotate) *
                         transform::scale(1 / scale));
        matrix = math::inverse(matrix);

        // Load input:
        file.open(path_input, io::READ);
        const size3_t shape = file.shape();
        size2_t shape_2d = {shape.x, shape.y};
        size_t half = shape.x / 2 + 1;
        cpu::memory::PtrHost<float> input(elements(shape));
        file.readAll(input.get(), false);
        file.close();

        // Go to Fourier space:
        cpu::memory::PtrHost<cfloat_t> input_fft(elementsFFT(shape));
        cpu::fft::r2c(input.get(), input_fft.get(), shape, 1, cpu_stream);
        const auto weight = 1.f / static_cast<float>(input.elements());
        cpu::math::multiplyByValue(input_fft.get(), math::sqrt(weight), input_fft.get(), input_fft.elements());

        // Copy FFT to GPU:
        cuda::Stream stream(cuda::Stream::CONCURRENT);
        cuda::memory::PtrDevice<cfloat_t> d_input_fft(input_fft.elements());
        cuda::memory::copy(input_fft.get(), d_input_fft.get(), input_fft.elements(), stream);

        // Phase shift:
        cuda::memory::PtrDevicePadded<cfloat_t> d_input_fft_centered(shapeFFT(shape));
        cuda::transform::fft::shift2D<fft::H2HC>(
                d_input_fft.get(), half,
                d_input_fft_centered.get(), d_input_fft_centered.pitch(), shape_2d,
                -center, 1, stream);

        // Transform the FFT:
        cuda::memory::PtrDevicePadded<cfloat_t> d_output_fft(d_input_fft_centered.shape());
        cuda::transform::fft::apply2D<fft::HC2H>(
                d_input_fft_centered.get(), d_input_fft_centered.pitch(),
                d_output_fft.get(), d_output_fft.pitch(), shape_2d,
                matrix, center + shift, cutoff, interp, stream);

        // Copy result back to CPU:
        cpu::memory::PtrHost<cfloat_t> h_output_fft(input_fft.elements());
        cuda::memory::copy(d_output_fft.get(), d_output_fft.pitch(),
                           h_output_fft.get(), half, d_output_fft.shape(), stream);
        stream.synchronize();

        // Go back to real space:
        cpu::fft::c2r(h_output_fft.get(), input.get(), shape, 1, cpu_stream);
        cpu::math::multiplyByValue(input.get(), math::sqrt(weight), input.get(), input.elements());

        // Load excepted and compare
        cpu::memory::PtrHost<float> expected(input.elements());
        file.open(path_expected, io::READ);
        file.readAll(expected.get(), false);
        file.close();
        test::Matcher<float> matcher(test::MATCH_ABS, input.get(), expected.get(), input.elements(), 1e-6);
        REQUIRE(matcher);
    }
}

TEMPLATE_TEST_CASE("cuda::transform::fft::apply2D(), no remap", "[noa][cuda][transform]", float, cfloat_t) {
    const size_t batches = 3;
    const size3_t shape = {255, 256, 1};
    const float cutoff = 0.5f;
    const auto interp = INTERP_LINEAR;

    // Prepare transformation:
    test::Randomizer<float> randomizer(-3, 3);
    cpu::memory::PtrHost<float22_t> transforms(batches);
    cpu::memory::PtrHost<float2_t> shifts(batches);
    for (size_t batch = 0; batch < batches; ++batch) {
        const float2_t scale = {0.9, 1.1};
        const float rotate = randomizer.get();
        float22_t matrix(transform::rotate(rotate) *
                         transform::scale(1 / scale));
        transforms[batch] = math::inverse(matrix);
        shifts[batch] = {randomizer.get() * 10, randomizer.get() * 10};
    }
    cuda::Stream stream;
    cpu::Stream cpu_stream;
    cuda::memory::PtrDevice<float22_t> d_transforms(batches);
    cuda::memory::PtrDevice<float2_t> d_shifts(batches);
    cuda::memory::copy(transforms.get(), d_transforms.get(), transforms.elements(), stream);
    cuda::memory::copy(shifts.get(), d_shifts.get(), shifts.elements(), stream);

    // Randomize input and copy to GPU:
    cpu::memory::PtrHost<TestType> input(elementsFFT(shape) * batches);
    test::randomize(input.get(), input.elements(), randomizer);
    cuda::memory::PtrDevice<TestType> d_input(input.elements());
    cuda::memory::copy(input.get(), d_input.get(), input.elements(), stream);

    // Transform and remap to non-centered.
    const size2_t shape_2d = {shape.x, shape.y};
    const size2_t pitch_2d = shapeFFT(shape_2d);
    size_t half = shape.x / 2 + 1;
    cuda::memory::PtrDevice<TestType> d_output_fft(input.elements());
    cuda::transform::fft::apply2D<fft::HC2HC>(
            d_input.get(), half, d_output_fft.get(), half, shape_2d,
            d_transforms.get(), d_shifts.get(), batches, cutoff, interp, stream);

    // Do the same on the CPU:
    cpu::memory::PtrHost<TestType> output_fft(input.elements());
    cpu::transform::fft::apply2D<fft::HC2HC>(
            input.get(), {pitch_2d.x, 0}, output_fft.get(), pitch_2d, shape_2d,
            transforms.get(), shifts.get(), batches, cutoff, interp, cpu_stream);

    // Copy results to CPU:
    cpu::memory::PtrHost<TestType> output_fft_cuda(input.elements());
    cuda::memory::copy(d_output_fft.get(), output_fft_cuda.get(), d_input.elements(), stream);
    stream.synchronize();

    test::Matcher<TestType> matcher(test::MATCH_ABS, output_fft.get(), output_fft_cuda.get(), input.elements(), 5e-4);
    REQUIRE(matcher);
}

TEST_CASE("cuda::transform::fft::apply3D()", "[assets][noa][cuda][transform]") {
    path_t path_base = test::PATH_NOA_DATA / "transform" / "fft";
    const YAML::Node& tests = YAML::LoadFile(path_base / "tests.yaml")["apply3D"]["tests"];
    io::ImageFile file;
    cpu::Stream cpu_stream;

    for (size_t i = 0; i < tests.size(); ++i) {
        INFO("test: " << i);

        const YAML::Node& test = tests[i];
        const auto path_input = path_base / test["input"].as<path_t>();
        const auto path_expected = path_base / test["expected"].as<path_t>();
        const auto scale = test["scale"].as<float3_t>();
        const auto rotate = transform::toMatrix(math::toRad(test["rotate"].as<float3_t>()));
        const auto center = test["center"].as<float3_t>();
        const auto shift = test["shift"].as<float3_t>();
        const auto cutoff = test["cutoff"].as<float>();
        const auto interp = test["interp"].as<InterpMode>();

        float33_t matrix(rotate * transform::scale(1 / scale));
        matrix = math::inverse(matrix);

        // Load input:
        file.open(path_input, io::READ);
        const size3_t shape = file.shape();
        size_t half = shape.x / 2 + 1;
        cpu::memory::PtrHost<float> input(elements(shape));
        file.readAll(input.get(), false);
        file.close();

        // Go to Fourier space:
        cpu::memory::PtrHost<cfloat_t> input_fft(elementsFFT(shape));
        cpu::fft::r2c(input.get(), input_fft.get(), shape, 1, cpu_stream);
        const auto weight = 1.f / static_cast<float>(input.elements());
        cpu::math::multiplyByValue(input_fft.get(), math::sqrt(weight), input_fft.get(), input_fft.elements());

        // Copy FFT to GPU:
        cuda::Stream stream(cuda::Stream::CONCURRENT);
        cuda::memory::PtrDevice<cfloat_t> d_input_fft(input_fft.elements());
        cuda::memory::copy(input_fft.get(), d_input_fft.get(), input_fft.elements(), stream);

        // Phase shift:
        cuda::memory::PtrDevicePadded<cfloat_t> d_input_fft_centered(shapeFFT(shape));
        cuda::transform::fft::shift3D<fft::H2HC>(
                d_input_fft.get(), half,
                d_input_fft_centered.get(), d_input_fft_centered.pitch(), shape,
                -center, 1, stream);

        // Transform the FFT:
        cuda::memory::PtrDevicePadded<cfloat_t> d_output_fft(d_input_fft_centered.shape());
        cuda::transform::fft::apply3D<fft::HC2H>(
                d_input_fft_centered.get(), d_input_fft_centered.pitch(),
                d_output_fft.get(), d_output_fft.pitch(), shape,
                matrix, center + shift, cutoff, interp, stream);

        // Copy result back to CPU:
        cpu::memory::PtrHost<cfloat_t> h_output_fft(input_fft.elements());
        cuda::memory::copy(d_output_fft.get(), d_output_fft.pitch(),
                           h_output_fft.get(), half, d_output_fft.shape(), stream);
        stream.synchronize();

        // Go back to real space:
        cpu::fft::c2r(h_output_fft.get(), input.get(), shape, 1, cpu_stream);
        cpu::math::multiplyByValue(input.get(), math::sqrt(weight), input.get(), input.elements());

        // Load excepted and compare
        cpu::memory::PtrHost<float> expected(input.elements());
        file.open(path_expected, io::READ);
        file.readAll(expected.get(), false);
        file.close();
        test::Matcher<float> matcher(test::MATCH_ABS, input.get(), expected.get(), input.elements(), 5e-6);
        REQUIRE(matcher);
    }
}

TEMPLATE_TEST_CASE("cuda::transform::fft::apply3D(), no remap", "[noa][cuda][transform]", float, cfloat_t) {
    const size_t batches = 2;
    const size3_t shape = {255, 256, 255};
    const size3_t pitch = shapeFFT(shape);
    const float cutoff = 0.5f;
    const auto interp = INTERP_LINEAR;

    // Prepare transformation:
    test::Randomizer<float> randomizer(-3, 3);
    cpu::memory::PtrHost<float33_t> transforms(batches);
    cpu::memory::PtrHost<float3_t> shifts(batches);
    for (size_t batch = 0; batch < batches; ++batch) {
        const float3_t scale = {0.9, 1.1, 0.99};
        const float3_t eulers = {1.54, -2.85, -0.53};
        float33_t matrix(transform::toMatrix(eulers) *
                         transform::scale(1 / scale));
        transforms[batch] = math::inverse(matrix);
        shifts[batch] = {randomizer.get() * 10, randomizer.get() * 10, randomizer.get() * 10};
    }
    cuda::Stream stream;
    cpu::Stream cpu_stream;
    cuda::memory::PtrDevice<float33_t> d_transforms(batches);
    cuda::memory::PtrDevice<float3_t> d_shifts(batches);
    cuda::memory::copy(transforms.get(), d_transforms.get(), transforms.elements(), stream);
    cuda::memory::copy(shifts.get(), d_shifts.get(), shifts.elements(), stream);

    // Randomize input and copy to GPU:
    cpu::memory::PtrHost<TestType> input(elementsFFT(shape) * batches);
    test::randomize(input.get(), input.elements(), randomizer);
    cuda::memory::PtrDevice<TestType> d_input(input.elements());
    cuda::memory::copy(input.get(), d_input.get(), input.elements(), stream);

    // Transform and remap to non-centered.
    cuda::memory::PtrDevice<TestType> d_output_fft(input.elements());
    cuda::transform::fft::apply3D<fft::HC2HC>(
            d_input.get(), pitch.x, d_output_fft.get(), pitch.x, shape,
            d_transforms.get(), d_shifts.get(), batches, cutoff, interp, stream);

    // Do the same on the CPU:
    cpu::memory::PtrHost<TestType> output_fft(input.elements());
    cpu::transform::fft::apply3D<fft::HC2HC>(
            input.get(), {pitch.x, pitch.y, 0}, output_fft.get(), pitch, shape,
            transforms.get(), shifts.get(), batches, cutoff, interp, cpu_stream);

    // Copy results to CPU:
    cpu::memory::PtrHost<TestType> output_fft_cuda(input.elements());
    cuda::memory::copy(d_output_fft.get(), output_fft_cuda.get(), d_input.elements(), stream);
    stream.synchronize();

    test::Matcher<TestType> matcher(test::MATCH_ABS, output_fft.get(), output_fft_cuda.get(), input.elements(), 5e-4);
    REQUIRE(matcher);
}
