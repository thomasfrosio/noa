#include <noa/common/io/ImageFile.h>
#include <noa/common/transform/Euler.h>
#include <noa/common/transform/Geometry.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/transform/fft/Apply.h>
#include <noa/cpu/transform/fft/Shift.h>
#include <noa/cpu/fft/Transforms.h>
#include <noa/cpu/fft/Remap.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::transform::fft::apply2D()", "[assets][noa][cpu][transform]") {
    path_t path_base = test::PATH_NOA_DATA / "transform" / "fft";
    const YAML::Node& tests = YAML::LoadFile(path_base / "tests.yaml")["apply2D"]["tests"];
    io::ImageFile file;

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
        cpu::memory::PtrHost<float> input(elements(shape));
        file.readAll(input.get(), false);
        file.close();

        // Go to Fourier space:
        cpu::memory::PtrHost<cfloat_t> input_fft(elementsFFT(shape));
        cpu::fft::r2c(input.get(), input_fft.get(), shape, 1);
        const auto weight = 1.f / static_cast<float>(input.elements());
        cpu::math::multiplyByValue(input_fft.get(), math::sqrt(weight), input_fft.get(), input_fft.elements());

        // Apply new geometry:
        size2_t shape_2d = {shape.x, shape.y};
        cpu::memory::PtrHost<cfloat_t> input_fft_centered(input_fft.elements());
        cpu::memory::PtrHost<cfloat_t> output_fft(input_fft.elements());
        cpu::transform::fft::shift2D<fft::H2HC>(
                input_fft.get(), input_fft_centered.get(), shape_2d, -center, 1);
        cpu::transform::fft::apply2D<fft::HC2H>(
                input_fft_centered.get(), output_fft.get(), shape_2d, matrix, center + shift, cutoff, interp);

        // Go back to real space:
        cpu::fft::c2r(output_fft.get(), input.get(), shape, 1);
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

TEMPLATE_TEST_CASE("cpu::transform::fft::apply2D(), remap", "[noa][cpu][transform]", float, double) {
    const size_t batches = 3;
    const size3_t shape = {255, 256, 1};
    const size3_t pitch = shapeFFT(shape);
    const float cutoff = 0.5;
    const auto interp = INTERP_LINEAR;

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

    using complex_t = Complex<TestType>;
    cpu::memory::PtrHost<complex_t> input(elementsFFT(shape) * batches);
    test::randomize(input.get(), input.elements(), randomizer);

    cpu::Stream stream;
    size2_t shape_2d = {shape.x, shape.y};
    cpu::memory::PtrHost<complex_t> output_fft(input.elements());
    cpu::transform::fft::apply2D<fft::HC2H>(
            input.get(), output_fft.get(), shape_2d,
            transforms.get(), shifts.get(), batches, cutoff, interp);

    cpu::memory::PtrHost<complex_t> output_fft_centered(input.elements());
    cpu::transform::fft::apply2D<fft::HC2HC>(
            input.get(), output_fft_centered.get(), shape_2d,
            transforms.get(), shifts.get(), batches, cutoff, interp);
    cpu::fft::remap(fft::HC2H, output_fft_centered.get(), pitch, input.get(), pitch, shape, batches, stream);

    test::Matcher<complex_t> matcher(test::MATCH_ABS, input.get(), output_fft.get(), input.elements(), 1e-7);
    REQUIRE(matcher);
}

//TEST_CASE("cpu::transform::fft::apply2D(), symmetry", "[assets][noa][cpu][transform]") {
//    path_t path_base = test::PATH_NOA_DATA / "transform" / "fft";
//    const YAML::Node& tests = YAML::LoadFile(path_base / "tests.yaml")["apply2D_sym"]["tests"];
//
//    io::ImageFile file;
//    for (size_t i = 0; i < tests.size(); ++i) {
//        INFO("test: " << i);
//        const YAML::Node& test = tests[i];
//        const auto path_input = path_base / test["input"].as<path_t>();
//        const auto path_expected = path_base / test["expected"].as<path_t>();
//        const auto scale = test["scale"].as<float2_t>();
//        const auto rotate = math::toRad(test["rotate"].as<float>());
//        const transform::Symmetry sym(test["symmetry"].as<std::string>());
//        const auto center = test["center"].as<float2_t>();
//        const auto shift = test["shift"].as<float2_t>();
//        const auto cutoff = test["cutoff"].as<float>();
//        const auto interp = test["interp"].as<InterpMode>();
//        const auto normalize = test["normalize"].as<bool>();
//
//        float22_t matrix(transform::rotate(rotate) *
//                         transform::scale(1 / scale));
//        matrix = math::inverse(matrix);
//
//        // Load input:
//        file.open(path_input, io::READ);
//        const size3_t shape = file.shape();
//        cpu::memory::PtrHost<float> input(elements(shape));
//        file.readAll(input.get(), false);
//
//        // Go to Fourier space:
//        cpu::memory::PtrHost<cfloat_t> input_fft(elementsFFT(shape));
//        cpu::fft::r2c(input.get(), input_fft.get(), shape, 1);
//        const auto weight = 1.f / static_cast<float>(input.elements());
//        cpu::math::multiplyByValue(input_fft.get(), math::sqrt(weight), input_fft.get(), input_fft.elements());
//
//        // Apply new geometry:
//        size2_t shape_2d = {shape.x, shape.y};
//        cpu::memory::PtrHost<cfloat_t> input_fft_centered(input_fft.elements());
//        cpu::memory::PtrHost<cfloat_t> output_fft(input_fft.elements());
//        cpu::transform::fft::shift2D<fft::H2HC>(
//                input_fft.get(), input_fft_centered.get(), shape_2d, -center, 1);
//        cpu::transform::fft::apply2D<fft::HC2H>(
//                input_fft_centered.get(), output_fft.get(), shape_2d,
//                matrix, sym, center + shift, cutoff, interp, normalize);
//
//        // Go back to real space:
//        cpu::fft::c2r(output_fft.get(), input.get(), shape, 1);
//        cpu::math::multiplyByValue(input.get(), math::sqrt(weight), input.get(), input.elements());
//
//        // Load excepted and compare
//        cpu::memory::PtrHost<float> expected(input.elements());
//        file.open(path_expected, io::READ);
//        file.readAll(expected.get());
//        test::Matcher<float> matcher(test::MATCH_ABS, input.get(), expected.get(), input.elements(), 1e-6);
//        REQUIRE(matcher);
//    }
//}

TEST_CASE("cpu::transform::fft::apply3D()", "[assets][noa][cpu][transform]") {
    path_t path_base = test::PATH_NOA_DATA / "transform" / "fft";
    const YAML::Node& tests = YAML::LoadFile(path_base / "tests.yaml")["apply3D"]["tests"];

    io::ImageFile file;
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
        cpu::memory::PtrHost<float> input(elements(shape));
        file.readAll(input.get(), false);

        // Go to Fourier space:
        cpu::memory::PtrHost<cfloat_t> input_fft(elementsFFT(shape));
        cpu::fft::r2c(input.get(), input_fft.get(), shape, 1);
        const auto weight = 1.f / static_cast<float>(input.elements());
        cpu::math::multiplyByValue(input_fft.get(), math::sqrt(weight), input_fft.get(), input_fft.elements());

        // Apply new geometry:
        cpu::memory::PtrHost<cfloat_t> input_fft_centered(input_fft.elements());
        cpu::memory::PtrHost<cfloat_t> output_fft(input_fft.elements());
        cpu::transform::fft::shift3D<fft::H2HC>(
                input_fft.get(), input_fft_centered.get(), shape, -center, 1);
        cpu::transform::fft::apply3D<fft::HC2H>(
                input_fft_centered.get(), output_fft.get(), shape, matrix, center + shift, cutoff, interp);

        // Go back to real space:
        cpu::fft::c2r(output_fft.get(), input.get(), shape, 1);
        cpu::math::multiplyByValue(input.get(), math::sqrt(weight), input.get(), input.elements());

        // Load excepted and compare
        cpu::memory::PtrHost<float> expected(input.elements());
        file.open(path_expected, io::READ);
        file.readAll(expected.get());
        test::Matcher<float> matcher(test::MATCH_ABS, input.get(), expected.get(), input.elements(), 5e-6);
        REQUIRE(matcher);
    }
}

TEMPLATE_TEST_CASE("cpu::transform::fft::apply3D(), remap", "[noa][cpu][transform]", float, double) {
    const size_t batches = 3;
    const size3_t shape = {160, 161, 160};
    const size3_t pitch = shapeFFT(shape);
    const float cutoff = 0.5;
    const auto interp = INTERP_LINEAR;

    test::Randomizer<float> randomizer(-3, 3);
    cpu::memory::PtrHost<float33_t> transforms(batches);
    cpu::memory::PtrHost<float3_t> shifts(batches);
    for (size_t batch = 0; batch < batches; ++batch) {
        const float3_t scale = {0.9, 1.1, 1};
        float33_t matrix(transform::toMatrix(float3_t{randomizer.get(), randomizer.get(), randomizer.get()}) *
                         transform::scale(1 / scale));
        transforms[batch] = math::inverse(matrix);
        shifts[batch] = {randomizer.get() * 10, randomizer.get() * 10, randomizer.get() * 10};
    }

    using complex_t = Complex<TestType>;
    cpu::memory::PtrHost<complex_t> input(elementsFFT(shape) * batches);
    test::randomize(input.get(), input.elements(), randomizer);

    cpu::Stream stream;
    cpu::memory::PtrHost<complex_t> output_fft(input.elements());
    cpu::transform::fft::apply3D<fft::HC2H>(
            input.get(), output_fft.get(), shape,
            transforms.get(), shifts.get(), batches, cutoff, interp);

    cpu::memory::PtrHost<complex_t> output_fft_centered(input.elements());
    cpu::transform::fft::apply3D<fft::HC2HC>(
            input.get(), output_fft_centered.get(), shape,
            transforms.get(), shifts.get(), batches, cutoff, interp);
    cpu::fft::remap(fft::HC2H, output_fft_centered.get(), pitch, input.get(), pitch, shape, batches, stream);

    test::Matcher<complex_t> matcher(test::MATCH_ABS, input.get(), output_fft.get(), input.elements(), 1e-7);
    REQUIRE(matcher);
}
