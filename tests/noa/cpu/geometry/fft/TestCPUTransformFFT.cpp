#include <noa/common/io/ImageFile.h>
#include <noa/common/geometry/Euler.h>
#include <noa/common/geometry/Transform.h>

#include <noa/cpu/math/Ewise.h>
#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/geometry/fft/Transform.h>
#include <noa/cpu/signal/fft/Shift.h>
#include <noa/cpu/fft/Transforms.h>
#include <noa/cpu/fft/Remap.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::geometry::fft::transform2D()", "[assets][noa][cpu][geometry]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node& tests = YAML::LoadFile(path_base / "tests.yaml")["transform2D"]["tests"];
    io::ImageFile file;
    cpu::Stream stream(cpu::Stream::DEFAULT);

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
        const size4_t stride_fft = shape_fft.strides();

        cpu::memory::PtrHost<float> input(shape.elements());
        file.readAll(input.get(), false);
        file.close();

        // Go to Fourier space:
        cpu::memory::PtrHost<cfloat_t> input_fft(shape_fft.elements());
        cpu::fft::r2c(input.share(), input_fft.share(), shape, cpu::fft::ESTIMATE, fft::NORM_ORTHO, stream);

        // Apply new geometry:
        cpu::memory::PtrHost<cfloat_t> input_fft_centered(input_fft.elements());
        cpu::memory::PtrHost<cfloat_t> output_fft(input_fft.elements());
        cpu::signal::fft::shift2D<fft::H2HC, cfloat_t>(
                input_fft.share(), stride_fft, input_fft_centered.share(), stride_fft, shape, -center, cutoff, stream);
        cpu::geometry::fft::transform2D<fft::HC2H, cfloat_t>(
                input_fft_centered.share(), stride_fft, output_fft.share(), stride_fft, shape,
                matrix, center + shift, cutoff, interp, stream);

        // Go back to real space:
        cpu::fft::c2r(output_fft.share(), input.share(), shape, cpu::fft::ESTIMATE, fft::NORM_ORTHO, stream);

        // Load excepted and compare
        cpu::memory::PtrHost<float> expected(input.elements());
        file.open(path_expected, io::READ);
        file.readAll(expected.get(), false);
        file.close();
        test::Matcher<float> matcher(test::MATCH_ABS, input.get(), expected.get(), input.elements(), 1e-4);
        REQUIRE(matcher);
    }
}

TEMPLATE_TEST_CASE("cpu::geometry::fft::transform2D(), remap", "[noa][cpu][geometry]", float, double) {
    const size4_t shape = test::getRandomShapeBatched(2);
    const size4_t stride = shape.fft().strides();
    const size_t elements = stride[0] * shape[0];
    const float cutoff = 0.5;
    const auto interp = INTERP_LINEAR;
    cpu::Stream stream(cpu::Stream::DEFAULT);

    test::Randomizer<float> randomizer(-3, 3);
    cpu::memory::PtrHost<float22_t> transforms(shape[0]);
    cpu::memory::PtrHost<float2_t> shifts(shape[0]);
    for (size_t batch = 0; batch < shape[0]; ++batch) {
        const float2_t scale = {0.9, 1.1};
        const float rotate = randomizer.get();
        float22_t matrix(geometry::rotate(rotate) *
                         geometry::scale(1 / scale));
        transforms[batch] = math::inverse(matrix);
        shifts[batch] = {randomizer.get() * 10, randomizer.get() * 10};
    }

    using complex_t = Complex<TestType>;
    cpu::memory::PtrHost<complex_t> input(elements);
    test::randomize(input.get(), input.elements(), randomizer);

    cpu::memory::PtrHost<complex_t> output_fft(elements);
    cpu::geometry::fft::transform2D<fft::HC2H, complex_t>(
            input.share(), stride, output_fft.share(), stride, shape,
            transforms.share(), shifts.share(), cutoff, interp, stream);

    cpu::memory::PtrHost<complex_t> output_fft_centered(elements);
    cpu::geometry::fft::transform2D<fft::HC2HC, complex_t>(
            input.share(), stride, output_fft_centered.share(), stride, shape,
            transforms.share(), shifts.share(), cutoff, interp, stream);
    cpu::fft::remap<complex_t>(fft::HC2H, output_fft_centered.share(), stride, input.share(), stride, shape, stream);

    test::Matcher<complex_t> matcher(test::MATCH_ABS, input.get(), output_fft.get(), elements, 1e-7);
    REQUIRE(matcher);
}

TEST_CASE("cpu::geometry::fft::transform3D()", "[assets][noa][cpu][geometry]") {
    const path_t path_base = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node& tests = YAML::LoadFile(path_base / "tests.yaml")["transform3D"]["tests"];

    io::ImageFile file;
    cpu::Stream stream(cpu::Stream::DEFAULT);
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
        const size4_t stride_fft = shape_fft.strides();

        cpu::memory::PtrHost<float> input(shape.elements());
        file.readAll(input.get(), false);

        // Go to Fourier space:
        cpu::memory::PtrHost<cfloat_t> input_fft(shape_fft.elements());
        cpu::fft::r2c(input.share(), input_fft.share(), shape, cpu::fft::ESTIMATE, fft::NORM_ORTHO, stream);

        // Apply new geometry:
        cpu::memory::PtrHost<cfloat_t> input_fft_centered(input_fft.elements());
        cpu::memory::PtrHost<cfloat_t> output_fft(input_fft.elements());
        cpu::signal::fft::shift3D<fft::H2HC, cfloat_t>(
                input_fft.share(), stride_fft, input_fft_centered.share(), stride_fft, shape, -center, cutoff, stream);
        cpu::geometry::fft::transform3D<fft::HC2H, cfloat_t>(
                input_fft_centered.share(), stride_fft, output_fft.share(), stride_fft, shape,
                matrix, center + shift, cutoff, interp, stream);

        // Go back to real space:
        cpu::fft::c2r(output_fft.share(), input.share(), shape, cpu::fft::ESTIMATE, fft::NORM_ORTHO, stream);

        // Load excepted and compare
        cpu::memory::PtrHost<float> expected(input.elements());
        file.open(path_expected, io::READ);
        file.readAll(expected.get());
        test::Matcher<float> matcher(test::MATCH_ABS, input.get(), expected.get(), input.elements(), 2e-4);
        REQUIRE(matcher);
    }
}

TEMPLATE_TEST_CASE("cpu::geometry::fft::transform3D(), remap", "[noa][cpu][geometry]", float, double) {
    const size4_t shape = test::getRandomShapeBatched(3);
    const size4_t stride = shape.fft().strides();
    const size_t elements = stride[0] * shape[0];
    const float cutoff = 0.5;
    const auto interp = INTERP_LINEAR;
    cpu::Stream stream(cpu::Stream::DEFAULT);

    test::Randomizer<float> randomizer(-3, 3);
    cpu::memory::PtrHost<float33_t> transforms(shape[0]);
    cpu::memory::PtrHost<float3_t> shifts(shape[0]);
    for (size_t batch = 0; batch < shape[0]; ++batch) {
        const float3_t scale = {0.9, 1.1, 1};
        float33_t matrix(geometry::euler2matrix(float3_t{randomizer.get(), randomizer.get(), randomizer.get()}) *
                         geometry::scale(1 / scale));
        transforms[batch] = math::inverse(matrix);
        shifts[batch] = {randomizer.get() * 10, randomizer.get() * 10, randomizer.get() * 10};
    }

    using complex_t = Complex<TestType>;
    cpu::memory::PtrHost<complex_t> input(elements);
    test::randomize(input.get(), input.elements(), randomizer);

    cpu::memory::PtrHost<complex_t> output_fft(elements);
    cpu::geometry::fft::transform3D<fft::HC2H, complex_t>(
            input.share(), stride, output_fft.share(), stride, shape,
            transforms.share(), shifts.share(), cutoff, interp, stream);

    cpu::memory::PtrHost<complex_t> output_fft_centered(elements);
    cpu::geometry::fft::transform3D<fft::HC2HC, complex_t>(
            input.share(), stride, output_fft_centered.share(), stride, shape,
            transforms.share(), shifts.share(), cutoff, interp, stream);
    cpu::fft::remap<complex_t>(fft::HC2H, output_fft_centered.share(), stride, input.share(), stride, shape, stream);

    test::Matcher<complex_t> matcher(test::MATCH_ABS, input.get(), output_fft.get(), elements, 1e-7);
    REQUIRE(matcher);
}
