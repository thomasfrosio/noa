#include <noa/common/io/ImageFile.h>
#include <noa/common/transform/Euler.h>
#include <noa/common/transform/Geometry.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/memory/Set.h>

#include <noa/cpu/math/Arithmetics.h>
#include <noa/cpu/math/Generics.h>
#include <noa/cpu/math/Reductions.h>

#include <noa/cpu/filter/Shape.h>
#include <noa/cpu/transform/fft/Apply.h>
#include <noa/cpu/transform/fft/Shift.h>
#include <noa/cpu/fft/Transforms.h>
#include <noa/cpu/fft/Remap.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::transform::fft::apply2D()", "[assets][noa][cpu][transform]") {
    io::ImageFile file;
    path_t path_base = test::PATH_TEST_DATA / "transform" / "fft";
    const YAML::Node& tests = YAML::LoadFile(path_base / "param.yaml")["apply2D"]["tests"];

    for (size_t i = 0; i < tests.size(); ++i) {
        const YAML::Node& test = tests[i];
        const auto path_input = path_base / test["input"].as<path_t>();
        const auto path_expected = path_base / test["expected"].as<path_t>();
        const auto rotate = math::toRad(test["rotate"].as<float>());
        const auto scale = test["scale"].as<float2_t>();
        const auto center = test["center"].as<float2_t>();
        const auto shift = test["shift"].as<float2_t>();
        const auto max_frequency = test["max_frequency"].as<float>();
        const auto interp = test["interp"].as<InterpMode>();

        float22_t matrix(transform::rotate(rotate) *
                         transform::scale(scale));
        matrix = math::inverse(matrix);

        // Load input:
        file.open(path_input, io::READ);
        const size3_t shape = file.shape();
        cpu::memory::PtrHost<float> input(elements(shape));
        file.readAll(input.get(), false);

        // Go to Fourier space:
        cpu::memory::PtrHost<cfloat_t> input_fft(elementsFFT(shape));
        cpu::fft::r2c(input.get(), input_fft.get(), shape, 1);
//        const auto weight = 1.f / static_cast<float>(input.elements());
//        cpu::math::multiplyByValue(input_fft.get(), weight, input_fft.get(), input_fft.elements());

        // Rotate:
        cpu::memory::PtrHost<cfloat_t> input_fft_centered(input_fft.elements());
        cpu::memory::PtrHost<cfloat_t> output_fft(input_fft.elements());
        cpu::transform::fft::shift2D<fft::H2HC>(input_fft.get(),
                                                input_fft_centered.get(),
                                                {shape.x, shape.y}, -center, 1);
        cpu::transform::fft::apply2D<fft::HC2H>(input_fft_centered.get(), output_fft.get(), {shape.x, shape.y},
                                                matrix, center + shift, max_frequency, interp);

        cpu::memory::PtrHost<cfloat_t> fft_full(input.elements());
        cpu::memory::PtrHost<cfloat_t> fft_full_centered(input.elements());
        cpu::fft::remap(fft::H2F, output_fft.get(), fft_full.get(), shape, 1);
        cpu::fft::remap(fft::F2FC, fft_full.get(), fft_full_centered.get(), shape, 1);
        cpu::math::real(fft_full_centered.get(), input.get(), input.elements());
        file.open(path_base / string::format("test_noa_real_fft_{}.mrc", i), io::WRITE);
        file.shape(shape);
        file.writeAll(input.get(), false);

//        // Go back to real space:
//        cpu::fft::c2r(output_fft.get(), input.get(), shape, 1);
//
//        // Load excepted and compare
//        cpu::memory::PtrHost<float> expected(input.elements());
//        file.open(path_expected, io::READ);
//        file.writeAll(expected.get());
    }



//    test::IntRandomizer<size_t> randomizer_size(256, 512);
//    size3_t shape = {512,512, 1};//{randomizer_size.get(), randomizer_size.get(), 1};
//    size_t elements = noa::elements(shape);
//    size3_t shape_fft = shapeFFT(shape);
//    size_t elements_fft = noa::elements(shape_fft);
//
//    test::RealRandomizer<float> randomizer(0.f, 0.5f);
//    float2_t scale = {1,1};//{1.f + randomizer.get(), 1.f + randomizer.get()};
//    float rotate = 6 * randomizer.get();
//    float2_t rotation_center = {shape.x / 2, shape.y / 2};
//    float2_t shift = {30,30};//{20 * randomizer.get(), 20 * randomizer.get()};
//    float max_frequency = 1.f;//randomizer.get();
//    InterpMode interp = GENERATE(INTERP_LINEAR);
//
//    float22_t matrix(transform::rotate(rotate) *
//                     transform::scale(scale));
//    matrix = math::inverse(matrix);
//
//    // Generate input:
//    float radius = 50;
//    cpu::memory::PtrHost<float> input(elements);
//    test::RealRandomizer<float> randomizer_data(-5.f, 5.f);
//    test::initDataRandom(input.get(), elements, randomizer_data);
//    cpu::filter::rectangle2D(input.get(), input.get(), {shape.x, shape.y}, 1,
//                             rotation_center, {radius, radius}, 8);
//    file.open(path_base / "test_input.mrc", io::WRITE);
//    file.shape(shape);
//    file.writeAll(input.get());
//    file.close();


}

//TEST_CASE("cpu::transform::fft::apply2D() 2", "[assets][noa][cpu][transform]") {
//    path_t path_base = test::PATH_TEST_DATA / "transform" / "fft";
//    YAML::Node param = YAML::LoadFile(path_base / "param.yaml")["apply2D"];
//
//    io::ImageFile file;
//    for (size_t nb = 0; nb < 1; ++nb) {
//        INFO("test number = " << nb);
//
//        const YAML::Node& test = param["tests"][nb];
//        auto input_filename = path_base / test["input"].as<path_t>();
////        auto expected_filename = path_base / test["expected"].as<path_t>();
//        auto scale = test["scale"].as<float2_t>();
//        auto rotate = math::toRad(test["rotate"].as<float>());
//        auto shift = test["shift"].as<float2_t>();
//        auto max_frequency = test["max_frequency"].as<float>();
//        auto interp = test["interp"].as<InterpMode>();
//
//        float22_t matrix(transform::rotate(rotate) *
//                         transform::scale(scale));
//        matrix = math::inverse(matrix);
//        // Get input.
//        file.open(input_filename, io::READ);
//        size3_t shape = file.shape();
//        size_t elements = noa::elements(shape);
//        cpu::memory::PtrHost<float> input(elements);
//        file.readAll(input.get());
//
//        // Compute FFT
//        size3_t shape_fft = shapeFFT(shape);
//        size_t elements_fft = noa::elements(shape_fft);
//        cpu::memory::PtrHost<float> input_fft(elements_fft);
//
//        cpu::memory::set(input_fft.get(), elements_fft, 1.f);
//
//        cpu::memory::PtrHost<float> output_fft(elements_fft);
//        cpu::transform::fft::apply2D(input_fft.get(), output_fft.get(), {shape.x, shape.y},
//                                     {}, {}, max_frequency, interp);
//
//        file.open(path_base / "test_real.mrc", io::WRITE);
//        file.shape(shape_fft);
//        file.writeAll(output_fft.get());
//    }
//}
