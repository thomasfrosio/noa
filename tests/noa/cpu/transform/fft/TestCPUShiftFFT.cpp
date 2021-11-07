#include <noa/common/io/ImageFile.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/fft/Remap.h>
#include <noa/cpu/transform/fft/Shift.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::transform::fft::shift2D()", "[assets][noa][cpu][transform]") {
    io::ImageFile file;
    path_t path_base = test::PATH_TEST_DATA / "transform" / "fft";
    YAML::Node param = YAML::LoadFile(path_base / "param.yaml")["shift"]["2D"];

    const auto shape = param["shape"].as<size2_t>();
    const auto shift = param["shift"].as<float2_t>();
    const auto path_output = path_base / param["output"].as<path_t>(); // these are non-redundant non-centered
    const auto path_input = path_base / param["input"].as<path_t>();

    cpu::memory::PtrHost<cfloat_t> input(elementsFFT(shape));
    file.open(path_input, io::READ);
    file.readAll(input.get(), false);
    cpu::transform::fft::shift2D<fft::H2H>(input.get(), input.get(), shape, shift, 1); // in-place if no remap is OK

    cpu::memory::PtrHost<cfloat_t> expected(input.elements());
    file.open(path_output, io::READ);
    file.readAll(expected.get(), false);
    cfloat_t diff = test::getAverageDifference(expected.get(), input.get(), expected.elements());
    REQUIRE_THAT(diff, test::WithinAbs(cfloat_t{0}, 1e-5));
}

TEMPLATE_TEST_CASE("cpu::transform::fft::shift2D(), h2hc", "[noa][cpu][transform]", cfloat_t, cdouble_t) {
    const size3_t shape = test::getRandomShape(2, true);
    const float2_t shift = {31.5, -15.2};

    test::RealRandomizer<TestType> randomizer(-1., 2.);
    cpu::memory::PtrHost<TestType> input(elementsFFT(shape));
    cpu::memory::PtrHost<TestType> output(input.elements());
    test::initDataRandom(input.get(), input.elements(), randomizer);

    cpu::transform::fft::shift2D<fft::H2H>(input.get(), output.get(), {shape.x, shape.y}, shift, 1);
    cpu::fft::remap(fft::H2HC, output.get(), output.get(), shape, 1);

    cpu::memory::PtrHost<TestType> output_centered(input.elements());
    cpu::transform::fft::shift2D<fft::H2HC>(input.get(), output_centered.get(), {shape.x, shape.y}, shift, 1);

    TestType diff = test::getAverageDifference(output.get(), output_centered.get(), output.elements());
    REQUIRE_THAT(diff, test::WithinAbs(TestType{0}, 1e-5));
}

TEMPLATE_TEST_CASE("cpu::transform::fft::shift2D(), hc2h", "[noa][cpu][transform]", cfloat_t, cdouble_t) {
    const size3_t shape = test::getRandomShape(2, true);
    const float2_t shift = {31.5, -15.2};

    test::RealRandomizer<TestType> randomizer(-1., 2.);
    cpu::memory::PtrHost<TestType> input(elementsFFT(shape));
    cpu::memory::PtrHost<TestType> output(input.elements());
    test::initDataRandom(input.get(), input.elements(), randomizer);

    cpu::transform::fft::shift2D<fft::H2H>(input.get(), output.get(), {shape.x, shape.y}, shift, 1);

    cpu::memory::PtrHost<TestType> output_2(input.elements());
    cpu::fft::remap(fft::H2HC, input.get(), input.get(), shape, 1);
    cpu::transform::fft::shift2D<fft::HC2H>(input.get(), output_2.get(), {shape.x, shape.y}, shift, 1);

    TestType diff = test::getAverageDifference(output.get(), output_2.get(), output.elements());
    REQUIRE_THAT(diff, test::WithinAbs(TestType{0}, 1e-5));
}

TEST_CASE("cpu::transform::fft::shift3D()", "[assets][noa][cpu][transform]") {
    io::ImageFile file;
    path_t path_base = test::PATH_TEST_DATA / "transform" / "fft";
    YAML::Node param = YAML::LoadFile(path_base / "param.yaml")["shift"]["3D"];

    const auto shape = param["shape"].as<size3_t>();
    const auto shift = param["shift"].as<float3_t>();
    const auto path_output = path_base / param["output"].as<path_t>(); // these are non-redundant non-centered
    const auto path_input = path_base / param["input"].as<path_t>();

    cpu::memory::PtrHost<cfloat_t> input(elementsFFT(shape));
    file.open(path_input, io::READ);
    file.readAll(input.get(), false);
    cpu::transform::fft::shift3D<fft::H2H>(input.get(), input.get(), shape, shift, 1); // in-place if no remap is OK

    cpu::memory::PtrHost<cfloat_t> expected(input.elements());
    file.open(path_output, io::READ);
    file.readAll(expected.get(), false);
    cfloat_t diff = test::getAverageDifference(expected.get(), input.get(), expected.elements());
    REQUIRE_THAT(diff, test::WithinAbs(cfloat_t{0}, 1e-5));
}

TEMPLATE_TEST_CASE("cpu::transform::fft::shift3D(), h2hc", "[noa][cpu][transform]", cfloat_t, cdouble_t) {
    const size3_t shape = test::getRandomShape(3, true);
    const float3_t shift = {31.5, -15.2, 25.8};

    test::RealRandomizer<TestType> randomizer(-1., 2.);
    cpu::memory::PtrHost<TestType> input(elementsFFT(shape));
    cpu::memory::PtrHost<TestType> output(input.elements());
    test::initDataRandom(input.get(), input.elements(), randomizer);

    cpu::transform::fft::shift3D<fft::H2H>(input.get(), output.get(), shape, shift, 1);
    cpu::fft::remap(fft::H2HC, output.get(), output.get(), shape, 1);

    cpu::memory::PtrHost<TestType> output_centered(input.elements());
    cpu::transform::fft::shift3D<fft::H2HC>(input.get(), output_centered.get(), shape, shift, 1);

    TestType diff = test::getAverageDifference(output.get(), output_centered.get(), output.elements());
    REQUIRE_THAT(diff, test::WithinAbs(TestType{0}, 1e-5));
}

TEMPLATE_TEST_CASE("cpu::transform::fft::shift3D(), hc2h", "[noa][cpu][transform]", cfloat_t, cdouble_t) {
    const size3_t shape = test::getRandomShape(2, true);
    const float3_t shift = {31.5, -15.2, 25.8};

    test::RealRandomizer<TestType> randomizer(-1., 2.);
    cpu::memory::PtrHost<TestType> input(elementsFFT(shape));
    cpu::memory::PtrHost<TestType> output(input.elements());
    test::initDataRandom(input.get(), input.elements(), randomizer);

    cpu::transform::fft::shift3D<fft::H2H>(input.get(), output.get(), shape, shift, 1);

    cpu::memory::PtrHost<TestType> output_2(input.elements());
    cpu::fft::remap(fft::H2HC, input.get(), input.get(), shape, 1);
    cpu::transform::fft::shift3D<fft::HC2H>(input.get(), output_2.get(), shape, shift, 1);

    TestType diff = test::getAverageDifference(output.get(), output_2.get(), output.elements());
    REQUIRE_THAT(diff, test::WithinAbs(TestType{0}, 1e-5));
}