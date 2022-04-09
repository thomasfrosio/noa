#include <noa/common/io/ImageFile.h>

#include <noa/cpu/memory/PtrHost.h>
#include <noa/cpu/fft/Remap.h>
#include <noa/cpu/geometry/fft/Shift.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace ::noa;

TEST_CASE("cpu::geometry::fft::shift2D()", "[assets][noa][cpu][geometry]") {
    io::ImageFile file;
    const path_t path_base = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node params = YAML::LoadFile(path_base / "tests.yaml")["shift"]["2D"];
    cpu::Stream stream(cpu::Stream::DEFAULT);

    for (size_t i = 0; i < params.size(); i++) {
        const YAML::Node& param = params[i];
        const auto shape = param["shape"].as<size4_t>();
        const auto shift = param["shift"].as<float2_t>();
        const auto cutoff = param["cutoff"].as<float>();
        const auto path_output = path_base / param["output"].as<path_t>(); // these are non-redundant non-centered
        const auto path_input = path_base / param["input"].as<path_t>();
        const size4_t stride = shape.fft().stride();
        const size_t elements = stride[0] * shape[0];

        cpu::memory::PtrHost<cfloat_t> input(elements);
        cpu::memory::PtrHost<cfloat_t> expected(elements);

        if (path_input.filename().empty()) {
            cpu::geometry::fft::shift2D<fft::H2H, cfloat_t>(
                    nullptr, {}, input.share(), stride, shape, shift, cutoff, stream);

            file.open(path_output, io::READ);
            file.readAll(expected.get(), false);

            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), input.get(), elements, 1e-4f));

        } else {
            file.open(path_input, io::READ);
            file.readAll(input.get(), false);
            cpu::geometry::fft::shift2D<fft::H2H, cfloat_t>(
                    input.share(), stride, input.share(), stride, shape, shift, cutoff, stream);

            file.open(path_output, io::READ);
            file.readAll(expected.get(), false);

            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), input.get(), elements, 1e-4f));
        }
    }
}

TEMPLATE_TEST_CASE("cpu::geometry::fft::shift2D(), h2hc", "[noa][cpu][geometry]", cfloat_t, cdouble_t) {
    const size4_t shape = test::getRandomShapeBatched(2, true); // even for inplace remap
    const size4_t stride = shape.fft().stride();
    const float2_t shift = {31.5, -15.2};
    const float cutoff = 0.5f;

    test::Randomizer<TestType> randomizer(-1., 2.);
    cpu::memory::PtrHost<TestType> input(stride[0] * shape[0]);
    cpu::memory::PtrHost<TestType> output(input.elements());
    test::randomize(input.get(), input.elements(), randomizer);

    cpu::Stream stream;
    cpu::geometry::fft::shift2D<fft::H2H, TestType>(
            input.share(), stride, output.share(), stride, shape, shift, cutoff, stream);
    cpu::fft::remap<TestType>(fft::H2HC, output.share(), stride, output.share(), stride, shape, stream);

    cpu::memory::PtrHost<TestType> output_centered(input.elements());
    cpu::geometry::fft::shift2D<fft::H2HC, TestType>(
            input.share(), stride, output_centered.share(), stride, shape, shift, cutoff, stream);
    stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS, output.get(), output_centered.get(), output.elements(), 1e-4f));
}

TEMPLATE_TEST_CASE("cpu::geometry::fft::shift2D(), hc2h", "[noa][cpu][geometry]", cfloat_t, cdouble_t) {
    const size4_t shape = test::getRandomShapeBatched(2, true); // even for inplace remap
    const size4_t stride = shape.fft().stride();
    const float2_t shift = {31.5, -15.2};
    const float cutoff = 0.5f;

    test::Randomizer<TestType> randomizer(-1., 2.);
    cpu::memory::PtrHost<TestType> input(stride[0] * shape[0]);
    cpu::memory::PtrHost<TestType> output(input.elements());
    test::randomize(input.get(), input.elements(), randomizer);

    cpu::Stream stream;
    cpu::geometry::fft::shift2D<fft::H2H, TestType>(
            input.share(), stride, output.share(), stride, shape, shift, cutoff, stream);

    cpu::memory::PtrHost<TestType> output_2(input.elements());
    cpu::fft::remap<TestType>(fft::H2HC, input.share(), stride, input.share(), stride, shape, stream);
    cpu::geometry::fft::shift2D<fft::HC2H, TestType>(
            input.share(), stride, output_2.share(), stride, shape, shift, cutoff, stream);
    stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS, output.get(), output_2.get(), output.elements(), 1e-4f));
}

TEST_CASE("cpu::geometry::fft::shift3D()", "[assets][noa][cpu][geometry]") {
    io::ImageFile file;
    const path_t path_base = test::NOA_DATA_PATH / "geometry" / "fft";
    const YAML::Node params = YAML::LoadFile(path_base / "tests.yaml")["shift"]["3D"];
    cpu::Stream stream(cpu::Stream::DEFAULT);

    for (size_t i = 0; i < params.size(); i++) {
        const YAML::Node& param = params[i];
        const auto shape = param["shape"].as<size4_t>();
        const auto shift = param["shift"].as<float3_t>();
        const auto cutoff = param["cutoff"].as<float>();
        const auto path_output = path_base / param["output"].as<path_t>(); // these are non-redundant non-centered
        const auto path_input = path_base / param["input"].as<path_t>();
        const size4_t stride = shape.fft().stride();
        const size_t elements = stride[0] * shape[0];

        cpu::memory::PtrHost<cfloat_t> input(elements);
        cpu::memory::PtrHost<cfloat_t> expected(elements);

        if (path_input.filename().empty()) {
            cpu::geometry::fft::shift3D<fft::H2H, cfloat_t>(
                    nullptr, {}, input.share(), stride, shape, shift, cutoff, stream);

            file.open(path_output, io::READ);
            file.readAll(expected.get(), false);

            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), input.get(), elements, 1e-4f));

        } else {
            file.open(path_input, io::READ);
            file.readAll(input.get(), false);
            cpu::geometry::fft::shift3D<fft::H2H, cfloat_t>(
                    input.share(), stride, input.share(), stride, shape, shift, cutoff, stream);

            file.open(path_output, io::READ);
            file.readAll(expected.get(), false);

            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), input.get(), elements, 1e-4f));
        }
    }
}

TEMPLATE_TEST_CASE("cpu::geometry::fft::shift3D(), h2hc", "[noa][cpu][geometry]", cfloat_t, cdouble_t) {
    const size4_t shape = test::getRandomShape(3, true);
    const size4_t stride = shape.fft().stride();
    const float3_t shift = {31.5, -15.2, 25.8};
    const float cutoff = 0.5f;

    test::Randomizer<TestType> randomizer(-1., 2.);
    cpu::memory::PtrHost<TestType> input(stride[0] * shape[0]);
    cpu::memory::PtrHost<TestType> output(input.elements());
    test::randomize(input.get(), input.elements(), randomizer);

    cpu::Stream stream;
    cpu::geometry::fft::shift3D<fft::H2H, TestType>(
            input.share(), stride, output.share(), stride, shape, shift, cutoff, stream);
    cpu::fft::remap<TestType>(fft::H2HC, output.share(), stride, output.share(), stride, shape, stream);

    cpu::memory::PtrHost<TestType> output_centered(input.elements());
    cpu::geometry::fft::shift3D<fft::H2HC, TestType>(
            input.share(), stride, output_centered.share(), stride, shape, shift, cutoff, stream);
    stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS, output.get(), output_centered.get(), output.elements(), 1e-4f));
}

TEMPLATE_TEST_CASE("cpu::geometry::fft::shift3D(), hc2h", "[noa][cpu][geometry]", cfloat_t, cdouble_t) {
    const size4_t shape = test::getRandomShape(3, true);
    const size4_t stride = shape.fft().stride();
    const float3_t shift = {31.5, -15.2, 25.8};
    const float cutoff = 0.5f;

    test::Randomizer<TestType> randomizer(-1., 2.);
    cpu::memory::PtrHost<TestType> input(stride[0] * shape[0]);
    cpu::memory::PtrHost<TestType> output(input.elements());
    test::randomize(input.get(), input.elements(), randomizer);

    cpu::Stream stream;
    cpu::geometry::fft::shift3D<fft::H2H, TestType>(
            input.share(), stride, output.share(), stride, shape, shift, cutoff, stream);

    cpu::memory::PtrHost<TestType> output_2(input.elements());
    cpu::fft::remap<TestType>(fft::H2HC, input.share(), stride, input.share(), stride, shape, stream);
    cpu::geometry::fft::shift3D<fft::HC2H, TestType>(
            input.share(), stride, output_2.share(), stride, shape, shift, cutoff, stream);
    stream.synchronize();

    REQUIRE(test::Matcher(test::MATCH_ABS, output.get(), output_2.get(), output.elements(), 1e-4f));
}
