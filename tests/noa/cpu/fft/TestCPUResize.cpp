#include <noa/common/io/ImageFile.h>
#include <noa/cpu/fft/Resize.h>
#include <noa/cpu/memory/PtrHost.h>

#include "Assets.h"
#include "Helpers.h"
#include <catch2/catch.hpp>

using namespace noa;

// These tests use that fact cropping after padding cancels the padding and returns the input array.
// These are not very good tests but it is better than nothing.
TEMPLATE_TEST_CASE("cpu::fft::resize()", "[noa][cpu][fft]", float, cfloat_t, double, cdouble_t) {
    test::Randomizer<TestType> randomizer(1., 5.);
    test::Randomizer<size_t> randomizer_int(0, 32);
    const uint ndim = GENERATE(1U, 2U, 3U);
    const size4_t shape = test::getRandomShapeBatched(ndim);
    size4_t shape_padded(shape);
    if (ndim > 2) shape_padded[1] += randomizer_int.get();
    if (ndim > 1) shape_padded[2] += randomizer_int.get();
    shape_padded[3] += randomizer_int.get();

    INFO(shape);
    INFO(shape_padded);
    cpu::Stream stream;

    AND_THEN("pad then crop") {
        const size4_t stride = shape.fft().stride();
        const size4_t stride_padded = shape_padded.fft().stride();
        const size_t elements = shape.fft().elements();
        const size_t elements_padded = shape_padded.fft().elements();
        cpu::memory::PtrHost<TestType> original(elements);
        cpu::memory::PtrHost<TestType> padded(elements_padded);
        cpu::memory::PtrHost<TestType> cropped(original.size());

        test::randomize(original.get(), original.elements(), randomizer);
        cpu::fft::resize<fft::H2H>(original.get(), stride, shape,
                                   padded.get(), stride_padded, shape_padded, stream);
        cpu::fft::resize<fft::H2H>(padded.get(), stride_padded, shape_padded,
                                   cropped.get(), stride, shape, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, original.get(), cropped.get(), original.elements(), 1e-10));
    }

    AND_THEN("padFull then cropFull") {
        const size4_t stride = shape.stride();
        const size4_t stride_padded = shape_padded.stride();
        const size_t elements = shape.elements();
        const size_t elements_padded = shape_padded.elements();
        cpu::memory::PtrHost<TestType> original(elements);
        cpu::memory::PtrHost<TestType> padded(elements_padded);
        cpu::memory::PtrHost<TestType> cropped(original.size());

        test::randomize(original.get(), original.elements(), randomizer);
        cpu::fft::resize<fft::F2F>(original.get(), stride, shape,
                                   padded.get(), stride_padded, shape_padded, stream);
        cpu::fft::resize<fft::F2F>(padded.get(), stride_padded, shape_padded,
                                   cropped.get(), stride, shape, stream);
        REQUIRE(test::Matcher(test::MATCH_ABS, original.get(), cropped.get(), original.elements(), 1e-10));
    }
}

TEST_CASE("cpu::fft::resize(), assets", "[assets][noa][cpu][fft]") {
    const fs::path path = test::NOA_DATA_PATH / "fft";
    YAML::Node tests = YAML::LoadFile(path / "tests.yaml")["resize"];
    io::ImageFile file;

    constexpr bool GENERATE_ASSETS = false;
    if constexpr (GENERATE_ASSETS) {
        test::Randomizer<float> randomizer(-128., 128.);

        for (const YAML::Node& node : tests["input"]) {
            const auto shape = node["shape"].as<size4_t>();
            const auto path_input = path / node["path"].as<path_t>();
            cpu::memory::PtrHost<float> input(shape.fft().elements());
            test::randomize(input.get(), input.elements(), randomizer);
            file.open(path_input, io::WRITE);
            file.shape(shape.fft());
            file.writeAll(input.get(), false);
        }
    }

    for (size_t i = 0; i < tests["tests"].size(); ++i) {
        const YAML::Node& test = tests["tests"][i];
        const auto filename_input = path / test["input"].as<path_t>();
        const auto filename_expected = path / test["expected"].as<path_t>();
        const auto shape_input = test["shape_input"].as<size4_t>();
        const auto shape_expected = test["shape_expected"].as<size4_t>();

        file.open(filename_input, io::READ);
        cpu::memory::PtrHost<float> input(shape_input.fft().elements());
        file.readAll(input.get(), false);

        cpu::memory::PtrHost<float> output(shape_expected.fft().elements());
        cpu::Stream stream;
        cpu::fft::resize<fft::H2H>(input.get(), shape_input.fft().stride(), shape_input,
                                   output.get(), shape_expected.fft().stride(), shape_expected, stream);

        if constexpr (GENERATE_ASSETS) {
            file.open(filename_expected, io::WRITE);
            file.shape(shape_expected.fft());
            file.writeAll(output.get(), false);
        } else {
            file.open(filename_expected, io::READ);
            cpu::memory::PtrHost<float> expected(file.shape().elements());
            file.readAll(expected.get(), false);
            REQUIRE(test::Matcher(test::MATCH_ABS, expected.get(), output.get(), expected.elements(), 1e-10));
        }
    }
}
